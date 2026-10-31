"""Catalog search and movie detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, func, literal, select, text
from sqlalchemy.orm import Session

from backend.app.db import get_db, trgm_available
from backend.app.deps import get_tmdb_client
from backend.app.ingest import ingest
from backend.app.models import Movie
from backend.app.schemas import (
    ExternalMovieOut,
    MovieOut,
    SearchResponse,
    external_to_out,
    movie_to_out,
    normalized_query_text,
)
from backend.app.tmdb import TmdbClient
from recsys.titles import split_genres

router = APIRouter(prefix="/api/movies", tags=["movies"])

# Below this many local matches a search is probably about a film the dataset predates, so TMDB is
# asked as well. The threshold keeps a plain "matrix" search from paying two round trips.
EXTERNAL_SEARCH_THRESHOLD = 5
EXTERNAL_SEARCH_LIMIT = 6


# Trigram fuzzing needs a query long enough to have a shape of its own; below this length it
# matches nearly every title and stops being a ranking signal.
FUZZY_MIN_LENGTH = 3
# A word-similarity score at or above this counts as a typo-tolerant hit. The ``<%`` operator reads
# a GUC, which the search sets with SET LOCAL so no other request inherits it.
FUZZY_THRESHOLD = 0.5


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def fuzzy_search_enabled(dialect: str, *, fuzzy: bool | None = None) -> bool:
    """Whether this request may use pg_trgm's fuzzy operators.

    ``fuzzy`` is an override so both branches can be tested without a live PostgreSQL server.
    """
    if fuzzy is not None:
        return fuzzy
    return dialect == "postgresql" and trgm_available()


def _contains_filter(statement: Select, normalized: str) -> Select:
    """The precise case: the title literally contains what was typed."""
    if not normalized:
        return statement
    return statement.where(Movie.title_search.like(f"%{_escape_like(normalized)}%", escape="\\"))


def _fuzzy_filter(statement: Select, normalized: str) -> Select:
    """Typo tolerance: ``<%`` is pg_trgm's word-similarity operator, served by the GIN index.

    This is a second query rather than an ``OR`` onto the precise one. On the live catalog the
    combined condition could not use the trigram index and cost ~180 ms per search instead of
    ~12 ms, and it also pulled loose matches into queries that already had exact answers.
    """
    if not normalized:
        return statement
    return statement.where(literal(normalized).op("<%")(Movie.title_search))


def fuzzy_escalation_needed(*, fuzzy: bool, normalized: str, found: int) -> bool:
    """Whether to fall back to approximate matching.

    Only a query that matched *nothing* is treated as a possible misspelling. Padding out a real
    answer with loosely similar titles is worse than a short list: on the live catalog
    "inception" matched one film, and escalation then added "Deception" and "The Exception"
    behind it. A short query has no trigram shape to match on, and a server without pg_trgm has no
    operator to run.
    """
    return bool(fuzzy and normalized and len(normalized) >= FUZZY_MIN_LENGTH and found == 0)


def _order_by_relevance(
    statement: Select, normalized: str, dialect: str, *, fuzzy: bool | None = None
) -> Select:
    """Rank matches: word-start tier, then popularity, then fuzzy closeness.

    Popularity outranks similarity inside a tier on purpose. Measured on the live catalog, plain
    ``similarity`` put "GodFather (1991)" (9 ratings, an exact trigram match) ahead of "The
    Godfather" (66,440 ratings) for the query "godfather", which is the wrong answer for anyone
    typing that query. Closeness still decides between titles of comparable standing.
    """
    if not normalized:
        return statement.order_by(Movie.rating_count.desc(), Movie.id)
    pattern = _escape_like(normalized)
    # A match at the start of a word beats one buried inside a longer word: searching "godfather"
    # must rank "The Godfather" ("the godfather") above "Tokyo Godfathers", not depend on length.
    word_start = Movie.title_search.like(f"{pattern}%", escape="\\") | Movie.title_search.like(
        f"% {pattern}%", escape="\\"
    )
    if fuzzy_search_enabled(dialect, fuzzy=fuzzy):
        return statement.order_by(
            word_start.desc(),
            Movie.rating_count.desc(),
            func.similarity(Movie.title_search, normalized).desc(),
            Movie.id,
        )
    # SQLite, and PostgreSQL without pg_trgm: no similarity(), so popularity decides.
    return statement.order_by(word_start.desc(), Movie.rating_count.desc(), Movie.id)


@router.get("/search", response_model=SearchResponse)
def search_movies(
    q: str = Query("", max_length=120, description="Partial title, case and punctuation free."),
    limit: int = Query(20, ge=1, le=50),
    include_external: bool = Query(
        True, description="Ask TMDB when the catalog has only a few matches."
    ),
    db: Session = Depends(get_db),
    tmdb: TmdbClient = Depends(get_tmdb_client),
) -> SearchResponse:
    normalized = normalized_query_text(q)
    dialect = db.get_bind().dialect.name
    fuzzy = fuzzy_search_enabled(dialect)

    statement = _order_by_relevance(
        _contains_filter(select(Movie), normalized), normalized, dialect, fuzzy=fuzzy
    )
    movies = list(db.scalars(statement.limit(limit)))

    # A misspelling gets exactly one second chance: when the precise query found nothing.
    if fuzzy_escalation_needed(fuzzy=fuzzy, normalized=normalized, found=len(movies)):
        # Both settings are transaction-scoped, so neither leaks to the next request that borrows
        # this pooled connection. The planner otherwise estimates the trigram index at cost 1680
        # against a 1457 sequential scan and takes the scan: measured on 87,585 rows, 153 ms
        # against 6 ms through the index. This statement only runs when the precise query matched
        # nothing, so the bitmap it selects is a small one.
        db.execute(text(f"SET LOCAL pg_trgm.word_similarity_threshold = {FUZZY_THRESHOLD}"))
        db.execute(text("SET LOCAL enable_seqscan = off"))
        fuzzy_statement = _order_by_relevance(
            _fuzzy_filter(select(Movie), normalized), normalized, dialect, fuzzy=True
        )
        movies = list(db.scalars(fuzzy_statement.limit(limit)))

    external: list[ExternalMovieOut] = []
    external_source: str | None = None
    if include_external and normalized and len(movies) < EXTERNAL_SEARCH_THRESHOLD:
        external, external_source = _external_movies(db, tmdb, q)

    return SearchResponse(
        query=q,
        normalized_query=normalized,
        count=len(movies),
        results=[movie_to_out(movie) for movie in movies],
        external=external,
        external_source=external_source,
    )


def _external_movies(
    db: Session, tmdb: TmdbClient, query: str
) -> tuple[list[ExternalMovieOut], str | None]:
    """TMDB hits the catalog does not already hold. Never writes anything."""
    results = tmdb.search(query, limit=EXTERNAL_SEARCH_LIMIT)
    if results is None:
        # Disabled, offline or rate-limited: the local results still stand.
        return [], None
    known = set(
        db.scalars(
            select(Movie.tmdb_id).where(Movie.tmdb_id.in_([r.tmdb_id for r in results]))
        ).all()
    )
    fresh = [external_to_out(result) for result in results if result.tmdb_id not in known]
    return fresh, "tmdb" if fresh else None


@router.get("/genres", response_model=list[str])
def list_genres(db: Session = Depends(get_db)) -> list[str]:
    """Every genre label present in the catalog, for the recommendation filter chips."""
    combinations = db.scalars(select(Movie.genres).distinct()).all()
    genres: set[str] = set()
    for combination in combinations:
        genres.update(split_genres(combination or ""))
    genres.discard("(no genres listed)")
    return sorted(genres)


def _enrich_from_tmdb(movie: Movie, tmdb: TmdbClient, db: Session) -> None:
    """Fill a missing poster or overview once, and never let TMDB break the request.

    Only called for a single movie at a time, so a cold cache costs one HTTP call and every later
    request is served from the local cache (or from the stored columns).
    """
    if not tmdb.enabled or movie.poster_path or movie.overview or not movie.tmdb_id:
        return
    fetched = tmdb.fetch(movie.tmdb_id)
    if fetched is None:
        return
    if fetched.poster_path:
        movie.poster_path = fetched.poster_path
    if fetched.overview:
        movie.overview = fetched.overview
    if fetched.poster_path or fetched.overview:
        db.commit()
        db.refresh(movie)


@router.get("/tmdb/{tmdb_id}", response_model=MovieOut)
def ingest_tmdb_movie(
    tmdb_id: int,
    db: Session = Depends(get_db),
    tmdb: TmdbClient = Depends(get_tmdb_client),
) -> MovieOut:
    """Open a film TMDB knows: add it to the catalog once, then serve it like any other.

    This is the *only* path that writes a searched film, so browsing cannot bloat the catalog. It
    is idempotent: opening the same film twice returns the same row.
    """
    if tmdb_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="tmdb_id must be positive"
        )
    if not tmdb.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TMDB is not configured on this server, so new films cannot be added.",
        )
    movie = ingest(db, tmdb, tmdb_id)
    if movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="TMDB could not supply that film right now; try again later.",
        )
    return movie_to_out(movie)


@router.get("/{movie_id}", response_model=MovieOut)
def read_movie(
    movie_id: int,
    db: Session = Depends(get_db),
    tmdb: TmdbClient = Depends(get_tmdb_client),
) -> MovieOut:
    movie = db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")
    _enrich_from_tmdb(movie, tmdb, db)
    return movie_to_out(movie)
