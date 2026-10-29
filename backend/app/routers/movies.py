"""Catalog search and movie detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from backend.app.db import get_db
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


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _order_by_relevance(statement: Select, normalized: str, dialect: str) -> Select:
    if not normalized:
        return statement.order_by(Movie.rating_count.desc(), Movie.id)
    pattern = _escape_like(normalized)
    # A match at the start of a word beats one buried inside a longer word: searching "godfather"
    # must rank "The Godfather" ("the godfather") by popularity rather than behind
    # "GodFather (1991)" just because the normalized title starts with "the".
    word_start = Movie.title_search.like(f"{pattern}%", escape="\\") | Movie.title_search.like(
        f"% {pattern}%", escape="\\"
    )
    if dialect == "postgresql":
        # pg_trgm ranks close spellings inside a tier; popularity breaks ties.
        return statement.order_by(
            word_start.desc(),
            func.similarity(Movie.title_search, normalized).desc(),
            Movie.rating_count.desc(),
            Movie.id,
        )
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
    statement = select(Movie)
    if normalized:
        pattern = f"%{_escape_like(normalized)}%"
        statement = statement.where(Movie.title_search.like(pattern, escape="\\"))

    dialect = db.get_bind().dialect.name
    statement = _order_by_relevance(statement, normalized, dialect)
    movies = list(db.scalars(statement.limit(limit)))

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
