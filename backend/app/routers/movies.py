"""Catalog search and movie detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.deps import get_tmdb_client
from backend.app.models import Movie
from backend.app.schemas import MovieOut, SearchResponse, movie_to_out, normalized_query_text
from backend.app.tmdb import TmdbClient
from recsys.titles import split_genres

router = APIRouter(prefix="/api/movies", tags=["movies"])


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
    db: Session = Depends(get_db),
) -> SearchResponse:
    normalized = normalized_query_text(q)
    statement = select(Movie)
    if normalized:
        pattern = f"%{_escape_like(normalized)}%"
        statement = statement.where(Movie.title_search.like(pattern, escape="\\"))

    dialect = db.get_bind().dialect.name
    statement = _order_by_relevance(statement, normalized, dialect)
    movies = list(db.scalars(statement.limit(limit)))

    return SearchResponse(
        query=q,
        normalized_query=normalized,
        count=len(movies),
        results=[movie_to_out(movie) for movie in movies],
    )


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
