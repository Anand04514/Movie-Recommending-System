"""Catalog search and movie detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import Movie
from backend.app.schemas import MovieOut, SearchResponse, movie_to_out, normalized_query_text

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


@router.get("/{movie_id}", response_model=MovieOut)
def read_movie(movie_id: int, db: Session = Depends(get_db)) -> MovieOut:
    movie = db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")
    return movie_to_out(movie)
