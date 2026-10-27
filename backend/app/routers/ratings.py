"""The signed-in user's ratings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.deps import get_current_user
from backend.app.models import Movie, Rating, User, utcnow
from backend.app.schemas import RatingCreate, RatingOut, movie_to_out

router = APIRouter(prefix="/api/ratings", tags=["ratings"])


def _rating_out(rating: Rating, movie: Movie) -> RatingOut:
    return RatingOut(
        movie=movie_to_out(movie),
        rating=rating.rating,
        updated_at=rating.updated_at,
    )


@router.get("", response_model=list[RatingOut])
def list_ratings(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[RatingOut]:
    rows = db.execute(
        select(Rating, Movie)
        .join(Movie, Rating.movie_id == Movie.id)
        .where(Rating.user_id == user.id)
        .order_by(Rating.updated_at.desc(), Rating.id.desc())
    ).all()
    return [_rating_out(rating, movie) for rating, movie in rows]


@router.post("", response_model=RatingOut)
def upsert_rating(
    payload: RatingCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RatingOut:
    movie = db.get(Movie, payload.movie_id)
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")

    rating = db.scalar(
        select(Rating).where(
            Rating.user_id == user.id,
            Rating.movie_id == payload.movie_id,
        )
    )
    if rating is None:
        rating = Rating(user_id=user.id, movie_id=payload.movie_id, rating=payload.rating)
        db.add(rating)
    else:
        rating.rating = payload.rating
        rating.updated_at = utcnow()

    db.commit()
    db.refresh(rating)
    return _rating_out(rating, movie)


@router.delete("/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rating(
    movie_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    rating = db.scalar(
        select(Rating).where(Rating.user_id == user.id, Rating.movie_id == movie_id)
    )
    if rating is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rating not found.")
    db.delete(rating)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
