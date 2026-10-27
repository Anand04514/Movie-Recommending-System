"""Personalized recommendations for the signed-in user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.deps import get_current_user
from backend.app.models import Movie, Rating, User
from backend.app.recommendations import (
    ArtifactsUnavailable,
    RecommendationService,
    get_recommender,
)
from backend.app.schemas import RecommendationOut, RecommendationResponse, movie_to_out

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.get("/recommend", response_model=RecommendationResponse)
def recommend(
    limit: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommender),
) -> RecommendationResponse:
    ratings = db.execute(
        select(Rating.movie_id, Rating.rating).where(Rating.user_id == user.id)
    ).all()
    try:
        batch = service.recommend(
            [(int(movie_id), float(value)) for movie_id, value in ratings], limit=limit
        )
    except ArtifactsUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    wanted = {item.movie_id for item in batch.items}
    wanted |= {
        item.source_movie_id for item in batch.items if item.source_movie_id is not None
    }
    movies = (
        {movie.id: movie for movie in db.scalars(select(Movie).where(Movie.id.in_(wanted)))}
        if wanted
        else {}
    )

    recommendations: list[RecommendationOut] = []
    for item in batch.items:
        movie = movies.get(item.movie_id)
        if movie is None:
            # The catalog may be older than the trained artifacts; skip rather than 500.
            continue
        if item.source_movie_id is not None and item.source_rating is not None:
            source = movies.get(item.source_movie_id)
            source_title = source.title_display if source else "a movie you rated"
            reason = f"Because you rated {source_title} {item.source_rating:g}★"
        else:
            reason = "Popular with MovieLens users"
        recommendations.append(
            RecommendationOut(
                movie=movie_to_out(movie),
                score=round(item.score, 4),
                predicted_rating=round(item.predicted_rating, 2),
                reason=reason,
            )
        )

    return RecommendationResponse(
        algorithm=batch.algorithm,
        cold_start=batch.cold_start,
        recommendations=recommendations,
    )
