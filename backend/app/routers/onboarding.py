"""Cold-start onboarding: a diverse, popular shortlist to rate before the models have signal."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import Movie
from backend.app.onboarding import diverse_picks
from backend.app.schemas import OnboardingResponse, movie_to_out

router = APIRouter(prefix="/api", tags=["onboarding"])

TARGET_RATINGS = 5
CANDIDATE_POOL = 8


@router.get("/onboarding", response_model=OnboardingResponse)
def onboarding(
    limit: int = Query(12, ge=5, le=20),
    db: Session = Depends(get_db),
) -> OnboardingResponse:
    """Well-known films spread across genres and decades, for a user with no ratings yet."""
    candidates = list(
        db.scalars(
            select(Movie)
            .order_by(Movie.rating_count.desc(), Movie.id)
            .limit(limit * CANDIDATE_POOL)
        )
    )
    picks = diverse_picks(candidates, limit=limit)
    return OnboardingResponse(
        target=TARGET_RATINGS,
        movies=[movie_to_out(movie) for movie in picks],
    )
