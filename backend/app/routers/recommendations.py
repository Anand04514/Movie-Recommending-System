"""Recommendations: filters, explanations, algorithm comparison and evaluation history."""

from __future__ import annotations

import statistics

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.deps import get_current_user
from backend.app.models import ModelRun, Movie, Rating, User
from backend.app.recommendations import (
    ArtifactsUnavailable,
    RecommendationBatch,
    RecommendationService,
    UserCfService,
    get_recommender,
    get_user_cf_recommender,
)
from backend.app.schemas import (
    AlgorithmComparison,
    ComparisonResponse,
    ModelRunOut,
    RecommendationEvidence,
    RecommendationFiltersApplied,
    RecommendationOut,
    RecommendationResponse,
    model_run_to_out,
    movie_to_out,
)
from recsys.titles import split_genres

router = APIRouter(prefix="/api", tags=["recommendations"])

SUPPORTED_ALGORITHMS = ("item_cf", "user_cf")
# With filters on, score a wider pool so the list can still be filled with matching films.
FILTERED_POOL_LIMIT = 200
MAX_EVIDENCE = 3


def _parse_genres(raw: str) -> list[str]:
    return [genre.strip() for genre in raw.split(",") if genre.strip()]


def _matches(
    movie: Movie,
    *,
    genres: list[str],
    year_min: int | None,
    year_max: int | None,
    min_score: float | None,
    predicted: float,
) -> bool:
    if genres:
        available = {genre.lower() for genre in split_genres(movie.genres)}
        if not {genre.lower() for genre in genres} & available:
            return False
    if year_min is not None and (movie.release_year is None or movie.release_year < year_min):
        return False
    if year_max is not None and (movie.release_year is None or movie.release_year > year_max):
        return False
    if min_score is not None and predicted < min_score:
        return False
    return True


def _shared_genres(source: Movie, target: Movie) -> list[str]:
    return sorted(set(split_genres(source.genres)) & set(split_genres(target.genres)))


def _reason(
    item, ratings_by_movie: dict[int, float], titles: dict[int, str], user_mean: float
) -> str:
    """One short sentence naming the strongest of the user's own ratings behind a pick.

    A rating above the user's mean means "you liked a similar film"; a rating below it means the
    pick is the opposite of something they did not like (item CF legitimately works both ways), so
    the wording says "only" to keep the sentence truthful either way.
    """
    if item.contributions:
        source_id, _strength = item.contributions[0]
        rating = ratings_by_movie.get(source_id)
        title = titles.get(source_id)
        if rating is not None and title:
            qualifier = "only " if rating < user_mean else ""
            return f"Because you rated {title} {qualifier}{rating:g}★"
    if item.neighbors_rated:
        count = int(item.neighbors_rated)
        return f"{count} {'viewer' if count == 1 else 'viewers'} with your taste rated it"
    return "Popular with MovieLens users"


def _build_items(
    db: Session,
    batch: RecommendationBatch,
    ratings_by_movie: dict[int, float],
    *,
    genres: list[str],
    year_min: int | None,
    year_max: int | None,
    min_score: float | None,
    limit: int,
) -> tuple[list[RecommendationOut], int]:
    """Resolve movie rows, apply the filters and build response objects with their evidence."""
    wanted = {item.movie_id for item in batch.items}
    wanted |= {movie_id for item in batch.items for movie_id, _ in item.contributions}
    movies = (
        {movie.id: movie for movie in db.scalars(select(Movie).where(Movie.id.in_(wanted)))}
        if wanted
        else {}
    )
    titles = {movie_id: movie.title_display for movie_id, movie in movies.items()}
    user_mean = statistics.fmean(ratings_by_movie.values()) if ratings_by_movie else 0.0

    considered = 0
    results: list[RecommendationOut] = []
    for item in batch.items:
        movie = movies.get(item.movie_id)
        if movie is None:
            # The catalog may be older or newer than the trained artifacts; skip rather than 500.
            continue
        considered += 1
        if not _matches(
            movie,
            genres=genres,
            year_min=year_min,
            year_max=year_max,
            min_score=min_score,
            predicted=item.predicted_rating,
        ):
            continue

        evidence: list[RecommendationEvidence] = []
        for source_id, strength in item.contributions:
            source = movies.get(source_id)
            rating = ratings_by_movie.get(source_id)
            if source is None or rating is None:
                continue
            evidence.append(
                RecommendationEvidence(
                    movie=movie_to_out(source),
                    rating=rating,
                    strength=round(float(strength), 4),
                    shared_genres=_shared_genres(source, movie),
                )
            )

        results.append(
            RecommendationOut(
                movie=movie_to_out(movie),
                score=round(item.score, 4),
                predicted_rating=round(item.predicted_rating, 2),
                reason=_reason(item, ratings_by_movie, titles, user_mean),
                evidence=evidence[:MAX_EVIDENCE],
                neighbors_rated=item.neighbors_rated,
            )
        )
        if len(results) >= limit:
            break
    return results, considered


def _user_ratings(db: Session, user: User) -> dict[int, float]:
    rows = db.execute(
        select(Rating.movie_id, Rating.rating).where(Rating.user_id == user.id)
    ).all()
    return {int(movie_id): float(value) for movie_id, value in rows}


@router.get("/recommend", response_model=RecommendationResponse)
def recommend(
    limit: int = Query(10, ge=1, le=50),
    genres: str = Query("", max_length=200, description="Comma-separated; any match qualifies."),
    year_min: int | None = Query(None, ge=1870, le=2100),
    year_max: int | None = Query(None, ge=1870, le=2100),
    min_score: float | None = Query(None, ge=0.5, le=5.0, description="Minimum predicted rating."),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: RecommendationService = Depends(get_recommender),
) -> RecommendationResponse:
    if year_min is not None and year_max is not None and year_min > year_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="year_min must not exceed year_max",
        )
    selected = _parse_genres(genres)
    filtered = bool(selected) or None not in (year_min, year_max) or min_score is not None
    pool = min(max(limit * 8, limit), FILTERED_POOL_LIMIT) if filtered else limit

    ratings_by_movie = _user_ratings(db, user)
    try:
        batch = service.recommend(list(ratings_by_movie.items()), limit=pool)
    except ArtifactsUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    items, considered = _build_items(
        db,
        batch,
        ratings_by_movie,
        genres=selected,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
        limit=limit,
    )
    return RecommendationResponse(
        algorithm=batch.algorithm,
        cold_start=batch.cold_start,
        filters=RecommendationFiltersApplied(
            genres=selected, year_min=year_min, year_max=year_max, min_score=min_score
        ),
        candidates_considered=considered,
        recommendations=items,
    )


@router.get("/compare", response_model=ComparisonResponse)
def compare_algorithms(
    limit: int = Query(10, ge=1, le=25),
    left: str = Query("item_cf", description="item_cf or user_cf"),
    right: str = Query("user_cf", description="item_cf or user_cf"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    item_service: RecommendationService = Depends(get_recommender),
    user_service: UserCfService = Depends(get_user_cf_recommender),
) -> ComparisonResponse:
    for name in (left, right):
        if name not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"algorithm must be one of {list(SUPPORTED_ALGORITHMS)}",
            )
    if left == right:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="choose two different algorithms",
        )

    ratings_by_movie = _user_ratings(db, user)
    services: dict[str, RecommendationService | UserCfService] = {
        "item_cf": item_service,
        "user_cf": user_service,
    }

    def run(name: str) -> AlgorithmComparison:
        try:
            batch = services[name].recommend(list(ratings_by_movie.items()), limit=limit)
        except ArtifactsUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        items, _ = _build_items(
            db,
            batch,
            ratings_by_movie,
            genres=[],
            year_min=None,
            year_max=None,
            min_score=None,
            limit=limit,
        )
        return AlgorithmComparison(
            algorithm=batch.algorithm,
            cold_start=batch.cold_start,
            recommendations=items,
        )

    left_result = run(left)
    right_result = run(right)
    right_ids = {item.movie.id for item in right_result.recommendations}
    overlap = [item.movie.id for item in left_result.recommendations if item.movie.id in right_ids]
    return ComparisonResponse(
        limit=limit,
        left=left_result,
        right=right_result,
        overlap=overlap,
        overlap_count=len(overlap),
    )


@router.get("/model-runs", response_model=list[ModelRunOut])
def model_runs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[ModelRunOut]:
    """The latest evaluation of each algorithm, newest first, for the accuracy dashboard."""
    rows = db.scalars(select(ModelRun).order_by(ModelRun.trained_at.desc(), ModelRun.id)).all()
    latest: dict[str, ModelRun] = {}
    for row in rows:
        latest.setdefault(row.algorithm, row)
    return [model_run_to_out(row) for row in list(latest.values())[:limit]]
