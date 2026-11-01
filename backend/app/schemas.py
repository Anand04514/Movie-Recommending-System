"""Pydantic request/response schemas."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models import ModelRun, Movie
from backend.app.tmdb import TmdbSearchResult, poster_url
from recsys.titles import normalize_for_search, split_genres

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_email(value: str) -> str:
    email = value.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("enter a valid email address")
    return email


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str) -> str:
        return _clean_email(value)

    @field_validator("display_name")
    @classmethod
    def check_display_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display name cannot be blank")
        return cleaned


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class PasswordResetRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return _clean_email(value)


class PasswordResetConfirm(BaseModel):
    token: str = Field(min_length=16, max_length=200)
    password: str = Field(min_length=8, max_length=128)


class AccountDeleteRequest(BaseModel):
    """Deletion is irreversible, so it re-asks for the password instead of trusting the token."""

    password: str = Field(min_length=1, max_length=128)


class MessageOut(BaseModel):
    message: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    display_name: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class MovieOut(BaseModel):
    id: int
    title: str
    release_year: int | None
    genres: list[str]
    rating_count: int
    in_model: bool
    # "movielens" for the dataset, "tmdb" for a film ingested on demand (5.10).
    source: str = "movielens"
    poster_path: str | None = None
    poster_url: str | None = None
    overview: str | None = None


def movie_to_out(movie: Movie) -> MovieOut:
    return MovieOut(
        id=movie.id,
        title=movie.title_display,
        release_year=movie.release_year,
        genres=split_genres(movie.genres),
        rating_count=movie.rating_count,
        in_model=movie.in_model,
        source=movie.source or "movielens",
        poster_path=movie.poster_path,
        poster_url=poster_url(movie.poster_path),
        overview=movie.overview,
    )


class ExternalMovieOut(BaseModel):
    """A TMDB search hit that is not in the catalog yet, so it has no catalog id."""

    tmdb_id: int
    title: str
    release_year: int | None
    genres: list[str]
    overview: str | None = None
    poster_path: str | None = None
    poster_url: str | None = None
    vote_count: int = 0


def external_to_out(result: TmdbSearchResult) -> ExternalMovieOut:
    return ExternalMovieOut(
        tmdb_id=result.tmdb_id,
        title=result.title,
        release_year=result.release_year,
        genres=list(result.genres),
        overview=result.overview,
        poster_path=result.poster_path,
        poster_url=poster_url(result.poster_path),
        vote_count=result.vote_count,
    )


class RatingCreate(BaseModel):
    movie_id: int = Field(gt=0)
    rating: float

    @field_validator("rating")
    @classmethod
    def check_half_star(cls, value: float) -> float:
        if not 0.5 <= value <= 5.0:
            raise ValueError("rating must be between 0.5 and 5.0")
        if abs(value * 2 - round(value * 2)) > 1e-9:
            raise ValueError("rating must be a half-star step")
        return value


class RatingOut(BaseModel):
    movie: MovieOut
    rating: float
    updated_at: datetime


class RecommendationEvidence(BaseModel):
    """One of the user's own ratings that pushed a recommendation up, and why it relates."""

    movie: MovieOut
    rating: float
    strength: float
    shared_genres: list[str]


class RecommendationOut(BaseModel):
    movie: MovieOut
    score: float
    predicted_rating: float
    reason: str
    evidence: list[RecommendationEvidence] = []
    neighbors_rated: int | None = None
    # A lower-confidence, out-of-profile pick reserved at the back of a full list.
    discovery: bool = False
    # Released inside the recency window, so it earned the re-ranking bonus (5.12).
    new_release: bool = False


class RecommendationFiltersApplied(BaseModel):
    genres: list[str] = []
    year_min: int | None = None
    year_max: int | None = None
    min_score: float | None = None


ALGORITHM_NAMES = (
    "item_cf",
    "user_cf",
    "svd",
    "content",
    "hybrid",
    "popularity",
)
AlgorithmName = Literal[
    "item_cf", "user_cf", "svd", "content", "hybrid", "popularity"
]


class RecommendationResponse(BaseModel):
    algorithm: AlgorithmName
    cold_start: bool
    filters: RecommendationFiltersApplied
    candidates_considered: int
    needs_onboarding: bool = False
    # True when the scored candidates came from precomputed_recs instead of a live model call.
    cached: bool = False
    # Set only when an experiment assigned this list; an explicitly chosen model leaves both null.
    experiment: str | None = None
    variant: str | None = None
    recommendations: list[RecommendationOut]


class AlgorithmComparison(BaseModel):
    algorithm: AlgorithmName
    cold_start: bool
    recommendations: list[RecommendationOut]


class ComparisonResponse(BaseModel):
    limit: int
    left: AlgorithmComparison
    right: AlgorithmComparison
    overlap: list[int]
    overlap_count: int


class VariantReportOut(BaseModel):
    variant: str
    users: int
    impressions: int
    likes: int
    rejects: int
    clicks: int
    rated_after: int
    positive_rate: float | None = None
    rated_rate: float | None = None
    engagement_rate: float | None = None


class ArmComparisonOut(BaseModel):
    """The measured gap between two arms, with its interval.

    ``significant`` is false whenever the interval crosses zero, so a reader never has to do the
    arithmetic that turns a small sample into a confident-sounding sentence.
    """

    metric: str
    a_variant: str
    b_variant: str
    difference: float
    low: float
    high: float
    confidence: float
    significant: bool


class ExperimentReportOut(BaseModel):
    name: str
    description: str
    variants: list[VariantReportOut]
    comparison: ArmComparisonOut | None = None


class ReleaseEngagementOut(BaseModel):
    """Impressions from the recency row and what users did with them (PLAN 5.12)."""

    impressions: int
    total_impressions: int
    share_of_impressions: float | None = None
    users: int
    films: int
    likes: int
    rejects: int
    clicks: int
    rated_after: int
    like_rate: float | None = None
    rated_rate: float | None = None


class ModelRunOut(BaseModel):
    id: int
    algorithm: str
    params: dict[str, Any]
    rmse: float | None
    mae: float | None
    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None
    trained_at: datetime


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    count: int
    results: list[MovieOut]
    # Films TMDB knows that the catalog does not, shown separately and never persisted (5.10).
    external: list[ExternalMovieOut] = []
    external_source: str | None = None


class OnboardingResponse(BaseModel):
    """A small, diverse set of well-known films for a new user to rate."""

    target: int = 5
    movies: list[MovieOut]


class FeedbackCreate(BaseModel):
    movie_id: int = Field(gt=0)
    action: Literal["liked", "rejected", "clicked"]
    source_algorithm: str | None = Field(default=None, max_length=32)


class FeedbackOut(BaseModel):
    movie_id: int
    action: str
    source_algorithm: str | None
    created_at: datetime


def normalized_query_text(query: str) -> str:
    """The same normalization stored in ``movies.title_search``, applied to user input."""
    return normalize_for_search(query)


def experiment_report_to_out(report: Any) -> ExperimentReportOut:
    """Convert an ``experiments.ExperimentReport`` into its response shape."""
    comparison = report.comparison
    return ExperimentReportOut(
        name=report.name,
        description=report.description,
        comparison=(
            ArmComparisonOut(
                metric=comparison.metric,
                a_variant=comparison.a_variant,
                b_variant=comparison.b_variant,
                difference=comparison.difference,
                low=comparison.low,
                high=comparison.high,
                confidence=comparison.confidence,
                significant=comparison.significant,
            )
            if comparison is not None
            else None
        ),
        variants=[
            VariantReportOut(
                variant=variant.variant,
                users=variant.users,
                impressions=variant.impressions,
                likes=variant.likes,
                rejects=variant.rejects,
                clicks=variant.clicks,
                rated_after=variant.rated_after,
                positive_rate=variant.positive_rate,
                rated_rate=variant.rated_rate,
                engagement_rate=variant.engagement_rate,
            )
            for variant in report.variants
        ],
    )


def model_run_to_out(run: ModelRun) -> ModelRunOut:
    """The stored row with its JSON settings parsed back into a mapping."""
    try:
        params: dict[str, Any] = json.loads(run.params)
        if not isinstance(params, dict):
            params = {}
    except (TypeError, json.JSONDecodeError):
        params = {}
    return ModelRunOut(
        id=run.id,
        algorithm=run.algorithm,
        params=params,
        rmse=run.rmse,
        mae=run.mae,
        precision_at_k=run.precision_at_k,
        recall_at_k=run.recall_at_k,
        ndcg_at_k=run.ndcg_at_k,
        trained_at=run.trained_at,
    )
