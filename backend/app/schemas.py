"""Pydantic request/response schemas."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models import ModelRun, Movie
from backend.app.tmdb import poster_url
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
        poster_path=movie.poster_path,
        poster_url=poster_url(movie.poster_path),
        overview=movie.overview,
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
