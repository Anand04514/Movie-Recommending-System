"""Pydantic request/response schemas."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models import Movie
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


class RecommendationOut(BaseModel):
    movie: MovieOut
    score: float
    predicted_rating: float
    reason: str


class RecommendationResponse(BaseModel):
    algorithm: Literal["item_cf", "popularity"]
    cold_start: bool
    recommendations: list[RecommendationOut]


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    count: int
    results: list[MovieOut]


def normalized_query_text(query: str) -> str:
    """The same normalization stored in ``movies.title_search``, applied to user input."""
    return normalize_for_search(query)
