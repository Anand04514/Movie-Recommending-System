"""Environment-driven settings for the API."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, overridable with environment variables or a local ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Movie Recs API"
    version: str = "0.1.0"
    # SQLite keeps early local development frictionless; PostgreSQL is the real target.
    database_url: str = "sqlite:///./movie_recs.db"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
