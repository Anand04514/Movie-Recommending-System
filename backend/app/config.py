"""Environment-driven settings for the API."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, overridable with environment variables or a local ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Movie Recs API"
    version: str = "0.2.0"
    # SQLite keeps early local development frictionless; PostgreSQL is the real target.
    database_url: str = "sqlite:///./movie_recs.db"
    # Where prepare_data.py and train_and_precompute.py write their artifacts.
    processed_dir: str = "data/processed"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Auth. The default is for local development only; set JWT_SECRET in any real deployment.
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7


@lru_cache
def get_settings() -> Settings:
    return Settings()
