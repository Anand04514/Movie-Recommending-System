"""Phase 6: settings that make a managed deployment work without hand-edited URLs."""

from __future__ import annotations

from backend.app.config import Settings


def test_a_hosted_postgres_url_uses_the_installed_driver() -> None:
    # Render, Railway and Heroku hand out this scheme; SQLAlchemy would otherwise want psycopg2.
    settings = Settings(database_url="postgres://user:pass@host:5432/movie_recs")
    assert settings.database_url == "postgresql+psycopg://user:pass@host:5432/movie_recs"


def test_a_plain_postgresql_url_is_rewritten_too() -> None:
    settings = Settings(database_url="postgresql://user:pass@host/movie_recs")
    assert settings.database_url == "postgresql+psycopg://user:pass@host/movie_recs"


def test_an_explicit_driver_is_left_alone() -> None:
    url = "postgresql+psycopg://user:pass@host/movie_recs"
    assert Settings(database_url=url).database_url == url


def test_sqlite_and_encoded_passwords_survive_untouched() -> None:
    assert Settings(database_url="sqlite:///./movie_recs.db").database_url == (
        "sqlite:///./movie_recs.db"
    )
    assert Settings(database_url="postgres://u:p%40ss@host/db").database_url == (
        "postgresql+psycopg://u:p%40ss@host/db"
    )


def test_frontend_serving_is_off_unless_a_deployment_asks_for_it() -> None:
    settings = Settings()
    assert settings.serve_frontend is False
    assert settings.frontend_dist == "frontend/dist"
