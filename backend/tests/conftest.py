"""Test configuration: never touch a developer's local database."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

_TEST_DIR = Path(tempfile.mkdtemp(prefix="movie-recsys-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DIR / 'test.db').as_posix()}"

from backend.app import ratelimit  # noqa: E402
from backend.app.db import get_engine, get_session_factory  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.models import Base, Movie  # noqa: E402
from recsys.titles import parse_title  # noqa: E402

Base.metadata.create_all(get_engine())


@pytest.fixture(autouse=True)
def clean_database() -> Iterator[None]:
    with get_engine().begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete())
    # Counters are process state: without this, one test's logins would spend another's budget.
    ratelimit.reset()
    yield


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def signup(
    client: TestClient,
) -> Callable[..., tuple[dict[str, str], dict[str, Any]]]:
    def _signup(
        email: str = "ada@example.com",
        password: str = "correct horse battery",
        display_name: str = "Ada",
    ) -> tuple[dict[str, str], dict[str, Any]]:
        response = client.post(
            "/api/auth/signup",
            json={"email": email, "password": password, "display_name": display_name},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]

    return _signup


@pytest.fixture
def add_movies() -> Callable[..., list[int]]:
    def _add(*specs: dict[str, Any]) -> list[int]:
        rows = []
        for spec in specs:
            parsed = parse_title(spec["title"])
            rows.append(
                Movie(
                    id=spec["id"],
                    title_display=parsed.display_title,
                    title_search=parsed.search_title,
                    release_year=spec.get("year", parsed.release_year),
                    genres=spec.get("genres", "Drama"),
                    rating_count=spec.get("rating_count", 0),
                    in_model=spec.get("in_model", True),
                    tmdb_id=spec.get("tmdb_id"),
                    poster_path=spec.get("poster_path"),
                    overview=spec.get("overview"),
                )
            )
        with get_session_factory()() as session:
            session.add_all(rows)
            session.commit()
        return [row.id for row in rows]

    return _add
