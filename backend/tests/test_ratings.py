"""Phase 2.6: ratings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

MOVIES: list[dict[str, Any]] = [
    {"id": 1, "title": "Matrix, The (1999)", "rating_count": 100},
    {"id": 2, "title": "Amélie (2001)", "rating_count": 50},
]


@pytest.fixture(autouse=True)
def movies(add_movies: Callable[..., list[int]]) -> None:
    add_movies(*MOVIES)


def test_rating_requires_sign_in(client: TestClient) -> None:
    assert client.get("/api/ratings").status_code == 401
    assert (
        client.post("/api/ratings", json={"movie_id": 1, "rating": 4.5}).status_code == 401
    )


def test_create_update_list_and_delete(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    headers, _ = signup()

    created = client.post("/api/ratings", json={"movie_id": 1, "rating": 4.5}, headers=headers)
    assert created.status_code == 200
    assert created.json()["rating"] == 4.5
    assert created.json()["movie"]["title"] == "The Matrix"

    listed = client.get("/api/ratings", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    updated = client.post("/api/ratings", json={"movie_id": 1, "rating": 3.0}, headers=headers)
    assert updated.status_code == 200
    assert updated.json()["rating"] == 3.0
    assert len(client.get("/api/ratings", headers=headers).json()) == 1

    deleted = client.delete("/api/ratings/1", headers=headers)
    assert deleted.status_code == 204
    assert client.get("/api/ratings", headers=headers).json() == []
    assert client.delete("/api/ratings/1", headers=headers).status_code == 404


@pytest.mark.parametrize("rating", [4.25, 5.5, 0.0, -1.0, 3.3])
def test_rating_must_be_a_half_star_in_range(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    rating: float,
) -> None:
    headers, _ = signup()
    response = client.post("/api/ratings", json={"movie_id": 1, "rating": rating}, headers=headers)
    assert response.status_code == 422


@pytest.mark.parametrize("rating", [0.5, 1.0, 2.5, 4.5, 5.0])
def test_half_star_boundaries_are_accepted(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    rating: float,
) -> None:
    headers, _ = signup()
    response = client.post("/api/ratings", json={"movie_id": 1, "rating": rating}, headers=headers)
    assert response.status_code == 200
    assert response.json()["rating"] == rating


def test_rating_unknown_movie_is_404(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    headers, _ = signup()
    response = client.post("/api/ratings", json={"movie_id": 999, "rating": 4.0}, headers=headers)
    assert response.status_code == 404


def test_ratings_are_isolated_between_users(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    ada, _ = signup(email="ada@example.com")
    grace, _ = signup(email="grace@example.com")

    client.post("/api/ratings", json={"movie_id": 1, "rating": 5.0}, headers=ada)
    assert len(client.get("/api/ratings", headers=ada).json()) == 1
    assert client.get("/api/ratings", headers=grace).json() == []

    client.post("/api/ratings", json={"movie_id": 2, "rating": 2.5}, headers=grace)
    grace_ratings = client.get("/api/ratings", headers=grace).json()
    assert len(grace_ratings) == 1
    assert grace_ratings[0]["movie"]["id"] == 2
