"""Phase 2.5: catalog search and movie detail."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

MOVIES: list[dict[str, Any]] = [
    {
        "id": 1,
        "title": "Matrix, The (1999)",
        "rating_count": 9_000,
        "genres": "Action|Sci-Fi",
    },
    {
        "id": 2,
        "title": "Matrix Reloaded, The (2003)",
        "rating_count": 5_000,
        "genres": "Action|Sci-Fi",
    },
    {"id": 3, "title": "Dune (2021)", "rating_count": 3_000, "genres": "Sci-Fi"},
    {"id": 4, "title": "Dune: Part Two (2024)", "rating_count": 1_500, "genres": "Sci-Fi"},
    {"id": 5, "title": "Amélie (2001)", "rating_count": 800, "genres": "Comedy|Romance"},
]


@pytest.fixture(autouse=True)
def movies(add_movies: Callable[..., list[int]]) -> None:
    add_movies(*MOVIES)


def test_search_normalizes_punctuation_and_articles(client: TestClient) -> None:
    response = client.get("/api/movies/search", params={"q": "Matrix, The"})
    assert response.status_code == 200
    body = response.json()
    assert body["normalized_query"] == "the matrix"
    assert body["count"] == 2
    assert {movie["id"] for movie in body["results"]} == {1, 2}
    # "the matrix..." is a prefix match for both, so popularity decides.
    assert body["results"][0]["id"] == 1
    assert body["results"][0]["title"] == "The Matrix"


def test_search_ranks_word_start_matches_by_popularity(
    client: TestClient, add_movies: Callable[..., list[int]]
) -> None:
    """The real catalog holds both "GodFather (1991)" and "Godfather, The (1972)".

    The normalized form of the 1972 film starts with "the", so a strict prefix tier used to push
    it below near-empty titles that merely begin with the query.
    """
    add_movies(
        {"id": 21, "title": "GodFather (1991)", "rating_count": 40},
        {"id": 22, "title": "Godfather's Story (2013)", "rating_count": 25},
        {"id": 23, "title": "Godfather, The (1972)", "rating_count": 100_000},
    )
    body = client.get("/api/movies/search", params={"q": "godfather", "limit": 10}).json()
    assert [movie["id"] for movie in body["results"]] == [23, 21, 22]
    assert body["results"][0]["title"] == "The Godfather"


def test_search_is_case_insensitive(client: TestClient) -> None:
    lower = client.get("/api/movies/search", params={"q": "matrix"}).json()
    upper = client.get("/api/movies/search", params={"q": "MATRIX"}).json()
    assert upper["normalized_query"] == "matrix"
    assert [m["id"] for m in lower["results"]] == [m["id"] for m in upper["results"]]


def test_search_finds_a_recent_release(client: TestClient) -> None:
    body = client.get("/api/movies/search", params={"q": "dune"}).json()
    ids = {movie["id"] for movie in body["results"]}
    assert ids == {3, 4}
    years = {movie["release_year"] for movie in body["results"]}
    assert years == {2021, 2024}


def test_search_shows_genres_as_a_list(client: TestClient) -> None:
    body = client.get("/api/movies/search", params={"q": "matrix, the"}).json()
    matrix = next(movie for movie in body["results"] if movie["id"] == 1)
    assert matrix["genres"] == ["Action", "Sci-Fi"]
    assert matrix["in_model"] is True


def test_search_without_matches_is_empty(client: TestClient) -> None:
    body = client.get("/api/movies/search", params={"q": "zzzzzz"}).json()
    assert body["count"] == 0
    assert body["results"] == []


def test_empty_query_returns_most_rated_movies(client: TestClient) -> None:
    body = client.get("/api/movies/search").json()
    assert body["normalized_query"] == ""
    assert [movie["id"] for movie in body["results"]] == [1, 2, 3, 4, 5]


def test_search_limit_is_validated(client: TestClient) -> None:
    assert client.get("/api/movies/search", params={"limit": 0}).status_code == 422
    assert client.get("/api/movies/search", params={"limit": 51}).status_code == 422


def test_movie_detail_and_missing_movie(client: TestClient) -> None:
    found = client.get("/api/movies/5")
    assert found.status_code == 200
    assert found.json()["title"] == "Amélie"
    assert found.json()["release_year"] == 2001

    missing = client.get("/api/movies/999999")
    assert missing.status_code == 404
