"""Phase 2.5: catalog search and movie detail."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.db import init_database, trgm_available
from backend.app.models import Movie
from backend.app.routers.movies import (
    _contains_filter,
    _fuzzy_filter,
    _order_by_relevance,
    fuzzy_escalation_needed,
)

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


def test_search_ordering_only_calls_similarity_when_trgm_is_installed() -> None:
    """A PostgreSQL server without pg_trgm must not receive a similarity() call.

    The extension is an optimisation, not a dependency: a build without contrib extensions (or a
    role that cannot install them) still has to answer search instead of failing every request.
    """
    with_similarity = str(_order_by_relevance(select(Movie), "matrix", "postgresql", fuzzy=True))
    without = str(_order_by_relevance(select(Movie), "matrix", "postgresql", fuzzy=False))

    assert "similarity(movies.title_search" in with_similarity
    assert "similarity(" not in without
    # Both still apply the word-start tier ahead of popularity.
    for sql in (with_similarity, without):
        assert "movies.title_search LIKE" in sql
        assert "movies.rating_count DESC" in sql


def test_popularity_outranks_similarity_inside_a_tier() -> None:
    """The live catalog showed why: an exact trigram match from a 9-rating film won a query.

    "GodFather (1991)" scores similarity 1.0 for "godfather"; "The Godfather" scores 0.77 with
    66k ratings. Closeness may only settle ties between films of comparable standing.
    """
    sql = str(_order_by_relevance(select(Movie), "godfather", "postgresql", fuzzy=True))

    assert sql.index("movies.rating_count DESC") < sql.index("similarity(")


def test_fuzzy_match_is_a_second_query_not_an_or() -> None:
    """A misspelling has to match in the WHERE clause, not just in the ORDER BY.

    With only an ORDER BY, ``similarity`` re-ranks rows that already contain the query, so
    "matrx" returned nothing at all on a live PostgreSQL server. It also has to stay a separate
    statement: ``LIKE OR <%`` cannot use the trigram index and took ~180 ms per search.
    """
    fuzzy = str(_fuzzy_filter(select(Movie), "matrx"))
    contains = str(_contains_filter(select(Movie), "matrx"))

    assert ":param_1 <% movies.title_search" in fuzzy
    assert "LIKE" not in fuzzy
    assert "<%" not in contains
    assert "movies.title_search LIKE" in contains


def test_fuzzy_is_only_attempted_when_nothing_matched_exactly() -> None:
    """A real answer must not be padded with loosely similar titles."""
    assert fuzzy_escalation_needed(fuzzy=True, normalized="matrx", found=0) is True
    assert fuzzy_escalation_needed(fuzzy=True, normalized="inception", found=1) is False
    assert fuzzy_escalation_needed(fuzzy=True, normalized="matrix", found=20) is False
    # Two letters of trigrams match nearly every title, so they are not worth a second query.
    assert fuzzy_escalation_needed(fuzzy=True, normalized="ma", found=0) is False
    assert fuzzy_escalation_needed(fuzzy=False, normalized="matrx", found=0) is False
    assert fuzzy_escalation_needed(fuzzy=True, normalized="", found=0) is False


def test_empty_query_matches_everything() -> None:
    assert "WHERE" not in str(_contains_filter(select(Movie), "")).upper()
    assert "WHERE" not in str(_fuzzy_filter(select(Movie), "")).upper()


def test_init_database_skips_extensions_on_sqlite() -> None:
    """Startup is dialect-aware, and leaves the probe false so search uses LIKE."""
    init_database()
    assert trgm_available() is False


def test_movie_detail_and_missing_movie(client: TestClient) -> None:
    found = client.get("/api/movies/5")
    assert found.status_code == 200
    assert found.json()["title"] == "Amélie"
    assert found.json()["release_year"] == 2001

    missing = client.get("/api/movies/999999")
    assert missing.status_code == 404
