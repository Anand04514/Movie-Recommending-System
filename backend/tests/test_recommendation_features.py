"""Phase 4.1-4.4: filters, explanations, algorithm comparison and the accuracy dashboard."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from backend.app.db import get_session_factory
from backend.app.main import app
from backend.app.models import ModelRun
from backend.app.recommendations import (
    RecommendationService,
    ScoredItem,
    UserCfService,
    get_recommender,
    get_user_cf_recommender,
)
from backend.app.routers.recommendations import _reason
from recsys.item_cf import build_item_neighbors, save_neighbors

# Two taste groups: u0-u2 like items 0, 1, 4 and dislike 2, 3, 6, 7; u3-u5 the reverse.
MATRIX = np.array(
    [
        [5.0, 4.0, 1.0, 1.0, 4.0, 3.0, 1.0, 2.0],
        [4.0, 5.0, 2.0, 1.0, 4.5, 3.5, 1.0, 2.0],
        [5.0, 4.0, 1.0, 2.0, 3.5, 3.0, 1.0, 2.5],
        [1.0, 2.0, 5.0, 4.0, 2.0, 2.5, 5.0, 4.0],
        [1.0, 1.0, 4.0, 5.0, 1.5, 2.0, 4.5, 5.0],
        [2.0, 1.0, 5.0, 4.0, 2.5, 2.0, 5.0, 4.0],
    ],
    dtype="float32",
)

MOVIE_IDS = [10, 11, 12, 13, 14, 15, 16, 17]
MOVIES: list[dict[str, Any]] = [
    {"id": 10, "title": "Alpha (2000)", "genres": "Action|Thriller", "rating_count": 6},
    {"id": 11, "title": "Beta (2001)", "genres": "Action|Sci-Fi", "rating_count": 6},
    {"id": 12, "title": "Gamma (2002)", "genres": "Drama", "rating_count": 6},
    {"id": 13, "title": "Delta (2003)", "genres": "Drama|Romance", "rating_count": 6},
    {"id": 14, "title": "Epsilon (1990)", "genres": "Action|Adventure", "rating_count": 6},
    {"id": 15, "title": "Zeta (1995)", "genres": "Comedy", "rating_count": 6},
    {"id": 16, "title": "Eta (2010)", "genres": "Horror", "rating_count": 6},
    {"id": 17, "title": "Theta (2012)", "genres": "Sci-Fi", "rating_count": 6},
]

# One training user's own taste profile, so the API user really has something to fold in.
RATED = ((10, 5.0), (11, 4.0), (12, 1.0), (13, 1.0))

# These tests exercise the item-CF path, so they name it instead of relying on the
# experiment default (which is a different model on purpose).
ITEM = {"algorithm": "item_cf"}


def write_artifacts(directory: Path) -> None:
    matrix = sparse.csr_matrix(MATRIX)
    neighbors = build_item_neighbors(matrix, k=4, block_size=4)
    save_neighbors(neighbors, directory / "item_neighbors.npz")
    sparse.save_npz(directory / "train_matrix.npz", matrix)
    pd.DataFrame({"movieId": MOVIE_IDS, "movie_index": list(range(len(MOVIE_IDS)))}).to_parquet(
        directory / "movie_id_map.parquet", index=False
    )


@pytest.fixture
def recommender(tmp_path: Path) -> Iterator[None]:
    write_artifacts(tmp_path)
    app.dependency_overrides[get_recommender] = lambda: RecommendationService(tmp_path)
    app.dependency_overrides[get_user_cf_recommender] = lambda: UserCfService(tmp_path)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def customer(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], add_movies
) -> tuple[dict[str, str], dict]:
    add_movies(*MOVIES)
    headers, profile = signup()
    for movie_id, rating in RATED:
        response = client.post(
            "/api/ratings", json={"movie_id": movie_id, "rating": rating}, headers=headers
        )
        assert response.status_code == 200, response.text
    return headers, profile


def test_unfiltered_recommendations_come_from_the_users_taste_group(
    client: TestClient, customer, recommender: None
) -> None:
    headers, _ = customer
    body = client.get("/api/recommend", params={**ITEM, "limit": 5}, headers=headers).json()

    assert body["algorithm"] == "item_cf"
    assert body["cold_start"] is False
    titles = [item["movie"]["title"] for item in body["recommendations"]]
    assert titles[0] == "Epsilon"  # the film their group loves and they have not rated
    assert "Alpha" not in titles and "Beta" not in titles  # never recommend what they rated
    assert "Eta" not in titles and "Theta" not in titles  # their group dislikes these
    assert body["filters"] == {
        "genres": [],
        "year_min": None,
        "year_max": None,
        "min_score": None,
    }
    assert body["candidates_considered"] >= len(titles)


def test_every_pick_explains_itself_from_the_users_own_ratings(
    client: TestClient, customer, recommender: None
) -> None:
    headers, _ = customer
    body = client.get("/api/recommend", params={**ITEM, "limit": 5}, headers=headers).json()
    pick = body["recommendations"][0]

    evidence = pick["evidence"]
    assert evidence, "an item-based pick must carry the ratings behind it"
    assert all(entry["strength"] > 0 for entry in evidence)
    assert all(entry["movie"]["id"] in dict(RATED) for entry in evidence)

    # The sentence names the strongest cited rating, and only that one.
    assert pick["reason"].startswith(f"Because you rated {evidence[0]['movie']['title']}")
    assert evidence[0]["rating"] == dict(RATED)[evidence[0]["movie"]["id"]]

    # Alpha (Action|Thriller) and Epsilon (Action|Adventure) share exactly one genre.
    strongest = evidence[0]["movie"]
    if strongest["id"] == 10:
        assert evidence[0]["shared_genres"] == ["Action"]
    strengths = [entry["strength"] for entry in evidence]
    assert strengths == sorted(strengths, reverse=True)
    assert len(evidence) <= 3


def test_reason_wording_distinguishes_a_liked_source_from_a_disliked_one() -> None:
    """A cited rating above the user's mean reads as "you liked it"; below it, as the opposite."""
    ratings = {10: 5.0, 12: 1.0}
    titles = {10: "Alpha", 12: "Gamma"}
    mean = 2.75

    liked = ScoredItem(
        movie_id=14,
        score=3.0,
        predicted_rating=4.1,
        source_movie_id=10,
        source_rating=5.0,
        contributions=((10, 1.4),),
    )
    assert _reason(liked, ratings, titles, mean) == "Because you rated Alpha 5★"

    aversion = ScoredItem(
        movie_id=14,
        score=3.0,
        predicted_rating=4.1,
        source_movie_id=12,
        source_rating=1.0,
        contributions=((12, 2.2),),
    )
    assert _reason(aversion, ratings, titles, mean) == "Because you rated Gamma only 1★"

    neighbour_based = ScoredItem(
        movie_id=14,
        score=3.0,
        predicted_rating=4.1,
        source_movie_id=None,
        source_rating=None,
        neighbors_rated=3,
    )
    assert _reason(neighbour_based, ratings, titles, mean) == "3 viewers with your taste rated it"
    assert (
        _reason(replace(neighbour_based, neighbors_rated=1), ratings, titles, mean)
        == "1 viewer with your taste rated it"
    )

    cold_start = ScoredItem(
        movie_id=14, score=3.0, predicted_rating=3.2, source_movie_id=None, source_rating=None
    )
    assert _reason(cold_start, ratings, titles, mean) == "Popular with MovieLens users"


def test_genre_filter_widens_the_pool_instead_of_truncating_it(
    client: TestClient, customer, recommender: None
) -> None:
    headers, _ = customer

    top_one = client.get("/api/recommend", params={**ITEM, "limit": 1}, headers=headers).json()
    assert [item["movie"]["title"] for item in top_one["recommendations"]] == ["Epsilon"]

    # Epsilon outranks Zeta for this user, so filtering *after* truncating to the page size would
    # return nothing here: the service has to score a wider pool than the page it returns.
    comedy = client.get(
        "/api/recommend", params={**ITEM, "limit": 1, "genres": "Comedy"}, headers=headers
    ).json()
    assert [item["movie"]["title"] for item in comedy["recommendations"]] == ["Zeta"]
    assert comedy["filters"]["genres"] == ["Comedy"]

    horror = client.get(
        "/api/recommend", params={**ITEM, "limit": 5, "genres": "horror"}, headers=headers
    ).json()
    assert horror["recommendations"] == []


def test_year_and_score_filters_apply(client: TestClient, customer, recommender: None) -> None:
    headers, _ = customer

    recent = client.get(
        "/api/recommend", params={**ITEM, "limit": 5, "year_min": 2005}, headers=headers
    ).json()
    assert recent["recommendations"] == []  # nothing above the group's taste is that new

    classic = client.get(
        "/api/recommend",
        params={**ITEM, "limit": 5, "year_min": 1985, "year_max": 1992},
        headers=headers,
    ).json()
    assert [item["movie"]["title"] for item in classic["recommendations"]] == ["Epsilon"]

    # The threshold is compared against the served predicted rating, so calibrate on it rather
    # than guessing a number: a hair below keeps the pick, a hair above drops it.
    first = client.get("/api/recommend", params={**ITEM, "limit": 5}, headers=headers).json()[
        "recommendations"
    ][0]
    predicted = first["predicted_rating"]

    def ids(min_score: float) -> set[int]:
        body = client.get(
            "/api/recommend", params={**ITEM, "limit": 5, "min_score": min_score}, headers=headers
        ).json()
        return {item["movie"]["id"] for item in body["recommendations"]}

    assert first["movie"]["id"] in ids(round(predicted - 0.05, 2))
    assert first["movie"]["id"] not in ids(round(predicted + 0.05, 2))

    combined = client.get(
        "/api/recommend",
        params={**ITEM, "limit": 5, "genres": "Comedy", "year_min": 1990, "year_max": 2000},
        headers=headers,
    ).json()
    assert [item["movie"]["title"] for item in combined["recommendations"]] == ["Zeta"]


def test_filter_validation_and_bad_ranges(client: TestClient, customer, recommender: None) -> None:
    headers, _ = customer

    assert (
        client.get(
            "/api/recommend", params={**ITEM, "year_min": 2010, "year_max": 1990}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.get("/api/recommend", params={**ITEM, "limit": 0}, headers=headers).status_code
        == 422
    )
    assert (
        client.get(
            "/api/recommend", params={**ITEM, "min_score": 9.0}, headers=headers
        ).status_code
        == 422
    )


def test_user_based_cf_serves_a_live_user_and_labels_its_reason(
    client: TestClient, customer, recommender: None
) -> None:
    headers, _ = customer
    body = client.get("/api/compare", params={"limit": 5}, headers=headers).json()

    assert body["left"]["algorithm"] == "item_cf"
    assert body["right"]["algorithm"] == "user_cf"
    assert body["right"]["cold_start"] is False
    neighbour_pick = body["right"]["recommendations"][0]
    assert neighbour_pick["movie"]["title"] == "Epsilon"
    assert neighbour_pick["neighbors_rated"] >= 1
    assert "with your taste rated it" in neighbour_pick["reason"]
    assert neighbour_pick["evidence"] == []  # user CF has no "because you rated X" to offer

    assert body["overlap"] and body["overlap_count"] == len(body["overlap"])
    left_ids = {item["movie"]["id"] for item in body["left"]["recommendations"]}
    assert set(body["overlap"]) <= left_ids


def test_comparison_validates_the_algorithms(
    client: TestClient, customer, recommender: None
) -> None:
    headers, _ = customer

    assert client.get("/api/compare", params={"left": "als"}, headers=headers).status_code == 422
    same = client.get(
        "/api/compare", params={"left": "item_cf", "right": "item_cf"}, headers=headers
    )
    assert same.status_code == 422
    assert "two different" in same.json()["detail"]


def test_model_runs_endpoint_returns_the_latest_run_per_algorithm(
    client: TestClient, recommender: None
) -> None:
    now = datetime.now(UTC)
    with get_session_factory()() as session:
        session.add_all(
            [
                ModelRun(
                    algorithm="item_cf",
                    params='{"k": 10, "coverage": 0.19}',
                    rmse=0.95,
                    mae=0.72,
                    precision_at_k=0.08,
                    recall_at_k=0.07,
                    ndcg_at_k=0.10,
                    trained_at=now - timedelta(days=3),
                ),
                ModelRun(
                    algorithm="item_cf",
                    params='{"k": 10, "coverage": 0.21}',
                    rmse=0.93,
                    mae=0.71,
                    precision_at_k=0.09,
                    recall_at_k=0.08,
                    ndcg_at_k=0.11,
                    trained_at=now,
                ),
                ModelRun(
                    algorithm="popularity",
                    params="{}",
                    rmse=0.97,
                    mae=0.75,
                    precision_at_k=0.06,
                    recall_at_k=0.05,
                    ndcg_at_k=0.07,
                    trained_at=now - timedelta(days=1),
                ),
            ]
        )
        session.commit()

    rows = client.get("/api/model-runs").json()
    assert [row["algorithm"] for row in rows] == ["item_cf", "popularity"]  # newest first
    latest = rows[0]
    assert latest["rmse"] == pytest.approx(0.93)  # the newer item_cf run, not the older one
    assert latest["params"]["coverage"] == pytest.approx(0.21)  # JSON parsed back into a mapping
    assert isinstance(latest["params"]["k"], int)
    assert latest["trained_at"].startswith(now.date().isoformat())


def test_model_runs_endpoint_tolerates_broken_params(
    client: TestClient, recommender: None
) -> None:
    with get_session_factory()() as session:
        session.add(
            ModelRun(
                algorithm="broken",
                params="not json at all",
                rmse=None,
                mae=None,
                precision_at_k=None,
                recall_at_k=None,
                ndcg_at_k=None,
                trained_at=datetime.now(UTC),
            )
        )
        session.commit()

    rows = client.get("/api/model-runs").json()
    assert [row["algorithm"] for row in rows] == ["broken"]
    assert rows[0]["params"] == {}
    assert rows[0]["rmse"] is None


def test_genre_list_comes_from_the_catalog(client: TestClient, add_movies) -> None:
    add_movies(*MOVIES)
    genres = client.get("/api/movies/genres").json()

    assert "Action" in genres and "Comedy" in genres
    assert genres == sorted(genres)
    assert "(no genres listed)" not in genres


def test_recommend_and_compare_still_require_auth(client: TestClient, recommender: None) -> None:
    assert client.get("/api/recommend").status_code == 401
    assert client.get("/api/compare").status_code == 401
