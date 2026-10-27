"""Phase 2.3/2.7: recommendation serving and explanations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from backend.app.main import app
from backend.app.recommendations import (
    ArtifactsUnavailable,
    RecommendationService,
    get_recommender,
)
from recsys.item_cf import build_item_neighbors, save_neighbors

MOVIE_IDS = [10, 11, 12, 13]
MOVIES: list[dict[str, Any]] = [
    {"id": 10, "title": "Alpha (2000)", "rating_count": 2},
    {"id": 11, "title": "Beta (2001)", "rating_count": 2},
    {"id": 12, "title": "Gamma (2002)", "rating_count": 2},
    {"id": 13, "title": "Delta (2003)", "rating_count": 2},
]


def build_service(directory: Path) -> RecommendationService:
    """Two positively correlated pairs: (10, 11) and (12, 13).

    Users rate all four movies, so mean-centering leaves real co-variation rather than the
    artificial anti-correlation a two-movie matrix would create.
    """
    matrix = sparse.csr_matrix(
        [
            [5.0, 4.0, 1.0, 1.0],
            [4.0, 5.0, 2.0, 1.0],
            [5.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 4.0],
            [1.0, 1.0, 4.0, 5.0],
            [2.0, 1.0, 5.0, 4.0],
        ],
        dtype="float32",
    )
    neighbors = build_item_neighbors(matrix, k=3, block_size=2)
    save_neighbors(neighbors, directory / "item_neighbors.npz")
    pd.DataFrame(
        {"movieId": MOVIE_IDS, "movie_index": [0, 1, 2, 3]}
    ).to_parquet(directory / "movie_id_map.parquet", index=False)
    return RecommendationService(directory)


@pytest.fixture
def recommender(tmp_path: Path) -> RecommendationService:
    return build_service(tmp_path)


@pytest.fixture
def use_recommender(recommender: RecommendationService) -> Iterator[None]:
    app.dependency_overrides[get_recommender] = lambda: recommender
    yield
    app.dependency_overrides.clear()


def test_service_item_cf_recommendations(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    batch = service.recommend([(10, 5.0), (12, 1.0)], limit=5)

    assert batch.algorithm == "item_cf"
    assert batch.cold_start is False
    assert batch.items
    movie_ids = [item.movie_id for item in batch.items]
    assert movie_ids == [11]
    assert 10 not in movie_ids and 12 not in movie_ids
    item = batch.items[0]
    assert item.source_movie_id == 10
    assert item.source_rating == 5.0
    assert 0.5 <= item.predicted_rating <= 5.0
    assert item.score > 0


def test_single_rating_falls_back_to_popularity(tmp_path: Path) -> None:
    batch = build_service(tmp_path).recommend([(10, 5.0)], limit=3)
    assert batch.algorithm == "popularity"
    # Popularity is never personalized, whatever the user has already rated.
    assert batch.cold_start is True
    assert all(item.movie_id != 10 for item in batch.items)
    assert len(batch.items) == 3


def test_equal_ratings_fall_back_to_popularity_without_serving_them(tmp_path: Path) -> None:
    """Ratings that all equal the user's mean leave no signal after mean-centering."""
    batch = build_service(tmp_path).recommend([(10, 5.0), (11, 5.0)], limit=4)
    assert batch.algorithm == "popularity"
    assert batch.cold_start is True
    movie_ids = [item.movie_id for item in batch.items]
    assert 10 not in movie_ids and 11 not in movie_ids
    assert movie_ids == [12, 13]


def test_cold_start_is_popularity(tmp_path: Path) -> None:
    batch = build_service(tmp_path).recommend([], limit=2)
    assert batch.algorithm == "popularity"
    assert batch.cold_start is True
    assert [item.movie_id for item in batch.items] == [10, 11]
    assert all(item.popularity_count == 6 for item in batch.items)


def test_unknown_movie_ids_are_ignored(tmp_path: Path) -> None:
    batch = build_service(tmp_path).recommend([(999_999, 5.0)], limit=4)
    assert batch.algorithm == "popularity"
    assert batch.cold_start is True


def test_every_rated_movie_is_excluded(tmp_path: Path) -> None:
    batch = build_service(tmp_path).recommend([(movie_id, 4.0) for movie_id in MOVIE_IDS])
    assert batch.items == []


def test_missing_artifacts_raise_a_helpful_error(tmp_path: Path) -> None:
    service = RecommendationService(tmp_path / "not-trained")
    with pytest.raises(ArtifactsUnavailable, match="train_and_precompute"):
        service.recommend([])


def test_recommend_endpoint_serves_item_cf_with_reasons(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    add_movies: Callable[..., list[int]],
    use_recommender: None,
) -> None:
    add_movies(*MOVIES)
    headers, _ = signup()

    for movie_id, rating in ((10, 5.0), (12, 1.0)):
        saved = client.post(
            "/api/ratings", json={"movie_id": movie_id, "rating": rating}, headers=headers
        )
        assert saved.status_code == 200

    response = client.get("/api/recommend", params={"limit": 5}, headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["algorithm"] == "item_cf"
    assert body["cold_start"] is False
    assert body["recommendations"]
    for recommendation in body["recommendations"]:
        assert recommendation["movie"]["id"] not in {10, 12}
        assert recommendation["reason"].startswith("Because you rated Alpha 5")
        assert 0.5 <= recommendation["predicted_rating"] <= 5.0


def test_recommend_endpoint_cold_start(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    add_movies: Callable[..., list[int]],
    use_recommender: None,
) -> None:
    add_movies(*MOVIES)
    headers, _ = signup(email="new@example.com")

    response = client.get("/api/recommend", params={"limit": 4}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "popularity"
    assert body["cold_start"] is True
    assert len(body["recommendations"]) == 4
    assert "Popular with MovieLens users" in body["recommendations"][0]["reason"]


def test_recommend_requires_auth(client: TestClient, use_recommender: None) -> None:
    assert client.get("/api/recommend").status_code == 401


def test_missing_artifacts_return_503(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_recommender] = lambda: RecommendationService(
        tmp_path / "empty"
    )
    try:
        headers, _ = signup()
        response = client.get("/api/recommend", headers=headers)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "train_and_precompute" in response.json()["detail"]
