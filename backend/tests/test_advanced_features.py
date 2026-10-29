"""Phase 5: advanced serving, the feedback loop and onboarding."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from backend.app.db import get_session_factory
from backend.app.main import app
from backend.app.models import RecommendationLog
from backend.app.recommendations import (
    ContentService,
    HybridService,
    RecommendationService,
    SvdService,
    UserCfService,
    get_content_recommender,
    get_hybrid_recommender,
    get_recommender,
    get_svd_recommender,
    get_user_cf_recommender,
)
from recsys.content import build_content_features, save_features
from recsys.item_cf import build_item_neighbors, save_neighbors
from recsys.svd import save_model, train_svd
from recsys.user_cf import build_user_neighbors, save_user_neighbors

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
RATED = ((10, 5.0), (11, 4.0), (12, 1.0), (13, 1.0))


def write_artifacts(directory: Path) -> None:
    matrix = sparse.csr_matrix(MATRIX)
    save_neighbors(
        build_item_neighbors(matrix, k=4, block_size=4), directory / "item_neighbors.npz"
    )
    save_user_neighbors(
        build_user_neighbors(matrix, k=3, block_size=3), directory / "user_neighbors.npz"
    )
    sparse.save_npz(directory / "train_matrix.npz", matrix)
    pd.DataFrame({"movieId": MOVIE_IDS, "movie_index": list(range(8))}).to_parquet(
        directory / "movie_id_map.parquet", index=False
    )
    save_model(train_svd(matrix, rank=3), directory / "svd_model.npz")

    catalog = pd.DataFrame(
        [
            {
                "movie_id": spec["id"],
                "genres": spec["genres"],
                "release_year": int(spec["title"][-5:-1]),
            }
            for spec in MOVIES
        ]
    )
    save_features(build_content_features(catalog), directory / "content_features.npz")


@pytest.fixture
def services(tmp_path: Path) -> Iterator[dict[str, Any]]:
    write_artifacts(tmp_path)
    item = RecommendationService(tmp_path)
    user = UserCfService(tmp_path)
    content = ContentService(tmp_path, fallback=item)
    svd = SvdService(tmp_path, fallback=item)
    hybrid = HybridService(
        item,
        user,
        content,
        processed_dir=tmp_path,
        base="user_cf",
        weight=0.8,
    )
    app.dependency_overrides[get_recommender] = lambda: item
    app.dependency_overrides[get_user_cf_recommender] = lambda: user
    app.dependency_overrides[get_content_recommender] = lambda: content
    app.dependency_overrides[get_svd_recommender] = lambda: svd
    app.dependency_overrides[get_hybrid_recommender] = lambda: hybrid
    yield {"item": item, "user": user, "content": content, "svd": svd, "hybrid": hybrid}
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


def test_hybrid_recommendations_come_from_both_models(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    body = client.get(
        "/api/recommend", params={"algorithm": "hybrid", "limit": 5}, headers=headers
    ).json()

    assert body["algorithm"] == "hybrid"
    assert body["cold_start"] is False
    assert body["needs_onboarding"] is True  # only four ratings so far
    ids = [item["movie"]["id"] for item in body["recommendations"]]
    assert 14 in ids  # the film this taste group loves, in both models' list
    assert not set(ids) & {10, 11, 12, 13}
    assert all(0.5 <= item["predicted_rating"] <= 5.0 for item in body["recommendations"])
    assert all(
        item["reason"].startswith(("Because you rated", "Discovery", "Popular"))
        or "with your taste" in item["reason"]
        for item in body["recommendations"]
    )


def test_svd_and_content_are_servable_algorithm_choices(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer

    svd = client.get(
        "/api/recommend", params={"algorithm": "svd", "limit": 4}, headers=headers
    ).json()
    assert svd["algorithm"] == "svd"
    assert svd["recommendations"]
    assert not {item["movie"]["id"] for item in svd["recommendations"]} & {10, 11, 12, 13}

    content = client.get(
        "/api/recommend", params={"algorithm": "content", "limit": 4}, headers=headers
    ).json()
    assert content["algorithm"] == "content"
    assert content["recommendations"]
    assert content["recommendations"][0]["evidence"]

    assert (
        client.get(
            "/api/recommend", params={"algorithm": "als"}, headers=headers
        ).status_code
        == 422
    )


def test_compare_can_contrast_any_two_methods(client: TestClient, customer, services) -> None:
    headers, _ = customer
    body = client.get(
        "/api/compare", params={"left": "svd", "right": "hybrid", "limit": 5}, headers=headers
    ).json()

    assert body["left"]["algorithm"] == "svd"
    assert body["right"]["algorithm"] == "hybrid"
    assert body["overlap_count"] == len(body["overlap"])
    assert body["left"]["recommendations"]


def test_rejecting_a_film_excludes_it_immediately_and_can_be_undone(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    before = client.get(
        "/api/recommend", params={"algorithm": "item_cf", "limit": 5}, headers=headers
    ).json()
    top = before["recommendations"][0]["movie"]["id"]

    posted = client.post(
        "/api/feedback",
        json={"movie_id": top, "action": "rejected", "source_algorithm": "item_cf"},
        headers=headers,
    )
    assert posted.status_code == 200
    assert posted.json()["action"] == "rejected"

    after = client.get(
        "/api/recommend", params={"algorithm": "item_cf", "limit": 5}, headers=headers
    ).json()
    assert top not in {item["movie"]["id"] for item in after["recommendations"]}

    listed = client.get("/api/feedback", headers=headers).json()
    assert [(row["movie_id"], row["action"]) for row in listed] == [(top, "rejected")]

    removed = client.delete(f"/api/feedback/{top}", headers=headers)
    assert removed.status_code == 204
    restored = client.get(
        "/api/recommend", params={"algorithm": "item_cf", "limit": 5}, headers=headers
    ).json()
    assert top in {item["movie"]["id"] for item in restored["recommendations"]}


def test_feedback_requires_auth_and_a_real_movie(client: TestClient, customer, services) -> None:
    headers, _ = customer
    assert client.post("/api/feedback", json={"movie_id": 10, "action": "liked"}).status_code == 401
    assert (
        client.post(
            "/api/feedback", json={"movie_id": 999_999, "action": "liked"}, headers=headers
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/feedback", json={"movie_id": 10, "action": "meh"}, headers=headers
        ).status_code
        == 422
    )
    assert client.delete("/api/feedback/14", headers=headers).status_code == 404


def test_every_shown_pick_is_logged(client: TestClient, customer, services) -> None:
    headers, _ = customer
    body = client.get(
        "/api/recommend", params={"algorithm": "item_cf", "limit": 3}, headers=headers
    ).json()
    shown = body["recommendations"]
    assert shown

    with get_session_factory()() as session:
        rows = session.query(RecommendationLog).order_by(RecommendationLog.id).all()
    assert len(rows) == len(shown)
    assert {row.algorithm for row in rows} == {"item_cf"}
    assert [row.movie_id for row in rows] == [item["movie"]["id"] for item in shown]
    assert all(row.score > 0 for row in rows)


def test_a_liked_film_also_leaves_the_pool(client: TestClient, customer, services) -> None:
    headers, _ = customer
    target = client.get(
        "/api/recommend", params={"algorithm": "hybrid", "limit": 5}, headers=headers
    ).json()["recommendations"][0]["movie"]["id"]

    client.post(
        "/api/feedback",
        json={"movie_id": target, "action": "liked", "source_algorithm": "hybrid"},
        headers=headers,
    )
    after = client.get(
        "/api/recommend", params={"algorithm": "hybrid", "limit": 5}, headers=headers
    ).json()
    assert target not in {item["movie"]["id"] for item in after["recommendations"]}


def test_onboarding_returns_a_diverse_shortlist(
    client: TestClient, add_movies
) -> None:
    add_movies(*MOVIES)
    response = client.get("/api/onboarding", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["target"] == 5
    assert len(body["movies"]) == 5
    assert len({movie["id"] for movie in body["movies"]}) == 5
    primary_genres = [movie["genres"][0] for movie in body["movies"]]
    assert len(set(primary_genres)) == 5  # one pick per genre before a second from any genre

    assert client.get("/api/onboarding", params={"limit": 2}).status_code == 422


def test_onboarding_works_on_an_empty_catalog(client: TestClient) -> None:
    body = client.get("/api/onboarding").json()
    assert body == {"target": 5, "movies": []}
