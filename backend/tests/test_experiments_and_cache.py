"""Phase 5.8/5.9: A/B assignment and the precomputed candidate cache."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse
from sqlalchemy import delete, update

from backend.app.config import Settings, get_settings
from backend.app.db import get_session_factory
from backend.app.experiments import (
    DEFAULT_MODEL_EXPERIMENT,
    EXPERIMENTS,
    Experiment,
    bucket_for,
    get_experiment,
)
from backend.app.main import app
from backend.app.models import ExperimentAssignment, PrecomputedRec, RecommendationLog
from backend.app.precompute import warm_user
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
from backend.app.routers import recommendations as rec_router
from recsys.content import build_content_features, save_features
from recsys.item_cf import build_item_neighbors, save_neighbors
from recsys.svd import save_model, train_svd
from recsys.user_cf import build_user_neighbors, save_user_neighbors

MOVIE_IDS = [10, 11, 12, 13, 14, 15, 16, 17]
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
        item, user, content, processed_dir=tmp_path, base="user_cf", weight=0.8
    )
    overrides = {
        get_recommender: item,
        get_user_cf_recommender: user,
        get_content_recommender: content,
        get_svd_recommender: svd,
        get_hybrid_recommender: hybrid,
    }
    def provider(service: Any):
        # A closure, not a default argument: FastAPI would treat a default as a query parameter
        # (and try to deep-copy the service, lock included).
        def _service():
            return service

        return _service

    for dependency, service in overrides.items():
        app.dependency_overrides[dependency] = provider(service)
    # The cache trusts rows newer than the serving artifacts, so point it at the test artifacts.
    original = rec_router.get_settings
    rec_router.get_settings = lambda: Settings(
        processed_dir=str(tmp_path), precompute_ttl_minutes=60
    )
    try:
        yield {
            "item_cf": item,
            "user_cf": user,
            "svd": svd,
            "content": content,
            "hybrid": hybrid,
        }
    finally:
        rec_router.get_settings = original
        get_settings.cache_clear()
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


# ------------------------------------------------------------------ experiment assignment


def test_assignment_is_deterministic_and_exactly_sticky() -> None:
    experiment = get_experiment()

    first = bucket_for(experiment, 4242)
    assert first in experiment.variants
    assert bucket_for(experiment, 4242) == first  # no randomness at all
    assert bucket_for(experiment, 4243) != ""  # and it is defined for every id


def test_bucketing_splits_a_population_near_evenly() -> None:
    experiment = Experiment(name="split_probe", variants=("a", "b"), description="")
    counts = {"a": 0, "b": 0}
    for user_id in range(1, 2001):
        counts[bucket_for(experiment, user_id)] += 1

    # A sticky hash must still be a real split, not a pile-up on one arm.
    assert counts["a"] + counts["b"] == 2000
    assert 0.42 <= counts["a"] / 2000 <= 0.58


def test_omitting_the_algorithm_assigns_a_variant_and_reports_it(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    body = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()

    assert body["experiment"] == DEFAULT_MODEL_EXPERIMENT
    assert body["variant"] in EXPERIMENTS[DEFAULT_MODEL_EXPERIMENT].variants
    assert body["algorithm"] == body["variant"]
    assert body["recommendations"]

    with get_session_factory()() as session:
        rows = session.query(ExperimentAssignment).all()
    assert len(rows) == 1
    assert rows[0].variant == body["variant"]


def test_a_returning_user_keeps_the_same_arm(client: TestClient, customer, services) -> None:
    headers, _ = customer
    first = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()

    # Move the stored arm to the other variant: the row, not the hash, is authoritative.
    other = next(
        variant
        for variant in EXPERIMENTS[DEFAULT_MODEL_EXPERIMENT].variants
        if variant != first["variant"]
    )
    with get_session_factory()() as session:
        session.execute(update(ExperimentAssignment).values(variant=other))
        session.commit()

    second = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    assert second["variant"] == other


def test_an_explicit_choice_is_not_counted_as_experiment_traffic(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    body = client.get(
        "/api/recommend", params={"algorithm": "item_cf", "limit": 3}, headers=headers
    ).json()

    assert body["algorithm"] == "item_cf"
    assert body["experiment"] is None and body["variant"] is None

    with get_session_factory()() as session:
        assert session.query(ExperimentAssignment).count() == 0
        variants = {row.variant for row in session.query(RecommendationLog).all()}
    assert variants == {None}


def test_unknown_experiment_is_rejected(client: TestClient, customer, services) -> None:
    headers, _ = customer
    response = client.get(
        "/api/recommend", params={"experiment": "nope"}, headers=headers
    )
    assert response.status_code == 422
    assert "unknown experiment" in response.json()["detail"]


# ------------------------------------------------------------------ engagement report


def test_report_counts_only_outcomes_that_followed_the_impression(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    shown = client.get("/api/recommend", params={"limit": 4}, headers=headers).json()
    variant = shown["variant"]
    picked = shown["recommendations"][0]["movie"]["id"]
    other = shown["recommendations"][1]["movie"]["id"]

    # A verdict recorded before the film was ever shown must not count for this arm.
    with get_session_factory()() as session:
        session.execute(
            update(RecommendationLog).values(shown_at=datetime.now(UTC))
        )
        session.commit()
    client.post(
        "/api/feedback",
        json={"movie_id": other, "action": "liked", "source_algorithm": variant},
        headers=headers,
    )
    client.post(
        "/api/feedback",
        json={"movie_id": picked, "action": "rejected", "source_algorithm": variant},
        headers=headers,
    )

    reports = client.get("/api/experiments").json()
    assert [report["name"] for report in reports] == list(EXPERIMENTS)
    arm = next(v for v in reports[0]["variants"] if v["variant"] == variant)
    assert arm["users"] == 1
    assert arm["impressions"] == len(shown["recommendations"])
    assert arm["likes"] == 1
    assert arm["rejects"] == 1
    assert arm["positive_rate"] == pytest.approx(0.5)


def test_report_is_empty_but_shaped_before_any_traffic(client: TestClient) -> None:
    reports = client.get("/api/experiments").json()
    variants = reports[0]["variants"]
    assert {v["variant"] for v in variants} == set(EXPERIMENTS[DEFAULT_MODEL_EXPERIMENT].variants)
    for arm in variants:
        assert arm["users"] == 0
        assert arm["impressions"] == 0
        assert arm["positive_rate"] is None
        assert arm["rated_rate"] is None


def test_rating_a_shown_pick_counts_as_engagement(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    shown = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    variant = shown["variant"]
    target = shown["recommendations"][0]["movie"]["id"]

    client.post("/api/ratings", json={"movie_id": target, "rating": 4.5}, headers=headers)

    arm = next(
        v for v in client.get("/api/experiments").json()[0]["variants"] if v["variant"] == variant
    )
    assert arm["rated_after"] >= 1
    assert arm["rated_rate"] == pytest.approx(arm["rated_after"] / arm["impressions"])


# ------------------------------------------------------------------ precomputed cache


def test_the_second_identical_request_is_served_from_cache(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    first = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    assert first["cached"] is False

    with get_session_factory()() as session:
        assert session.query(PrecomputedRec).count() > 0

    second = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    assert second["cached"] is True
    assert [item["movie"]["id"] for item in second["recommendations"]] == [
        item["movie"]["id"] for item in first["recommendations"]
    ]
    # Cached candidates still get live explanations and live exclusions.
    assert second["algorithm"] == first["algorithm"]
    assert second["recommendations"][0]["reason"].startswith(("Because you rated", "Discovery"))


def test_cache_hit_still_respects_new_exclusions(client: TestClient, customer, services) -> None:
    headers, _ = customer
    first = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    assert first["cached"] is False
    target = first["recommendations"][0]["movie"]["id"]

    client.post(
        "/api/feedback", json={"movie_id": target, "action": "rejected"}, headers=headers
    )
    second = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()

    ids = [item["movie"]["id"] for item in second["recommendations"]]
    assert target not in ids
    # The pool only ever held three candidates, so two survive the exclusion and the short list
    # is filled live rather than silently padded.
    assert ids and len(ids) == 2
    assert second["cached"] is False


def test_a_bigger_page_than_the_warmed_pool_goes_live(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    client.get("/api/recommend", params={"limit": 3}, headers=headers)
    body = client.get("/api/recommend", params={"limit": 8}, headers=headers).json()

    # Three warmed rows cannot fill an eight-film page, so the request goes to the model.
    assert body["cached"] is False
    assert body["recommendations"]
    assert body["algorithm"]


def test_filters_never_use_the_cache(client: TestClient, customer, services) -> None:
    headers, _ = customer
    client.get("/api/recommend", params={"limit": 3}, headers=headers)
    body = client.get(
        "/api/recommend", params={"limit": 3, "genres": "Action"}, headers=headers
    ).json()

    assert body["cached"] is False
    assert body["filters"]["genres"] == ["Action"]


def test_expired_rows_are_ignored(client: TestClient, customer, services) -> None:
    headers, _ = customer
    client.get("/api/recommend", params={"limit": 3}, headers=headers)
    stale = datetime.now(UTC) - timedelta(hours=3)
    with get_session_factory()() as session:
        session.execute(update(PrecomputedRec).values(computed_at=stale))
        session.commit()

    body = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    assert body["cached"] is False  # the TTL is one hour in this fixture


def test_rows_older_than_a_retrain_are_ignored(
    client: TestClient, customer, services, tmp_path: Path
) -> None:
    headers, _ = customer
    client.get("/api/recommend", params={"limit": 3}, headers=headers)

    # Pretend the artifacts were retrained after the rows were written.
    outdated = datetime.now(UTC) - timedelta(minutes=30)
    with get_session_factory()() as session:
        session.execute(update(PrecomputedRec).values(computed_at=outdated))
        session.commit()
    later = datetime.now(UTC).timestamp() + 60
    import os

    os.utime(tmp_path / "item_neighbors.npz", (later, later))

    body = client.get("/api/recommend", params={"limit": 3}, headers=headers).json()
    assert body["cached"] is False


def test_warm_up_job_fills_the_cache_without_serving(
    customer, services, tmp_path: Path
) -> None:
    headers, profile = customer
    with get_session_factory()() as session:
        written = warm_user(
            session, profile["id"], services, algorithms=("item_cf", "hybrid"), limit=6
        )
    assert written["item_cf"] > 0 and written["hybrid"] > 0

    with get_session_factory()() as session:
        rows = session.query(PrecomputedRec).all()
        ranks = sorted(row.rank for row in rows if row.algorithm == "item_cf")
    assert ranks == list(range(written["item_cf"]))  # contiguous, ordered ranks


def test_warm_up_skips_a_user_with_no_ratings(
    client: TestClient, signup, add_movies, services
) -> None:
    add_movies(*MOVIES)
    _, profile = signup(email="blank@example.com")
    with get_session_factory()() as session:
        assert warm_user(session, profile["id"], services, algorithms=("item_cf",), limit=5) == {}

    with get_session_factory()() as session:
        session.execute(delete(PrecomputedRec))
        session.commit()
