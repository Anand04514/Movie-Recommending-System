"""PLAN 5.11: the retrain job merges app ratings and promotes films that reach 20 ratings."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sqlalchemy import select

from backend.app.db import get_session_factory
from backend.app.models import Movie, Rating, User
from recsys.augment import SOURCE_APP, USER_ID_OFFSET
from recsys.item_cf import load_neighbors
from scripts.train_and_precompute import load_app_ratings
from scripts.train_and_precompute import main as train_main


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    """A tiny but complete set of artifacts, as prepare_data.py would leave them."""
    directory = tmp_path / "processed"
    directory.mkdir()
    matrix = sparse.csr_matrix(np.array([[4.0, 0.0], [0.0, 3.5]], dtype="float32"))
    sparse.save_npz(directory / "train_matrix.npz", matrix)
    pd.DataFrame({"movieId": [10, 20], "movie_index": [0, 1]}).to_parquet(
        directory / "movie_id_map.parquet", index=False
    )
    pd.DataFrame({"userId": [1, 2], "user_index": [0, 1]}).to_parquet(
        directory / "user_id_map.parquet", index=False
    )
    pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "rating_count": [1_000, 800, 0],
            "title_display": ["Alpha", "Beta", "Gamma"],
        }
    ).to_parquet(directory / "catalog.parquet", index=False)
    return directory


@pytest.fixture
def accounts(
    add_movies: Callable[..., list[int]],
) -> dict[str, int]:
    """A real account with two ratings, a seeded MovieLens profile, and three catalogue films."""
    add_movies(
        {"id": 10, "title": "Alpha (2001)", "in_model": True, "rating_count": 1_000},
        {"id": 20, "title": "Beta (2002)", "in_model": True, "rating_count": 800},
        {"id": 30, "title": "Gamma (2025)", "in_model": False, "rating_count": 0},
    )
    with get_session_factory()() as session:
        maya = User(
            email="maya@example.com",
            password_hash="x",
            display_name="Maya",
            is_seeded=False,
        )
        ravi = User(
            email="ravi@example.com",
            password_hash="x",
            display_name="Ravi",
            is_seeded=False,
        )
        demo_user = User(
            email="demo-1@movielens.invalid",
            password_hash="!",
            display_name="MovieLens user 1",
            is_seeded=True,
        )
        session.add_all([maya, ravi, demo_user])
        session.flush()
        session.add_all(
            [
                Rating(user_id=maya.id, movie_id=10, rating=5.0),
                Rating(user_id=maya.id, movie_id=30, rating=4.5),
                Rating(user_id=ravi.id, movie_id=30, rating=4.0),
                # A mirrored MovieLens profile is already in the training data; merging it again
                # would count the same person twice.
                Rating(user_id=demo_user.id, movie_id=20, rating=1.0),
            ]
        )
        session.commit()
        return {"maya": maya.id, "ravi": ravi.id, "demo": demo_user.id}


def _run(processed: Path, *extra: str) -> int:
    return train_main(
        [
            "--processed-dir",
            str(processed),
            "--include-app-ratings",
            "--no-svd",
            "--no-content",
            *extra,
        ]
    )


def test_only_real_accounts_are_read(accounts: dict[str, int]) -> None:
    frame = load_app_ratings()

    assert len(frame) == 3
    assert set(frame["movie_id"]) == {10, 30}
    assert sorted(frame["rating"]) == [4.0, 4.5, 5.0]


def test_a_film_that_reaches_the_threshold_joins_the_model(
    processed: Path, accounts: dict[str, int]
) -> None:
    assert _run(processed, "--min-item-ratings", "2") == 0

    matrix = sparse.load_npz(processed / "train_matrix.npz")
    movies = pd.read_parquet(processed / "movie_id_map.parquet")
    users = pd.read_parquet(processed / "user_id_map.parquet")

    # Two MovieLens users plus two app accounts; two model films plus the promoted one.
    assert matrix.shape == (4, 3)
    assert movies["movieId"].tolist() == [10, 20, 30]
    assert movies["source"].tolist()[:2] == ["movielens", "movielens"]
    assert movies.loc[2, "source"] == SOURCE_APP
    # Each app account is a real row, under a synthetic id that cannot collide with MovieLens.
    assert users["userId"].tolist() == [
        1,
        2,
        USER_ID_OFFSET + accounts["maya"],
        USER_ID_OFFSET + accounts["ravi"],
    ]
    # Their ratings land in the right cells: 5.0 on Alpha, 4.5 and 4.0 on the promoted Gamma.
    assert matrix[2, 0] == pytest.approx(5.0)
    assert matrix[2, 2] == pytest.approx(4.5)
    assert matrix[3, 2] == pytest.approx(4.0)

    report = json.loads((processed / "app_merge.json").read_text(encoding="utf-8"))
    assert report["ratings_merged"] == 3
    assert report["users_added"] == 2
    assert report["promoted"] == [30]
    assert report["min_item_ratings"] == 2


def test_a_film_below_the_threshold_stays_out(processed: Path, accounts: dict[str, int]) -> None:
    assert _run(processed, "--min-item-ratings", "4") == 0

    matrix = sparse.load_npz(processed / "train_matrix.npz")
    # Maya's rating on Alpha is still trainable, so her account joins; Gamma does not.
    assert matrix.shape == (3, 2)
    report = json.loads((processed / "app_merge.json").read_text(encoding="utf-8"))
    assert report["ratings_merged"] == 1
    assert report["below_threshold"] == [30]


def test_rerunning_the_retrain_does_not_count_the_same_ratings_twice(
    processed: Path, accounts: dict[str, int]
) -> None:
    assert _run(processed, "--min-item-ratings", "2") == 0
    first = sparse.load_npz(processed / "train_matrix.npz")

    assert _run(processed, "--min-item-ratings", "2") == 0
    second = sparse.load_npz(processed / "train_matrix.npz")

    assert second.shape == first.shape
    assert second.nnz == first.nnz
    assert (second != first).nnz == 0
    report = json.loads((processed / "app_merge.json").read_text(encoding="utf-8"))
    assert report["ratings_merged"] == 3
    assert report["previous_users_stripped"] == 2
    assert report["previous_movies_stripped"] == 1


def test_sync_db_marks_the_promoted_film(
    processed: Path, accounts: dict[str, int]
) -> None:
    assert _run(processed, "--min-item-ratings", "2", "--sync-db") == 0

    with get_session_factory()() as session:
        gamma = session.get(Movie, 30)
        assert gamma is not None
        assert gamma.in_model is True
        # The films that were already in the model are untouched.
        assert session.get(Movie, 10).in_model is True


def test_without_sync_db_the_catalogue_is_left_alone(
    processed: Path, accounts: dict[str, int]
) -> None:
    assert _run(processed, "--min-item-ratings", "2") == 0

    with get_session_factory()() as session:
        assert session.get(Movie, 30).in_model is False


def test_neighbor_artifacts_cover_the_promoted_column(
    processed: Path, accounts: dict[str, int]
) -> None:
    assert _run(processed, "--min-item-ratings", "2") == 0

    neighbors = load_neighbors(processed / "item_neighbors.npz")
    assert neighbors.counts.size == 3
    # Serving compares this map against the neighbor table, so they must agree.
    movies = pd.read_parquet(processed / "movie_id_map.parquet")
    assert len(movies) == neighbors.counts.size


def test_merging_without_the_user_map_fails_clearly(
    processed: Path, accounts: dict[str, int]
) -> None:
    (processed / "user_id_map.parquet").unlink()

    assert _run(processed, "--min-item-ratings", "2") == 1


def test_an_unmigrated_database_is_an_operator_error(
    processed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode() -> Any:
        raise RuntimeError("no such table: users")

    monkeypatch.setattr("scripts.train_and_precompute.load_app_ratings", explode)

    assert _run(processed, "--min-item-ratings", "2") == 1


def test_seeded_accounts_never_reach_the_training_matrix(
    processed: Path, accounts: dict[str, int]
) -> None:
    with get_session_factory()() as session:
        demo_ratings = session.scalars(
            select(Rating).where(Rating.user_id == accounts["demo"])
        ).all()
    assert demo_ratings, "the fixture must hold a seeded rating, or the test proves nothing"

    assert _run(processed, "--min-item-ratings", "2") == 0

    matrix = sparse.load_npz(processed / "train_matrix.npz")
    # Four rows: two MovieLens profiles plus two app accounts, with the seeded profile not added
    # a second time.
    assert matrix.shape[0] == 4
    assert matrix[1, 1] == pytest.approx(3.5)  # the seeded profile's original training cell
