"""Phase 1.2-1.6: catalog, chunked ratings, filtering, sampling, splits, maps, CSR."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.data import (
    PipelineConfig,
    apply_id_map,
    build_csr,
    build_id_map,
    chronological_split,
    iter_rating_chunks,
    model_movie_ids,
    read_movies,
    run_pipeline,
    scan_ratings,
    select_sample_users,
    verify_raw_dir,
    write_filtered_ratings,
)

MOVIES = [
    (1, "Matrix, The (1999)", "Action|Sci-Fi"),
    (2, "Toy Story (1995)", "Animation|Children|Comedy"),
    (3, "Heat (1995)", "Action|Crime|Thriller"),
    (4, "Rare Gem (2001)", "Drama"),
    (5, "No Genre Movie (2010)", "(no genres listed)"),
    (6, "Recent Movie (2021)", "Drama"),
]

RATINGS = [
    (1, 1, 5.0, 100),
    (1, 2, 4.5, 110),
    (1, 3, 3.5, 120),
    (1, 6, 4.0, 130),
    (2, 1, 4.0, 200),
    (2, 2, 3.0, 210),
    (2, 3, 4.5, 220),
    (2, 6, 2.5, 230),
    (3, 1, 3.5, 300),
    (3, 2, 5.0, 310),
    (3, 3, 4.0, 320),
    (4, 1, 2.5, 400),
    (4, 2, 3.5, 410),
    (5, 1, 4.0, 500),
    (5, 6, 3.0, 510),
    (6, 4, 4.0, 600),
]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A tiny ml-32m-shaped directory with known rating counts."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(MOVIES, columns=["movieId", "title", "genres"]).to_csv(
        raw / "movies.csv", index=False
    )
    pd.DataFrame(
        [
            (1, 111, 603),
            (2, 222, 862),
            (3, 333, None),
            (4, 444, 999),
            (5, 555, 1000),
            (6, 666, 1001),
        ],
        columns=["movieId", "imdbId", "tmdbId"],
    ).to_csv(raw / "links.csv", index=False)
    pd.DataFrame(RATINGS, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        raw / "ratings.csv", index=False
    )
    pd.DataFrame(
        [(1, 1, "dystopia", 100)],
        columns=["userId", "movieId", "tag", "timestamp"],
    ).to_csv(raw / "tags.csv", index=False)
    return raw


def snapshot(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------
def test_read_movies_builds_the_full_catalog(raw_dir: Path) -> None:
    catalog = read_movies(raw_dir)
    assert len(catalog) == len(MOVIES)
    assert list(catalog.columns) == [
        "movie_id",
        "title_raw",
        "title",
        "title_display",
        "title_search",
        "release_year",
        "genres",
        "source",
        "imdb_id",
        "tmdb_id",
    ]
    matrix = catalog.set_index("movie_id")
    assert matrix.loc[1, "title_display"] == "The Matrix"
    assert matrix.loc[1, "title_search"] == "the matrix"
    assert matrix.loc[1, "release_year"] == 1999
    assert matrix.loc[3, "tmdb_id"] is pd.NA or pd.isna(matrix.loc[3, "tmdb_id"])
    assert matrix.loc[6, "release_year"] == 2021
    assert (catalog["source"] == "movielens").all()


def test_verify_raw_dir_explains_how_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        verify_raw_dir(tmp_path / "missing")
    assert "grouplens.org" in str(excinfo.value)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        verify_raw_dir(empty)
    assert "ratings.csv" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Chunked ratings and filters
# --------------------------------------------------------------------------------------
def test_iter_rating_chunks_streams_in_bounded_chunks(raw_dir: Path) -> None:
    chunks = list(iter_rating_chunks(raw_dir / "ratings.csv", chunksize=3))
    assert len(chunks) == 6
    assert all(len(chunk) <= 3 for chunk in chunks)
    combined = pd.concat(chunks, ignore_index=True)
    assert len(combined) == len(RATINGS)
    assert str(combined["rating"].dtype) == "float32"
    assert set(combined.columns) == {"userId", "movieId", "rating", "timestamp"}


def test_scan_ratings_counts_everything_once(raw_dir: Path) -> None:
    scan = scan_ratings(raw_dir / "ratings.csv", chunksize=5)
    assert scan.n_ratings == len(RATINGS)
    assert scan.n_users == 6
    # movie 5 exists in the catalog but has no ratings, so the ratings scan never sees it.
    assert scan.n_movies == 5
    assert scan.movie_counts.loc[1] == 5
    assert scan.movie_counts.loc[4] == 1
    assert scan.user_counts.loc[6] == 1
    assert scan.movie_sums.loc[1] == pytest.approx(19.0)
    assert int(scan.rating_histogram.sum()) == len(RATINGS)
    assert scan.rating_histogram.index.min() == 2.5
    assert scan.rating_histogram.index.max() == 5.0


def test_invalid_rating_values_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        [(1, 1, 4.25, 10)], columns=["userId", "movieId", "rating", "timestamp"]
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="half-star"):
        list(iter_rating_chunks(bad))

    out_of_range = tmp_path / "range.csv"
    pd.DataFrame(
        [(1, 1, 6.0, 10)], columns=["userId", "movieId", "rating", "timestamp"]
    ).to_csv(out_of_range, index=False)
    with pytest.raises(ValueError, match="0.5-5.0"):
        list(iter_rating_chunks(out_of_range))


def test_model_movie_ids_applies_the_item_threshold(raw_dir: Path) -> None:
    scan = scan_ratings(raw_dir / "ratings.csv", chunksize=4)
    assert model_movie_ids(scan, min_item_ratings=3).tolist() == [1, 2, 3, 6]
    assert model_movie_ids(scan, min_item_ratings=1).tolist() == [1, 2, 3, 4, 6]


def test_select_sample_users_filters_and_respects_budgets(raw_dir: Path) -> None:
    scan = scan_ratings(raw_dir / "ratings.csv", chunksize=4)

    everyone = select_sample_users(scan, min_user_ratings=2, max_users=None, max_ratings=None)
    assert everyone.tolist() == [1, 2, 3, 4, 5]

    sampled = select_sample_users(scan, min_user_ratings=2, max_users=3, seed=7)
    assert sampled.size == 3
    assert set(sampled).issubset({1, 2, 3, 4, 5})
    assert sampled.tolist() == select_sample_users(
        scan, min_user_ratings=2, max_users=3, seed=7
    ).tolist()

    budgeted = select_sample_users(scan, min_user_ratings=2, max_users=None, max_ratings=9)
    counts = scan.user_counts.loc[budgeted]
    assert counts.sum() <= 9
    assert len(budgeted) < 5


def test_write_filtered_ratings_keeps_only_the_sample(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "filtered.parquet"
    rows = write_filtered_ratings(
        raw_dir / "ratings.csv",
        out,
        user_ids=[1, 2, 3],
        movie_ids=[1, 2, 3, 6],
        chunksize=4,
    )
    kept = pd.read_parquet(out)
    assert rows == len(kept) == 11
    assert set(kept["userId"]) == {1, 2, 3}
    assert set(kept["movieId"]) == {1, 2, 3, 6}
    assert str(kept["rating"].dtype) == "float32"


# --------------------------------------------------------------------------------------
# Chronological split and ID maps
# --------------------------------------------------------------------------------------
def test_chronological_split_holds_out_the_most_recent_rows_per_user() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "movieId": [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
            "rating": [3.0] * 10,
            "timestamp": [50, 10, 30, 40, 20, 100, 200, 300, 400, 500],
        }
    )
    train, test = chronological_split(frame, test_fraction=0.4)

    assert len(train) == 6
    assert len(test) == 4
    for user in (1, 2):
        user_test = test[test["userId"] == user]["timestamp"]
        user_train = train[train["userId"] == user]["timestamp"]
        assert len(user_test) == 2
        assert user_test.min() > user_train.max()

    # Row order inside each split is chronological, with the original ids preserved.
    assert test[test["userId"] == 1]["timestamp"].tolist() == [40, 50]


def test_chronological_split_never_starves_a_user_of_training_data() -> None:
    frame = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 11], "rating": [4.0, 4.5], "timestamp": [1, 2]}
    )
    train, test = chronological_split(frame, test_fraction=0.2)
    assert len(train) == 2
    assert test.empty


def test_build_id_map_is_compact_sorted_and_reversible() -> None:
    user_map = build_id_map([9, 3, 9, 7], "userId", "user_index")
    assert user_map["userId"].tolist() == [3, 7, 9]
    assert user_map["user_index"].tolist() == [0, 1, 2]

    movie_map = build_id_map([50, 10], "movieId", "movie_index")
    frame = pd.DataFrame({"userId": [3, 9], "movieId": [10, 50], "rating": [4.0, 5.0]})
    mapped = apply_id_map(frame, user_map, movie_map)
    assert mapped["user_index"].tolist() == [0, 2]
    assert mapped["movie_index"].tolist() == [0, 1]


def test_apply_id_map_rejects_unmapped_ids() -> None:
    user_map = build_id_map([1], "userId", "user_index")
    movie_map = build_id_map([10], "movieId", "movie_index")
    frame = pd.DataFrame({"userId": [1], "movieId": [99], "rating": [4.0]})
    with pytest.raises(KeyError, match="unmapped movies"):
        apply_id_map(frame, user_map, movie_map)


def test_build_csr_is_sparse_and_rejects_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "user_index": [0, 0, 1],
            "movie_index": [0, 2, 1],
            "rating": [5.0, 3.5, 4.0],
        }
    )
    matrix = build_csr(frame, n_users=2, n_movies=3)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (2, 3)
    assert matrix.nnz == 3
    assert matrix[0, 2] == pytest.approx(3.5)
    assert matrix[1, 0] == 0.0  # never observed, never a rating

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_csr(duplicated, n_users=2, n_movies=3)


# --------------------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------------------
def test_run_pipeline_produces_every_artifact_and_leaves_raw_untouched(
    raw_dir: Path, tmp_path: Path
) -> None:
    before = snapshot(raw_dir)
    processed = tmp_path / "processed"
    manifest = run_pipeline(
        PipelineConfig(
            raw_dir=raw_dir,
            processed_dir=processed,
            min_item_ratings=3,
            min_user_ratings=2,
            max_users=4,
            max_ratings=1_000,
            test_fraction=0.25,
            seed=3,
            chunk_size=3,
        ),
        log=lambda *_: None,
    )

    assert snapshot(raw_dir) == before  # raw data is read-only
    for name in (
        "catalog.parquet",
        "ratings_sample.parquet",
        "ratings_train.parquet",
        "ratings_test.parquet",
        "user_id_map.parquet",
        "movie_id_map.parquet",
        "train_matrix.npz",
        "manifest.json",
    ):
        assert (processed / name).is_file(), name

    stored = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert manifest["raw"]["ratings"] == len(RATINGS)
    assert manifest["model_universe"]["eligible_movies"] == 4
    assert manifest["model_universe"]["sampled_users"] == 4
    assert manifest["sample"]["users"] == 4
    # Only the four model-universe movies can appear in the sample.
    sample = pd.read_parquet(processed / "ratings_sample.parquet")
    assert set(sample["movieId"]).issubset({1, 2, 3, 6})
    assert manifest["sample"]["ratings"] == len(sample)
    assert manifest["split"]["train_rows"] + manifest["split"]["test_rows"] == len(sample)

    catalog = pd.read_parquet(processed / "catalog.parquet")
    assert len(catalog) == len(MOVIES)
    assert int(catalog["in_model"].sum()) == 4
    assert set(catalog.loc[catalog["in_model"], "movie_id"]) == {1, 2, 3, 6}

    train = pd.read_parquet(processed / "ratings_train.parquet")
    test = pd.read_parquet(processed / "ratings_test.parquet")
    assert {"user_index", "movie_index"} <= set(train.columns)
    assert {"user_index", "movie_index"} <= set(test.columns)
    assert len(set(zip(train["user_index"], train["movie_index"], strict=True)) & set(
        zip(test["user_index"], test["movie_index"], strict=True)
    )) == 0

    matrix = sparse.load_npz(processed / "train_matrix.npz")
    assert matrix.shape == (
        manifest["matrix"]["shape"][0],
        manifest["matrix"]["shape"][1],
    )
    assert matrix.nnz == len(train)
    # No training cell for a pair that only exists in the test split.
    train_pairs = set(zip(train["user_index"], train["movie_index"], strict=True))
    for user_index, movie_index in zip(
        test["user_index"], test["movie_index"], strict=True
    ):
        if (user_index, movie_index) not in train_pairs:
            assert matrix[user_index, movie_index] == 0.0

    for user in train["userId"].unique():
        last_train = train.loc[train["userId"] == user, "timestamp"].max()
        if user in set(test["userId"]):
            assert test.loc[test["userId"] == user, "timestamp"].min() >= last_train

    assert manifest["matrix"]["memory_bytes"] > 0
    assert manifest["matrix"]["memory_mb"] >= 0
    assert manifest["files"]["catalog.parquet"]["bytes"] > 0


def test_run_pipeline_refuses_to_write_inside_raw(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="raw data directory"):
        run_pipeline(
            PipelineConfig(
                raw_dir=raw_dir,
                processed_dir=raw_dir / "processed",
                min_item_ratings=3,
                min_user_ratings=2,
            ),
            log=lambda *_: None,
        )


def test_split_is_not_random(tmp_path: Path) -> None:
    """Same rows, different timestamps: the chronological split must follow time."""
    frame = pd.DataFrame(
        {
            "userId": [1] * 4,
            "movieId": [10, 11, 12, 13],
            "rating": [3.0] * 4,
            "timestamp": [1, 2, 3, 4],
        }
    )
    _, test = chronological_split(frame, test_fraction=0.5)
    assert test["movieId"].tolist() == [12, 13]
    assert np.all(test["timestamp"].to_numpy() > 2)
