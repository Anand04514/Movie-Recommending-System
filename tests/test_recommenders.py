"""Phase 2.1-2.3: baselines and item-based CF, tested without a database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from recsys.baselines import (
    damped_item_mean,
    global_mean,
    item_counts_and_sums,
    item_means,
)
from recsys.item_cf import (
    build_item_neighbors,
    load_neighbors,
    mean_center,
    popularity_recommendations,
    recommend_from_ratings,
    save_neighbors,
)

USERS = 3
ITEMS = 4
RATINGS = [
    (0, 0, 5.0),
    (0, 1, 4.0),
    (0, 2, 1.0),
    (1, 0, 4.0),
    (1, 1, 5.0),
    (1, 2, 2.0),
    (2, 1, 4.0),
    (2, 2, 5.0),
    (2, 3, 1.0),
]


def build_matrix() -> sparse.csr_matrix:
    rows = [row for row, _, _ in RATINGS]
    cols = [col for _, col, _ in RATINGS]
    values = [value for _, _, value in RATINGS]
    return sparse.csr_matrix((values, (rows, cols)), shape=(USERS, ITEMS), dtype="float32")


def test_global_mean_and_item_stats() -> None:
    matrix = build_matrix()
    assert global_mean(matrix) == pytest.approx(31 / 9)

    counts, sums = item_counts_and_sums(matrix)
    assert counts.tolist() == [2, 3, 3, 1]
    assert sums.tolist() == pytest.approx([9.0, 13.0, 8.0, 1.0])
    assert item_means(matrix).tolist() == pytest.approx([4.5, 13 / 3, 8 / 3, 1.0])


def test_damped_item_mean_shrinks_small_samples_toward_the_global_mean() -> None:
    matrix = build_matrix()
    mu = global_mean(matrix)
    undamped = damped_item_mean(matrix, damping=0.0)
    assert undamped.tolist() == pytest.approx(item_means(matrix).tolist())

    damped = damped_item_mean(matrix, damping=10.0)
    # item 3 has a single rating of 1.0, so damping pulls it strongly toward mu.
    assert damped[3] > undamped[3]
    assert damped[3] < mu + 0.5
    # item 0 has two high ratings; damping moves it down toward mu.
    assert damped[0] < undamped[0]


def test_mean_center_subtracts_only_observed_ratings() -> None:
    matrix = build_matrix()
    centered, means = mean_center(matrix)
    assert means.tolist() == pytest.approx([10 / 3, 11 / 3, 10 / 3])
    row_sums = np.asarray(centered.sum(axis=1)).ravel().tolist()
    assert row_sums == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)
    # Unrated cells stay absent from the sparse matrix.
    assert centered[0, 3] == 0.0
    assert centered.nnz <= matrix.nnz


def test_build_item_neighbors_finds_corated_items_and_excludes_self() -> None:
    neighbors = build_item_neighbors(build_matrix(), k=2, block_size=2)
    assert neighbors.indices.shape == (ITEMS, 2)
    assert neighbors.similarities.shape == (ITEMS, 2)
    assert neighbors.counts.tolist() == [2, 3, 3, 1]

    top_first = int(neighbors.indices[0, 0])
    assert top_first == 1  # items 0 and 1 are rated together and both highly
    assert neighbors.similarities[0, 0] > 0
    assert top_first != 0
    assert neighbors.indices[0].tolist().count(0) == 0

    # Every stored similarity is positive; padding uses -1.
    stored = neighbors.similarities > 0
    assert np.all(neighbors.indices[stored] >= 0)
    assert np.all(neighbors.similarities[~stored] == 0)


def test_build_item_neighbors_is_deterministic() -> None:
    first = build_item_neighbors(build_matrix(), k=2, block_size=3)
    second = build_item_neighbors(build_matrix(), k=2, block_size=3)
    assert np.array_equal(first.indices, second.indices)
    assert np.allclose(first.similarities, second.similarities)


def test_recommend_from_ratings_uses_neighbors_and_excludes_rated_items() -> None:
    matrix = build_matrix()
    neighbors = build_item_neighbors(matrix, k=3, block_size=2)
    recs = recommend_from_ratings(
        neighbors,
        rated_indices=np.array([0, 2]),
        rated_values=np.array([5.0, 1.0]),
        limit=3,
    )
    assert recs, "expected at least one recommendation"
    indexes = [rec.item_index for rec in recs]
    assert 0 not in indexes and 2 not in indexes
    assert 1 in indexes
    for rec in recs:
        assert 0.5 <= rec.predicted_rating <= 5.0
        assert rec.source_index in (0, 2)
        assert rec.source_index not in (rec.item_index,)


def test_recommend_from_ratings_excludes_unrated_but_blocked_items() -> None:
    neighbors = build_item_neighbors(build_matrix(), k=3, block_size=2)
    recs = recommend_from_ratings(
        neighbors,
        rated_indices=np.array([0, 2]),
        rated_values=np.array([5.0, 1.0]),
        exclude={1},
    )
    assert all(rec.item_index != 1 for rec in recs)


def test_cold_start_uses_popularity_and_respects_exclusions() -> None:
    neighbors = build_item_neighbors(build_matrix(), k=2, block_size=2)
    recs = popularity_recommendations(neighbors, limit=3, exclude={1})
    assert [rec.item_index for rec in recs] == [2, 0, 3]
    assert all(rec.source_index is None for rec in recs)
    assert recs[0].predicted_rating == pytest.approx(8 / 3)


def test_neighbors_roundtrip_through_disk(tmp_path: Path) -> None:
    neighbors = build_item_neighbors(build_matrix(), k=2, block_size=2)
    path = tmp_path / "item_neighbors.npz"
    save_neighbors(neighbors, path)
    loaded = load_neighbors(path)
    assert np.array_equal(loaded.indices, neighbors.indices)
    assert np.allclose(loaded.similarities, neighbors.similarities)
    assert loaded.counts.tolist() == neighbors.counts.tolist()
    assert loaded.n_users == USERS


def test_load_neighbors_explains_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_and_precompute"):
        load_neighbors(tmp_path / "missing.npz")


def test_recommendations_carry_ranked_contributions() -> None:
    neighbors = build_item_neighbors(build_matrix(), k=3, block_size=2)
    recs = recommend_from_ratings(
        neighbors,
        rated_indices=np.array([0, 2]),
        rated_values=np.array([5.0, 1.0]),
        limit=3,
        explain=2,
    )

    assert recs
    for rec in recs:
        assert rec.contributions, "a recommendation must explain itself"
        strengths = [contribution.strength for contribution in rec.contributions]
        assert strengths == sorted(strengths, reverse=True)
        assert all(contribution.strength > 0 for contribution in rec.contributions)
        assert all(contribution.item_index in (0, 2) for contribution in rec.contributions)
        assert len(rec.contributions) <= 2
        assert rec.source_index == rec.contributions[0].item_index


def test_explain_controls_how_much_evidence_is_returned() -> None:
    neighbors = build_item_neighbors(build_matrix(), k=3, block_size=2)
    rated = np.array([0, 2])
    values = np.array([5.0, 1.0])

    single = recommend_from_ratings(neighbors, rated, values, limit=2, explain=1)
    assert all(len(rec.contributions) == 1 for rec in single)

    none = recommend_from_ratings(neighbors, rated, values, limit=2, explain=0)
    assert all(rec.contributions == () and rec.source_index is None for rec in none)

    with pytest.raises(ValueError, match="explain must be non-negative"):
        recommend_from_ratings(neighbors, rated, values, explain=-1)
