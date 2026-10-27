"""Phase 3.3: sampled user-based CF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from recsys.user_cf import build_user_neighbors, load_user_neighbors, save_user_neighbors

# Two taste clusters: users 0-2 rate items 0-1 highly and 2-3 badly, users 3-5 the reverse.
RATINGS = [
    (0, 0, 5.0),
    (0, 1, 4.0),
    (0, 2, 1.0),
    (0, 3, 1.5),
    (1, 0, 4.5),
    (1, 1, 5.0),
    (1, 2, 1.0),
    (1, 3, 1.0),
    (2, 0, 5.0),
    (2, 1, 4.5),
    (2, 2, 1.5),
    (2, 3, 1.5),
    (3, 0, 1.0),
    (3, 1, 1.5),
    (3, 2, 5.0),
    (3, 3, 4.0),
    (4, 0, 1.5),
    (4, 1, 1.0),
    (4, 2, 4.5),
    (4, 3, 5.0),
    (5, 0, 1.0),
    (5, 1, 1.0),
    (5, 2, 5.0),
    (5, 3, 4.5),
]


def build_matrix() -> sparse.csr_matrix:
    rows = [row for row, _, _ in RATINGS]
    cols = [col for _, col, _ in RATINGS]
    values = [value for _, _, value in RATINGS]
    return sparse.csr_matrix((values, (rows, cols)), shape=(6, 4), dtype="float32")


def test_neighbors_stay_inside_a_taste_cluster() -> None:
    neighbors = build_user_neighbors(build_matrix(), k=2, block_size=3)

    assert neighbors.indices.shape == (6, 2)
    assert {user for user, _ in neighbors.top_neighbors(0)} == {1, 2}
    assert {user for user, _ in neighbors.top_neighbors(4)} == {3, 5}
    assert all(similarity > 0 for _, similarity in neighbors.top_neighbors(0))
    # A user is never their own neighbor, even when they have no positive edges left.
    assert all(user != index for index in range(6) for user in neighbors.indices[index])


def test_padding_marks_users_without_enough_positive_neighbors() -> None:
    neighbors = build_user_neighbors(build_matrix(), k=5, block_size=2)

    stored = neighbors.similarities[0] > 0
    assert int(stored.sum()) == 2  # only the two cluster mates correlate
    assert np.all(neighbors.indices[0][~stored] == -1)
    assert np.all(neighbors.similarities[0][~stored] == 0)


def test_neighbors_are_deterministic_and_the_matrix_matches_the_table() -> None:
    first = build_user_neighbors(build_matrix(), k=2, block_size=2)
    second = build_user_neighbors(build_matrix(), k=2, block_size=5)
    assert np.array_equal(first.indices, second.indices)
    assert np.allclose(first.similarities, second.similarities)

    matrix = first.similarity_matrix()
    assert matrix.shape == (6, 6)
    assert matrix.nnz == 12  # six users, two positive edges each
    slot = int(np.flatnonzero(first.indices[0] == 1)[0])
    assert matrix[0, 1] == pytest.approx(float(first.similarities[0, slot]))
    assert matrix[0, 1] > 0.0
    assert matrix[0, 0] == 0.0
    assert matrix[0, 3] == 0.0  # the other cluster is never an edge


def test_user_statistics_come_from_the_rating_matrix() -> None:
    neighbors = build_user_neighbors(build_matrix(), k=2)
    assert neighbors.counts.tolist() == [4, 4, 4, 4, 4, 4]
    assert neighbors.means.tolist() == pytest.approx(
        [2.875, 2.875, 3.125, 2.875, 3.0, 2.875]
    )
    assert neighbors.n_items == 4
    assert neighbors.n_users == 6


def test_neighbors_roundtrip_through_disk(tmp_path: Path) -> None:
    neighbors = build_user_neighbors(build_matrix(), k=2, block_size=3)
    path = tmp_path / "user_neighbors.npz"
    save_user_neighbors(neighbors, path)

    loaded = load_user_neighbors(path)
    assert np.array_equal(loaded.indices, neighbors.indices)
    assert np.allclose(loaded.similarities, neighbors.similarities)
    assert loaded.counts.tolist() == neighbors.counts.tolist()
    assert loaded.n_items == neighbors.n_items


def test_build_and_load_validate_their_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        build_user_neighbors(build_matrix(), k=0)
    with pytest.raises(ValueError, match="block_size must be positive"):
        build_user_neighbors(build_matrix(), block_size=0)
    with pytest.raises(ValueError, match="at least two users"):
        build_user_neighbors(sparse.csr_matrix((1, 4), dtype="float32"))
    with pytest.raises(IndexError, match="outside the neighbor table"):
        build_user_neighbors(build_matrix(), k=1).top_neighbors(99)
    with pytest.raises(FileNotFoundError, match="evaluate_models"):
        load_user_neighbors(tmp_path / "missing.npz")
