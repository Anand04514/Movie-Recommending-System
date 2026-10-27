"""User-based collaborative filtering for development-scale evaluation.

Similarity is cosine over per-user mean-centered vectors, so only co-rated items contribute and
missing ratings are never treated as zeros. Similarities are computed one block of users at a time,
so the full users x users matrix is never materialized; the top-K edges are saved and reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from recsys.item_cf import mean_center


@dataclass(frozen=True)
class UserNeighbors:
    """Top-K user neighbors plus the per-user statistics the fold-in needs."""

    indices: np.ndarray  # (n_users, k) int32, -1 where fewer than k neighbors exist
    similarities: np.ndarray  # (n_users, k) float32
    counts: np.ndarray  # (n_users,) int64
    means: np.ndarray  # (n_users,) float64
    n_items: int
    min_similarity: float

    @property
    def n_users(self) -> int:
        return int(self.indices.shape[0])

    def similarity_matrix(self) -> sparse.csr_matrix:
        """The neighbor table as a users x users CSR matrix holding positive edges only."""
        n_users, k = self.indices.shape
        rows = np.repeat(np.arange(n_users, dtype="int64"), k)
        cols = self.indices.reshape(-1).astype("int64")
        values = self.similarities.reshape(-1).astype("float64")
        keep = (cols >= 0) & (values > 0)
        return sparse.csr_matrix(
            (values[keep], (rows[keep], cols[keep])), shape=(n_users, n_users), dtype="float64"
        )

    def top_neighbors(self, user_index: int, *, limit: int = 10) -> list[tuple[int, float]]:
        """Neighbors of one user as ``(user_index, similarity)``, strongest first."""
        if not 0 <= user_index < self.n_users:
            raise IndexError(f"user index {user_index} is outside the neighbor table")
        pairs = [
            (int(neighbor), float(similarity))
            for neighbor, similarity in zip(
                self.indices[user_index], self.similarities[user_index], strict=True
            )
            if neighbor >= 0 and similarity > 0
        ]
        return pairs[:limit]


def build_user_neighbors(
    matrix: sparse.spmatrix,
    *,
    k: int = 50,
    block_size: int = 512,
    min_similarity: float = 1e-6,
) -> UserNeighbors:
    """Compute cosine similarity between users block by block and keep the top ``k`` each.

    Each block is at most ``block_size x n_users`` and is discarded after its neighbors are
    selected, so memory stays bounded on a development subset.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    matrix = matrix.tocsr()
    n_users, n_items = matrix.shape
    if n_users < 2:
        raise ValueError("need at least two users to build neighbors")

    centered, means = mean_center(matrix)
    norms = np.sqrt(np.asarray(centered.multiply(centered).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms)
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    normalized = (sparse.diags(inverse) @ centered).tocsr()

    take = min(k, n_users - 1)
    indices = np.full((n_users, k), -1, dtype="int32")
    similarities = np.zeros((n_users, k), dtype="float32")

    for start in range(0, n_users, block_size):
        stop = min(start + block_size, n_users)
        block = (normalized[start:stop] @ normalized.T).toarray().astype("float32")
        width = stop - start
        block[np.arange(width), np.arange(start, stop)] = 0.0  # a user is not their own neighbor
        block[block < min_similarity] = 0.0

        top = np.argpartition(-block, kth=take - 1, axis=1)[:, :take]
        values = np.take_along_axis(block, top, axis=1)
        order = np.argsort(-values, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)

        indices[start:stop, :take] = top
        similarities[start:stop, :take] = values

    empty = similarities <= 0
    similarities[empty] = 0.0
    indices[empty] = -1

    counts = np.diff(matrix.indptr).astype("int64")
    return UserNeighbors(
        indices=indices,
        similarities=similarities,
        counts=counts,
        means=means,
        n_items=int(n_items),
        min_similarity=float(min_similarity),
    )


def save_user_neighbors(neighbors: UserNeighbors, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=neighbors.indices,
        similarities=neighbors.similarities,
        counts=neighbors.counts,
        means=neighbors.means,
        n_items=np.int64(neighbors.n_items),
        min_similarity=np.float64(neighbors.min_similarity),
    )


def load_user_neighbors(path: Path) -> UserNeighbors:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"user neighbors not found at {path}; run scripts/evaluate_models.py to build them"
        )
    with np.load(path) as data:
        return UserNeighbors(
            indices=data["indices"].astype("int32"),
            similarities=data["similarities"].astype("float32"),
            counts=data["counts"].astype("int64"),
            means=data["means"].astype("float64"),
            n_items=int(data["n_items"]),
            min_similarity=float(data["min_similarity"]),
        )
