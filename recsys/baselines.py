"""Simple, hard-to-beat baselines: global mean, damped item mean, popularity.

All functions take a users x movies ``scipy.sparse`` matrix of raw ratings and never densify it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse


def global_mean(matrix: sparse.spmatrix) -> float:
    """Mean of every observed rating."""
    if matrix.nnz == 0:
        return 0.0
    return float(matrix.sum() / matrix.nnz)


def item_counts_and_sums(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Per-item rating counts and sums, vectorized over the sparse indexes."""
    n_items = matrix.shape[1]
    counts = np.bincount(matrix.indices, minlength=n_items).astype("int64")
    sums = np.bincount(
        matrix.indices, weights=matrix.data.astype("float64"), minlength=n_items
    )
    return counts, sums


def item_means(matrix: sparse.spmatrix) -> np.ndarray:
    """Per-item mean rating; 0 for items with no ratings."""
    counts, sums = item_counts_and_sums(matrix)
    means = np.zeros_like(sums, dtype="float64")
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def damped_item_mean(matrix: sparse.spmatrix, damping: float = 10.0) -> np.ndarray:
    """Shrink each item mean toward the global mean: ``(sum + m * mu) / (count + m)``."""
    if damping < 0:
        raise ValueError("damping must be non-negative")
    counts, sums = item_counts_and_sums(matrix)
    mu = global_mean(matrix)
    denominator = counts + damping
    means = np.full(counts.shape, mu, dtype="float64")
    np.divide(sums + damping * mu, denominator, out=means, where=denominator > 0)
    return means


def popularity_ranking(
    matrix: sparse.spmatrix,
    *,
    limit: int = 10,
    exclude: Iterable[int] = (),
    min_count: int = 0,
) -> np.ndarray:
    """Item indexes ordered by rating count, most rated first.

    Ties break on lower item index so results are deterministic.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    counts, _ = item_counts_and_sums(matrix)
    order = np.lexsort((np.arange(counts.size), -counts))
    blocked = set(int(index) for index in exclude)
    results = [
        int(index)
        for index in order
        if counts[index] >= min_count and int(index) not in blocked
    ]
    return np.asarray(results[:limit], dtype="int64")
