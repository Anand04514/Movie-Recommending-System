"""Item-based collaborative filtering on sparse, mean-centered ratings.

Similarity is cosine over per-user mean-centered vectors. With sparse vectors the dot product
only sums co-rated items, which is exactly what the plan requires: missing ratings are never
treated as zeros.

Neighbors are built offline in bounded blocks and saved, so the API never computes similarities
on a request. ``recommend_from_ratings`` folds a user's (possibly brand-new) ratings into those
fixed neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from recsys.baselines import item_counts_and_sums


@dataclass(frozen=True)
class ItemNeighbors:
    """Top-K item neighbors plus the popularity statistics needed for cold-start fallbacks."""

    indices: np.ndarray  # (n_items, k) int32, -1 where fewer than k neighbors exist
    similarities: np.ndarray  # (n_items, k) float32
    counts: np.ndarray  # (n_items,) int64
    means: np.ndarray  # (n_items,) float64
    n_users: int
    min_similarity: float

    def similarity_matrix(self) -> sparse.csr_matrix:
        """The neighbor table as an items x items CSR matrix holding positive edges only.

        Ranking and rating evaluation multiply whole user blocks with it, which keeps the
        fold-in vectorized instead of walking each user's list of rated movies.
        """
        n_items, k = self.indices.shape
        rows = np.repeat(np.arange(n_items, dtype="int64"), k)
        cols = self.indices.reshape(-1).astype("int64")
        values = self.similarities.reshape(-1).astype("float64")
        keep = (cols >= 0) & (values > 0)
        return sparse.csr_matrix(
            (values[keep], (rows[keep], cols[keep])),
            shape=(n_items, n_items),
            dtype="float64",
        )


@dataclass(frozen=True)
class Contribution:
    """One of the user's ratings that pushed a recommendation up, with its weight."""

    item_index: int
    strength: float


@dataclass(frozen=True)
class Recommendation:
    """One ranked item with its predicted rating and the ratings that drove it."""

    item_index: int
    score: float
    predicted_rating: float
    source_index: int | None
    contributions: tuple[Contribution, ...] = ()


def mean_center(matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Subtract each user's mean from their observed ratings only."""
    matrix = matrix.tocsr()
    counts = np.diff(matrix.indptr)
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    means = np.zeros(matrix.shape[0], dtype="float64")
    np.divide(sums, counts, out=means, where=counts > 0)
    centered = matrix.astype("float32").copy()
    centered.data -= np.repeat(means.astype("float32"), counts)
    centered.eliminate_zeros()
    return centered, means


def build_item_neighbors(
    matrix: sparse.spmatrix,
    *,
    k: int = 50,
    block_size: int = 512,
    min_similarity: float = 1e-6,
) -> ItemNeighbors:
    """Compute cosine similarity block by block and keep only the top ``k`` per item.

    A dense item x item matrix is never materialized: each block is at most
    ``n_items x block_size`` and is discarded after its neighbors are selected.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if matrix.shape[1] < 2:
        raise ValueError("need at least two items to build neighbors")

    centered, user_means = mean_center(matrix)
    norms = np.sqrt(np.asarray(centered.multiply(centered).sum(axis=0)).ravel())
    inverse = np.zeros_like(norms)
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    normalized = (centered @ sparse.diags(inverse)).tocsr()

    n_items = matrix.shape[1]
    take = min(k, n_items - 1)
    indices = np.full((n_items, k), -1, dtype="int32")
    similarities = np.zeros((n_items, k), dtype="float32")

    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        block = (normalized.T @ normalized[:, start:stop]).toarray().astype("float32")
        width = stop - start
        rows = np.arange(start, stop)
        block[rows, np.arange(width)] = 0.0  # an item is never its own neighbor
        block[block < min_similarity] = 0.0

        top = np.argpartition(-block, kth=take - 1, axis=0)[:take]
        values = np.take_along_axis(block, top, axis=0)
        order = np.argsort(-values, axis=0, kind="stable")
        top = np.take_along_axis(top, order, axis=0)
        values = np.take_along_axis(values, order, axis=0)

        indices[start:stop, :take] = top.T
        similarities[start:stop, :take] = values.T

    # Neighbor slots with zero similarity (or padding) carry no information.
    empty = similarities <= 0
    similarities[empty] = 0.0
    indices[empty] = -1

    counts, sums = item_counts_and_sums(matrix)
    means = np.zeros_like(sums, dtype="float64")
    np.divide(sums, counts, out=means, where=counts > 0)

    return ItemNeighbors(
        indices=indices,
        similarities=similarities,
        counts=counts,
        means=means,
        n_users=int(matrix.shape[0]),
        min_similarity=float(min_similarity),
    )


def recommend_from_ratings(
    neighbors: ItemNeighbors,
    rated_indices: np.ndarray,
    rated_values: np.ndarray,
    *,
    limit: int = 10,
    exclude: set[int] | None = None,
    damping: float = 4.0,
    explain: int = 3,
) -> list[Recommendation]:
    """Score candidates from a user's ratings and the precomputed item neighbors.

    ``score`` is the sum of ``similarity * (rating - user_mean)`` over the user's rated items;
    ``predicted_rating`` shrinks that toward the user's mean by ``damping`` so a handful of
    neighbors cannot push every prediction to 5.0, then clamps to 0.5-5.0.

    Each recommendation carries its ``explain`` strongest positive contributions, strongest
    first, so the API can say which of the user's ratings drove a pick.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if explain < 0:
        raise ValueError("explain must be non-negative")
    rated_indices = np.asarray(rated_indices, dtype="int64")
    rated_values = np.asarray(rated_values, dtype="float64")
    if rated_indices.shape != rated_values.shape:
        raise ValueError("rated_indices and rated_values must have the same shape")

    blocked = set(int(index) for index in rated_indices)
    if exclude:
        blocked |= {int(index) for index in exclude}

    n_items = neighbors.indices.shape[0]
    if rated_indices.size == 0:
        return []

    user_mean = float(rated_values.mean())
    scores = np.zeros(n_items, dtype="float64")
    weights = np.zeros(n_items, dtype="float64")
    slots = max(explain, 1)
    top_strengths = np.full((n_items, slots), -np.inf, dtype="float64")
    top_sources = np.full((n_items, slots), -1, dtype="int32")

    for source, rating in zip(rated_indices, rated_values, strict=True):
        source = int(source)
        if not 0 <= source < n_items:
            raise IndexError(f"rated item index {source} is outside the neighbor table")
        targets = neighbors.indices[source]
        sims = neighbors.similarities[source]
        valid = (targets >= 0) & (sims > 0)
        if not valid.any():
            continue
        targets = targets[valid]
        sims = sims[valid].astype("float64")
        centered = float(rating) - user_mean

        np.add.at(scores, targets, sims * centered)
        np.add.at(weights, targets, sims)

        # Only positive contributions explain a recommendation; rating something 1-star is not
        # a sensible reason to recommend a similar movie. Insert them into per-candidate slots
        # (an insertion sort, vectorized one source at a time) and keep the strongest few.
        strength = sims * centered
        active = strength > 0
        if not active.any():
            continue
        candidates = targets[active]
        values = strength[active]
        pending = np.arange(candidates.size)
        for slot in range(slots):
            if pending.size == 0:
                break
            rows = candidates[pending]
            better = values[pending] > top_strengths[rows, slot]
            if not better.any():
                continue
            chosen = pending[better]
            rows = candidates[chosen]
            for shift in range(slots - 1, slot, -1):
                top_strengths[rows, shift] = top_strengths[rows, shift - 1]
                top_sources[rows, shift] = top_sources[rows, shift - 1]
            top_strengths[rows, slot] = values[chosen]
            top_sources[rows, slot] = source
            pending = pending[~better]

    candidates = (scores > 0) & (weights > 0)
    if blocked:
        blocked_array = np.fromiter(blocked, dtype="int64", count=len(blocked))
        blocked_array = blocked_array[(blocked_array >= 0) & (blocked_array < n_items)]
        candidates[blocked_array] = False

    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size == 0:
        return []

    predicted = np.clip(
        user_mean + scores[candidate_indices] / (weights[candidate_indices] + damping),
        0.5,
        5.0,
    )
    order = np.lexsort((candidate_indices, -scores[candidate_indices]))[:limit]
    chosen = candidate_indices[order]
    results: list[Recommendation] = []
    for i, index in enumerate(chosen):
        slots = list(zip(top_sources[index], top_strengths[index], strict=True))[:explain]
        contributions = tuple(
            Contribution(item_index=int(origin), strength=float(value))
            for origin, value in slots
            if origin >= 0 and value > 0
        )
        results.append(
            Recommendation(
                item_index=int(index),
                score=float(scores[index]),
                predicted_rating=float(predicted[order[i]]),
                source_index=contributions[0].item_index if contributions else None,
                contributions=contributions,
            )
        )
    return results


def popularity_recommendations(
    neighbors: ItemNeighbors,
    *,
    limit: int = 10,
    exclude: set[int] | None = None,
) -> list[Recommendation]:
    """Cold-start fallback: most-rated items, with their mean rating as the estimate."""
    blocked = {int(index) for index in (exclude or set())}
    order = np.lexsort((np.arange(neighbors.counts.size), -neighbors.counts))
    results: list[Recommendation] = []
    for index in order:
        if int(index) in blocked or neighbors.counts[index] == 0:
            continue
        results.append(
            Recommendation(
                item_index=int(index),
                score=float(neighbors.counts[index]),
                predicted_rating=float(np.clip(neighbors.means[index], 0.5, 5.0)),
                source_index=None,
            )
        )
        if len(results) >= limit:
            break
    return results


def save_neighbors(neighbors: ItemNeighbors, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=neighbors.indices,
        similarities=neighbors.similarities,
        counts=neighbors.counts,
        means=neighbors.means,
        n_users=np.int64(neighbors.n_users),
        min_similarity=np.float64(neighbors.min_similarity),
    )


def load_neighbors(path: Path) -> ItemNeighbors:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"item neighbors not found at {path}; run scripts/train_and_precompute.py first"
        )
    with np.load(path) as data:
        return ItemNeighbors(
            indices=data["indices"].astype("int32"),
            similarities=data["similarities"].astype("float32"),
            counts=data["counts"].astype("int64"),
            means=data["means"].astype("float64"),
            n_users=int(data["n_users"]),
            min_similarity=float(data["min_similarity"]),
        )
