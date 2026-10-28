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


@dataclass(frozen=True)
class FoldInModel:
    """Training data prepared for scoring a user who is not in the matrix.

    The offline neighbor table cannot help the API: app users have no row in the training matrix.
    Folding in computes their neighbours on the fly from the items they actually rated, which is
    what makes user-based CF servable at all.
    """

    centered: sparse.csr_matrix  # users x items, mean-centred per user
    mask: sparse.csr_matrix  # users x items, 1.0 where rated
    norms: np.ndarray  # ||centered row||, for the cosine denominator
    by_item: sparse.csc_matrix  # users x items in CSC, for item -> raters
    n_users: int
    n_items: int


def build_fold_in_model(matrix: sparse.spmatrix) -> FoldInModel:
    """Prepare the centered matrix, rating mask and item-major view a fold-in needs."""
    matrix = matrix.tocsr()
    centered, _ = mean_center(matrix)
    norms = np.sqrt(np.asarray(centered.multiply(centered).sum(axis=1)).ravel())
    mask = matrix.copy()
    mask.data = np.ones_like(mask.data)
    return FoldInModel(
        centered=centered,
        mask=mask.astype("float64"),
        norms=norms,
        by_item=centered.tocsc(),
        n_users=int(matrix.shape[0]),
        n_items=int(matrix.shape[1]),
    )


@dataclass(frozen=True)
class FoldedUser:
    """The neighbour set computed for a user outside the training matrix."""

    neighbor_indexes: np.ndarray
    similarities: np.ndarray
    mean: float
    variance: float  # squared norm of the user's centred ratings
    rated_indexes: np.ndarray  # what the user has already rated, never to be recommended back

    @property
    def has_neighbors(self) -> bool:
        return bool(self.neighbor_indexes.size)

    @property
    def rated_count(self) -> int:
        return int(self.rated_indexes.size)


def fold_in(
    model: FoldInModel,
    rated_indices: np.ndarray,
    rated_values: np.ndarray,
    *,
    k: int = 50,
    min_similarity: float = 1e-6,
) -> FoldedUser:
    """Nearest training users for a new user, cosine over co-rated items only.

    A user whose ratings are all equal has no deviation to correlate, so they come back with no
    neighbours rather than with arbitrary ones; the caller falls back to the popularity baseline.
    """
    if k < 1:
        raise ValueError("k must be positive")
    rated_indices = np.asarray(rated_indices, dtype="int64")
    rated_values = np.asarray(rated_values, dtype="float64")
    if rated_indices.shape != rated_values.shape:
        raise ValueError("rated_indices and rated_values must have the same shape")

    empty = FoldedUser(
        neighbor_indexes=np.empty(0, dtype="int64"),
        similarities=np.empty(0, dtype="float64"),
        mean=float(rated_values.mean()) if rated_values.size else 0.0,
        variance=0.0,
        rated_indexes=rated_indices.copy(),
    )
    if rated_indices.size == 0:
        return empty
    if np.any(rated_indices < 0) or np.any(rated_indices >= model.n_items):
        raise IndexError("rated item index is outside the fold-in model")

    mean = float(rated_values.mean())
    deviations = rated_values - mean
    variance = float(np.sum(deviations**2))
    if variance <= 0:
        return empty

    numerator = np.zeros(model.n_users, dtype="float64")
    for position, item in enumerate(rated_indices):
        start, stop = model.by_item.indptr[item], model.by_item.indptr[item + 1]
        if start == stop:
            continue
        raters = model.by_item.indices[start:stop]
        np.add.at(numerator, raters, deviations[position] * model.by_item.data[start:stop])

    denominator = float(np.sqrt(variance)) * model.norms
    similarities = np.zeros(model.n_users, dtype="float64")
    np.divide(numerator, denominator, out=similarities, where=denominator > 0)
    similarities[similarities < min_similarity] = 0.0

    take = min(k, model.n_users)
    partial = np.argpartition(-similarities, take - 1)[:take]
    values = similarities[partial]
    order = np.lexsort((partial, -values))
    partial, values = partial[order], values[order]
    keep = values > 0
    return FoldedUser(
        neighbor_indexes=partial[keep].astype("int64"),
        similarities=values[keep],
        mean=mean,
        variance=variance,
        rated_indexes=rated_indices.copy(),
    )


@dataclass(frozen=True)
class FoldInScores:
    """Per-item fold-in output for a live user."""

    scores: np.ndarray  # weighted deviations, the ranking signal
    weights: np.ndarray  # total neighbour similarity that touched the item
    predicted: np.ndarray  # 0.5-5.0 estimate, clipped
    rater_counts: np.ndarray  # how many neighbours rated the item


def score_items(
    model: FoldInModel, folded: FoldedUser, *, damping: float = 4.0
) -> FoldInScores:
    """Score every item from the folded user's neighbours.

    ``score = Σ_v sim(v) x centered[v, j]`` and ``weight = Σ_v sim(v)`` over the neighbours that
    rated item ``j``; the prediction shrinks the ratio toward the user's own mean by ``damping``.
    """
    if damping < 0:
        raise ValueError("damping must be non-negative")
    n_items = model.n_items
    if not folded.has_neighbors:
        return FoldInScores(
            scores=np.zeros(n_items, dtype="float64"),
            weights=np.zeros(n_items, dtype="float64"),
            predicted=np.full(n_items, min(max(folded.mean, 0.5), 5.0), dtype="float64"),
            rater_counts=np.zeros(n_items, dtype="int64"),
        )

    neighbors = folded.neighbor_indexes
    scaling = sparse.diags(folded.similarities)
    scores = np.asarray((scaling @ model.centered[neighbors]).sum(axis=0)).ravel()
    weights = np.asarray((scaling @ model.mask[neighbors]).sum(axis=0)).ravel()
    rater_counts = np.asarray(
        (model.mask[neighbors] > 0).sum(axis=0), dtype="int64"
    ).ravel()
    predicted = np.zeros(n_items, dtype="float64")
    np.divide(scores, weights + damping, out=predicted, where=weights > 0)
    predicted = np.clip(folded.mean + predicted, 0.5, 5.0)
    return FoldInScores(
        scores=scores, weights=weights, predicted=predicted, rater_counts=rater_counts
    )


@dataclass(frozen=True)
class FoldInRecommendation:
    """One ranked item for a folded-in user, with the evidence user-based CF can honestly give."""

    item_index: int
    score: float
    predicted_rating: float
    neighbors_rated: int


def recommend_for_folded_user(
    model: FoldInModel,
    folded: FoldedUser,
    *,
    limit: int = 10,
    exclude: set[int] | None = None,
    damping: float = 4.0,
) -> list[FoldInRecommendation]:
    """Top items for a folded-in user.

    Candidates need positive support from the neighbours, exactly like the item-based path: an item
    only the user's *dissimilar* neighbours touched is never recommended. Everything the user has
    already rated is excluded here rather than left to the caller.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    scores = score_items(model, folded, damping=damping)
    candidates = (scores.weights > 0) & (scores.scores > 0)
    blocked = {int(index) for index in (exclude or set())}
    blocked |= {int(index) for index in folded.rated_indexes}
    if blocked:
        blocked_array = np.fromiter(blocked, dtype="int64", count=len(blocked))
        blocked_array = blocked_array[(blocked_array >= 0) & (blocked_array < model.n_items)]
        candidates[blocked_array] = False

    indexes = np.flatnonzero(candidates)
    if indexes.size == 0:
        return []
    order = np.lexsort((indexes, -scores.scores[indexes]))[:limit]
    chosen = indexes[order]
    return [
        FoldInRecommendation(
            item_index=int(index),
            score=float(scores.scores[index]),
            predicted_rating=float(scores.predicted[index]),
            neighbors_rated=int(scores.rater_counts[index]),
        )
        for index in chosen
    ]


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
