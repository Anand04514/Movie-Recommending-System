"""Biased matrix factorization from a truncated SVD, with fold-in for new users.

The model is deliberately the cheap, robust kind that fits this dataset on a laptop:

1. Estimate a baseline ``mu + b_u + b_i`` in two closed-form passes, so a film that everyone
   rates highly and a user who rates everything highly are both explained before any interaction
   is modelled. (Iterating the two bias equations alternates without damping and diverges on
   real data; a single item-then-user pass is stable and fits better, so that is what this uses.)
2. Take a truncated SVD of the *residual* matrix. Missing entries stay missing: a sparse matrix
   has no value at all where nothing was rated, and the truncated reconstruction scores them as
   "the baseline, with whatever latent taste the observed ratings imply". That is not the same as
   treating a missing rating as zero, which would drag every unseen film toward a bad prediction
   in proportion to how few people rated it.
3. Fold a new user in by ridge-regressing their residuals onto the fixed item factors. Their
   vector is then used for ranking immediately, with no retraining.

Fold-in is what makes SVD servable: app users never appear in the training matrix, so their
factors must be computed from the handful of ratings they have. The ``users`` x ``items`` dense
matrix is never built; predictions are always materialised one block of users at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from recsys.item_cf import Contribution, Recommendation

DEFAULT_RANK = 50
DEFAULT_BIAS_REGULARISATION = 10.0
DEFAULT_FACTOR_REGULARISATION = 0.05


@dataclass(frozen=True)
class SvdModel:
    """Fixed factors plus the biases needed to predict and to fold new users in."""

    global_mean: float
    user_bias: np.ndarray  # (n_users,) float64
    item_bias: np.ndarray  # (n_items,) float64
    user_factors: np.ndarray  # (n_users, rank) float32
    item_factors: np.ndarray  # (n_items, rank) float32
    rank: int
    bias_regularisation: float
    factor_regularisation: float
    seed: int

    @property
    def n_users(self) -> int:
        return int(self.user_factors.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_factors.shape[0])

    def predict_pair(self, user_index: int, item_index: int) -> float:
        """Predicted rating for a training user, clamped to the half-star range."""
        if not 0 <= user_index < self.n_users:
            raise IndexError(f"user index {user_index} is outside the model")
        if not 0 <= item_index < self.n_items:
            raise IndexError(f"item index {item_index} is outside the model")
        value = (
            self.global_mean
            + self.user_bias[user_index]
            + self.item_bias[item_index]
            + float(self.user_factors[user_index] @ self.item_factors[item_index])
        )
        return float(np.clip(value, 0.5, 5.0))

    def predict_for_users(self, user_indexes: np.ndarray) -> np.ndarray:
        """Predicted ratings for a block of training users, one column per item.

        The block is at most ``len(user_indexes) x n_items``, so it stays small regardless of how
        many users the model holds.
        """
        users = np.asarray(user_indexes, dtype="int64")
        if users.ndim != 1:
            raise ValueError("user_indexes must be one-dimensional")
        if users.size and (users.min() < 0 or users.max() >= self.n_users):
            raise IndexError("a user index is outside the model")
        scores = (
            self.user_factors[users] @ self.item_factors.T
            + self.item_bias[None, :]
            + self.user_bias[users][:, None]
            + self.global_mean
        )
        return np.asarray(scores, dtype="float32")


def train_svd(
    matrix: sparse.spmatrix,
    *,
    rank: int = DEFAULT_RANK,
    bias_regularisation: float = DEFAULT_BIAS_REGULARISATION,
    factor_regularisation: float = DEFAULT_FACTOR_REGULARISATION,
    seed: int = 0,
) -> SvdModel:
    """Fit ``mu + b_u + b_i + U V^T`` to the observed ratings of a sparse matrix."""
    if rank < 1:
        raise ValueError("rank must be positive")
    if bias_regularisation < 0:
        raise ValueError("bias_regularisation must be non-negative")
    if factor_regularisation <= 0:
        raise ValueError("factor_regularisation must be positive")

    matrix = sparse.csr_matrix(matrix, dtype="float64", copy=True)
    n_users, n_items = matrix.shape
    if matrix.nnz == 0:
        raise ValueError("cannot train on a matrix with no ratings")
    # svds needs k < min(shape); a tiny test matrix simply gets a smaller rank.
    max_rank = min(n_users, n_items) - 1
    if max_rank < 1:
        raise ValueError("need at least two users and two items to factorize")
    rank = min(rank, max_rank)

    counts_users = np.diff(matrix.indptr).astype("float64")
    counts_items = np.bincount(matrix.indices, minlength=n_items).astype("float64")
    sums_users = np.asarray(matrix.sum(axis=1)).ravel()
    sums_items = np.asarray(matrix.sum(axis=0)).ravel()
    global_mean = float(matrix.sum() / matrix.nnz)

    # Item bias first: a regularised damped item mean around the global mean. Then the user bias
    # absorbs what the item bias still leaves over. Order matters only in that the second pass
    # must see the first one's result; both are closed-form, so nothing can oscillate.
    residual_items = sums_items - counts_items * global_mean
    item_bias = np.divide(
        residual_items,
        bias_regularisation + counts_items,
        out=np.zeros_like(residual_items),
        where=counts_items > 0,
    )
    residual_users = sums_users - counts_users * global_mean - matrix @ item_bias
    user_bias = np.divide(
        residual_users,
        bias_regularisation + counts_users,
        out=np.zeros_like(residual_users),
        where=counts_users > 0,
    )

    residuals = matrix.copy()
    rows = np.repeat(np.arange(n_users, dtype="int64"), counts_users.astype("int64"))
    residuals.data = (
        matrix.data
        - global_mean
        - user_bias[rows]
        - item_bias[matrix.indices]
    )

    left, singular_values, right = svds(residuals, k=rank, random_state=seed)
    order = np.argsort(-singular_values, kind="stable")
    left = left[:, order]
    singular_values = singular_values[order]
    right = right[order]

    # Split the singular values symmetrically so a user factor and an item factor are on the same
    # scale; this only matters for the regularisation of the ridge fold-in.
    scale = np.sqrt(singular_values)
    user_factors = (left * scale[None, :]).astype("float32")
    item_factors = (right.T * scale[None, :]).astype("float32")

    return SvdModel(
        global_mean=global_mean,
        user_bias=user_bias.astype("float32").astype("float64"),
        item_bias=item_bias.astype("float32").astype("float64"),
        user_factors=user_factors,
        item_factors=item_factors,
        rank=int(rank),
        bias_regularisation=float(bias_regularisation),
        factor_regularisation=float(factor_regularisation),
        seed=int(seed),
    )


@dataclass(frozen=True)
class FoldedSvdUser:
    """A new user's estimated bias and latent vector, plus what they have already rated."""

    user_bias: float
    factors: np.ndarray  # (rank,) float64
    mean: float
    rated_indexes: np.ndarray
    rated_values: np.ndarray

    @property
    def rated_count(self) -> int:
        return int(self.rated_indexes.size)


def factor_gram(model: SvdModel) -> np.ndarray:
    """``V^T V + lambda I``, the matrix a fold-in solves against. Cache this per process."""
    gram = model.item_factors.T @ model.item_factors
    regularised = np.asarray(gram, dtype="float64")
    regularised[np.diag_indices_from(regularised)] += model.factor_regularisation
    return regularised


def fold_in(
    model: SvdModel,
    rated_indices: np.ndarray,
    rated_values: np.ndarray,
    *,
    gram: np.ndarray | None = None,
) -> FoldedSvdUser:
    """Estimate the factors of a user who is not in the training matrix.

    The user's bias is the regularised mean of what the item biases and the global mean cannot
    explain; their factors are the ridge-least-squares projection of the remaining residuals onto
    the fixed item factors. Ratings a user happens to share with the training set are not used.
    """
    rated_indices = np.asarray(rated_indices, dtype="int64")
    rated_values = np.asarray(rated_values, dtype="float64")
    if rated_indices.shape != rated_values.shape:
        raise ValueError("rated_indices and rated_values must have the same shape")
    if rated_indices.ndim != 1:
        raise ValueError("rated_indices must be one-dimensional")
    if rated_values.size == 0:
        return FoldedSvdUser(
            user_bias=0.0,
            factors=np.zeros(model.rank, dtype="float64"),
            mean=0.0,
            rated_indexes=np.empty(0, dtype="int64"),
            rated_values=np.empty(0, dtype="float64"),
        )
    if rated_indices.min() < 0 or rated_indices.max() >= model.n_items:
        raise IndexError("a rated item index is outside the model")
    if not np.all(np.isfinite(rated_values)):
        raise ValueError("rated values must be finite")

    mean = float(rated_values.mean())
    base = model.global_mean + model.item_bias[rated_indices]
    user_bias = float(
        np.clip(
            (rated_values - base).sum() / (model.bias_regularisation + rated_values.size),
            -2.0,
            2.0,
        )
    )
    residuals = rated_values - base - user_bias
    system = gram if gram is not None else factor_gram(model)
    factors = np.linalg.solve(system, model.item_factors[rated_indices].T @ residuals)
    return FoldedSvdUser(
        user_bias=user_bias,
        factors=np.asarray(factors, dtype="float64"),
        mean=mean,
        rated_indexes=rated_indices,
        rated_values=rated_values,
    )


def recommend_for_folded_user(
    model: SvdModel,
    folded: FoldedSvdUser,
    *,
    limit: int = 10,
    exclude: set[int] | None = None,
    explain: int = 3,
) -> list[Recommendation]:
    """Rank every item for a folded-in user and explain picks from their own ratings.

    A pick's evidence is the rated film whose *item factor* points most closely at it, weighted by
    how far above the user's average that rating was. Only positive evidence is cited.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if explain < 0:
        raise ValueError("explain must be non-negative")
    if folded.rated_indexes.size == 0:
        return []

    scores = (
        model.item_factors @ folded.factors
        + model.item_bias
        + model.global_mean
        + folded.user_bias
    ).astype("float64")
    predicted = np.clip(scores, 0.5, 5.0)

    blocked = {int(index) for index in folded.rated_indexes}
    if exclude:
        blocked |= {int(index) for index in exclude}
    if blocked:
        allowed = np.ones(model.n_items, dtype=bool)
        blocked_array = np.fromiter(blocked, dtype="int64", count=len(blocked))
        blocked_array = blocked_array[(blocked_array >= 0) & (blocked_array < model.n_items)]
        allowed[blocked_array] = False
        scores = np.where(allowed, scores, -np.inf)

    available = int(np.isfinite(scores).sum())
    if available == 0:
        return []
    take = min(limit, available)
    top = np.argpartition(-scores, take - 1)[:take]
    chosen = top[np.lexsort((top, -scores[top]))]

    contributions: list[tuple[Contribution, ...]] = [() for _ in chosen]
    if explain:
        norms = np.linalg.norm(model.item_factors, axis=1)
        positive = norms > 0
        # Only ratings above the user's own mean count as evidence: a film they disliked must
        # never be cited as the reason to watch something else.
        centred = folded.rated_values - folded.mean
        if positive[folded.rated_indexes].all():
            rated_vectors = model.item_factors[folded.rated_indexes] / norms[
                folded.rated_indexes, None
            ]
            strengths = (model.item_factors[chosen] / norms[chosen, None]) @ rated_vectors.T
            strengths *= np.clip(centred, 0.0, None)[None, :]
            for row, _item_index in enumerate(chosen):
                ranked = np.argsort(-strengths[row], kind="stable")[:explain]
                contributions[row] = tuple(
                    Contribution(
                        item_index=int(folded.rated_indexes[column]),
                        strength=float(strengths[row, column]),
                    )
                    for column in ranked
                    if strengths[row, column] > 0
                )

    results: list[Recommendation] = []
    for row, index in enumerate(chosen):
        sources = contributions[row]
        results.append(
            Recommendation(
                item_index=int(index),
                score=float(predicted[index]),
                predicted_rating=float(predicted[index]),
                source_index=sources[0].item_index if sources else None,
                contributions=sources,
            )
        )
    return results


def save_model(model: SvdModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_mean=np.float64(model.global_mean),
        user_bias=model.user_bias.astype("float64"),
        item_bias=model.item_bias.astype("float64"),
        user_factors=model.user_factors.astype("float32"),
        item_factors=model.item_factors.astype("float32"),
        rank=np.int64(model.rank),
        bias_regularisation=np.float64(model.bias_regularisation),
        factor_regularisation=np.float64(model.factor_regularisation),
        seed=np.int64(model.seed),
    )


def load_model(path: Path) -> SvdModel:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SVD factors not found at {path}; run scripts/train_and_precompute.py first"
        )
    with np.load(path) as data:
        return SvdModel(
            global_mean=float(data["global_mean"]),
            user_bias=data["user_bias"].astype("float64"),
            item_bias=data["item_bias"].astype("float64"),
            user_factors=data["user_factors"].astype("float32"),
            item_factors=data["item_factors"].astype("float32"),
            rank=int(data["rank"]),
            bias_regularisation=float(data["bias_regularisation"]),
            factor_regularisation=float(data["factor_regularisation"]),
            seed=int(data["seed"]),
        )
