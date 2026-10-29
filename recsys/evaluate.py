"""Walk-forward evaluation on the chronological per-user split.

``data/processed`` already holds the split: the training matrix has everything except each user's
most recent 20% of ratings, and the test Parquet holds the rest. Every method is scored on exactly
the same held-out pairs, so RMSE and MAE are directly comparable:

- ``global_mean``: the training mean for every pair. No ranking.
- ``damped_item_mean``: each item's mean shrunk toward the global mean by ``damping``.
- ``popularity``: rank by rating count, predict the item's mean rating.
- ``item_cf``: fold each user's training ratings into the precomputed top-K item neighbors.
- ``user_cf``: the same fold-in along the user axis over precomputed top-K user neighbors.

Ranking uses the top-K list per user with the user's training items removed, counting a held-out
rating of 4.0 or higher as relevant. A collaborative method with no support for a pair (no
neighbor co-rated that item) falls back to the damped item mean; the share of such pairs is
reported as ``fallback_rate`` rather than hidden, and it is why RMSE stays finite for every pair.

Nothing here is imported by the API: this is the offline measurement path.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.baselines import damped_item_mean, global_mean, item_counts_and_sums, item_means
from recsys.content import DEFAULT_PREDICTION_SLOPE, ContentFeatures
from recsys.data import build_csr, chronological_split
from recsys.hybrid import blend_matrix, combine_predictions
from recsys.item_cf import ItemNeighbors, mean_center
from recsys.metrics import RankingMetrics, average_ranking_metrics, mae, rmse
from recsys.svd import SvdModel
from recsys.user_cf import UserNeighbors

DEFAULT_RELEVANCE_THRESHOLD = 4.0


@dataclass(frozen=True)
class SplitData:
    """The held-out pairs of the evaluated users, plus what the models need from training."""

    train: sparse.csr_matrix
    user_means: np.ndarray
    test_users: np.ndarray
    test_items: np.ndarray
    test_ratings: np.ndarray
    evaluated_users: np.ndarray
    relevant_by_user: dict[int, set[int]]
    pairs_by_user: dict[int, np.ndarray]
    train_items_by_user: dict[int, np.ndarray]

    @property
    def n_users(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.train.shape[1])

    @property
    def n_pairs(self) -> int:
        return int(self.test_users.size)


def split_from_arrays(
    train: sparse.csr_matrix,
    test_users: np.ndarray,
    test_items: np.ndarray,
    test_ratings: np.ndarray,
    *,
    evaluated_users: np.ndarray | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> SplitData:
    """Assemble a :class:`SplitData` from a training matrix and held-out rating arrays."""
    test_users = np.asarray(test_users, dtype="int64")
    test_items = np.asarray(test_items, dtype="int64")
    test_ratings = np.asarray(test_ratings, dtype="float64")
    if not (test_users.shape == test_items.shape == test_ratings.shape):
        raise ValueError("test users, items and ratings must have the same shape")

    if evaluated_users is None:
        evaluated = np.unique(test_users)
    else:
        evaluated = np.sort(np.asarray(evaluated_users, dtype="int64"))
        keep = np.isin(test_users, evaluated)
        test_users = test_users[keep]
        test_items = test_items[keep]
        test_ratings = test_ratings[keep]

    order = np.argsort(test_users, kind="stable")
    test_users = test_users[order]
    test_items = test_items[order]
    test_ratings = test_ratings[order]

    boundaries = np.searchsorted(test_users, evaluated)
    ends = np.append(boundaries[1:], test_users.size)
    pairs_by_user = {
        int(user): np.arange(start, end)
        for user, start, end in zip(evaluated, boundaries, ends, strict=True)
    }
    relevant_by_user = {
        int(user): {
            int(item)
            for item, rating in zip(
                test_items[start:end], test_ratings[start:end], strict=True
            )
            if rating >= relevance_threshold
        }
        for user, start, end in zip(evaluated, boundaries, ends, strict=True)
    }

    train = train.tocsr()
    train_counts = np.diff(train.indptr)
    user_means = np.zeros(train.shape[0], dtype="float64")
    np.divide(
        np.asarray(train.sum(axis=1)).ravel(),
        train_counts,
        out=user_means,
        where=train_counts > 0,
    )
    train_items_by_user = {
        int(user): train.indices[train.indptr[user] : train.indptr[user + 1]]
        for user in evaluated
    }

    return SplitData(
        train=train,
        user_means=user_means,
        test_users=test_users,
        test_items=test_items,
        test_ratings=test_ratings,
        evaluated_users=evaluated,
        relevant_by_user=relevant_by_user,
        pairs_by_user=pairs_by_user,
        train_items_by_user=train_items_by_user,
    )


def load_split(
    processed_dir: str | Path,
    *,
    users: int = 10_000,
    seed: int = 42,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> SplitData:
    """Load the training matrix and sample held-out users from the chronological test split."""
    processed = Path(processed_dir)
    train_path = processed / "train_matrix.npz"
    test_path = processed / "ratings_test.parquet"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"missing {train_path.name} or {test_path.name}; run scripts/prepare_data.py first"
        )
    if users < 1:
        raise ValueError("users must be positive")

    train = sparse.load_npz(train_path).tocsr()
    test = pd.read_parquet(test_path, columns=["user_index", "movie_index", "rating"])
    test_users = test["user_index"].to_numpy(dtype="int64")
    test_items = test["movie_index"].to_numpy(dtype="int64")
    test_ratings = test["rating"].to_numpy(dtype="float64")

    candidates = np.unique(test_users)
    if users >= candidates.size:
        evaluated = candidates
    else:
        rng = np.random.default_rng(seed)
        evaluated = np.sort(rng.choice(candidates, size=users, replace=False))

    return split_from_arrays(
        train,
        test_users,
        test_items,
        test_ratings,
        evaluated_users=evaluated,
        relevance_threshold=relevance_threshold,
    )


class Scorer(Protocol):
    """A method that can predict held-out pairs and rank candidates for a block of users."""

    name: str
    params: dict[str, object]
    supports_ranking: bool
    n_items: int

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``(prediction_scores, support_weights, ranking_scores)`` for these users."""


class GlobalMeanScorer:
    """The trivial predictor: every pair gets the training mean."""

    name = "global_mean"
    supports_ranking = False
    params: dict[str, object]

    def __init__(self, train: sparse.spmatrix) -> None:
        self.value = global_mean(train)
        self.n_items = int(train.shape[1])
        self.params = {"value": round(self.value, 4)}

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        scores = np.full((len(users), self.n_items), self.value, dtype="float32")
        return scores, None, None


class ItemVectorScorer:
    """Item-based baselines: one score per item, shared by every user."""

    supports_ranking = True

    def __init__(
        self,
        name: str,
        *,
        prediction_vector: np.ndarray,
        ranking_vector: np.ndarray,
        counts: np.ndarray,
        params: dict[str, object],
    ) -> None:
        self.name = name
        self.params = params
        self.prediction_vector = np.asarray(prediction_vector, dtype="float32")
        self.n_items = int(self.prediction_vector.size)
        ranking = np.asarray(ranking_vector, dtype="float64")
        # An item nobody rated in training cannot be ranked meaningfully.
        ranking = np.where(counts > 0, ranking, -np.inf)
        self.ranking_vector = ranking.astype("float32")

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        shape = (len(users), self.n_items)
        return (
            np.broadcast_to(self.prediction_vector, shape),
            None,
            np.broadcast_to(self.ranking_vector, shape),
        )


class FoldInScorer:
    """Item- or user-based CF folded into a fixed neighbor table.

    ``prediction = clip(user_mean + scores / (support + damping))``, where ``scores`` sums
    ``similarity x (rating - user_mean)`` over the user's training items and ``support`` sums the
    absolute similarities. That is the same formula the API serves, with one extra knob: pairs the
    neighbor table cannot support fall back to the damped item mean so RMSE stays comparable.
    """

    supports_ranking = True

    def __init__(
        self,
        name: str,
        *,
        kind: str,
        centered: sparse.csr_matrix,
        mask: sparse.csr_matrix,
        structure: sparse.csr_matrix,
        damping: float,
        params: dict[str, object],
        user_means: np.ndarray,
        fallback_predictions: np.ndarray,
    ) -> None:
        if kind not in {"item", "user"}:
            raise ValueError("kind must be 'item' or 'user'")
        if damping < 0:
            raise ValueError("damping must be non-negative")
        self.name = name
        self.kind = kind
        self.centered = centered
        self.mask = mask
        self.structure = structure
        self.damping = float(damping)
        self.params = params
        self.n_items = int(centered.shape[1])
        self.user_means = np.asarray(user_means, dtype="float64")
        self.fallback_predictions = np.asarray(fallback_predictions, dtype="float64")

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.kind == "item":
            scores = self.centered[users] @ self.structure
            weights = self.mask[users] @ self.structure
        else:
            left = self.structure[users]
            scores = left @ self.centered
            weights = left @ self.mask
        dense_scores = np.asarray(scores.toarray(), dtype="float32")
        dense_weights = np.asarray(weights.toarray(), dtype="float32")
        return dense_scores, dense_weights, dense_scores

    def predicted_for(self, users: np.ndarray) -> np.ndarray:
        """Calibrated predicted ratings, using the same shrinkage rule as the API.

        The raw block scores are evidence sums, not ratings; blending them directly into a
        hybrid prediction inflates it by an order of magnitude.
        """
        users = np.asarray(users, dtype="int64")
        scores, weights, _ = self.blocks(users)
        assert weights is not None  # fold-in always reports support
        means = self.user_means[users][:, None]
        filled = means + np.divide(
            scores,
            weights + self.damping,
            out=np.zeros_like(scores, dtype="float64"),
            where=weights > 0,
        )
        fallback = np.broadcast_to(self.fallback_predictions[None, :], scores.shape)
        return np.clip(np.where(weights > 0, filled, fallback), 0.5, 5.0)


class SvdScorer:
    """Biased matrix factorization for users who are already in the training matrix.

    The API folds new users in; evaluation users *are* training users, so their factor rows are
    read straight out of the model. Predictions are clamped to the rating range, while ranking
    keeps the unclamped score so ties at five stars do not destroy the ordering.
    """

    supports_ranking = True
    name = "svd"

    def __init__(self, model: SvdModel) -> None:
        self.model = model
        self.n_items = model.n_items
        self.params: dict[str, object] = {
            "rank": model.rank,
            "bias_regularisation": model.bias_regularisation,
            "factor_regularisation": model.factor_regularisation,
        }

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        ranking = self.model.predict_for_users(np.asarray(users, dtype="int64"))
        return np.clip(ranking, 0.5, 5.0), None, ranking


class ContentScorer:
    """Genre, decade and tag profiles scored against the model-universe feature rows."""

    supports_ranking = True
    name = "content"

    def __init__(
        self,
        train: sparse.spmatrix,
        item_features: sparse.spmatrix,
        *,
        slope: float = DEFAULT_PREDICTION_SLOPE,
        features: ContentFeatures | None = None,
    ) -> None:
        if slope <= 0:
            raise ValueError("slope must be positive")
        self.train = train.tocsr()
        self.item_features = sparse.csr_matrix(item_features, dtype="float64")
        if self.item_features.shape[0] != self.train.shape[1]:
            raise ValueError("item_features must have one row per model item")
        counts = np.diff(self.train.indptr)
        sums = np.asarray(self.train.sum(axis=1)).ravel()
        self.user_means = np.zeros(self.train.shape[0], dtype="float64")
        np.divide(sums, counts, out=self.user_means, where=counts > 0)
        self.slope = float(slope)
        self.n_items = int(self.train.shape[1])
        self.params: dict[str, object] = {
            "features": int(self.item_features.shape[1]),
            "prediction_slope": self.slope,
            "catalogue_movies": features.n_movies if features is not None else None,
        }
        if features is not None:
            self.params["feature_kinds"] = {
                "genres": sum(1 for name in features.feature_names if name.startswith("genre:")),
                "decades": sum(1 for name in features.feature_names if name.startswith("decade:")),
                "tags": sum(1 for name in features.feature_names if name.startswith("tag:")),
            }

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        block = self.train[users]
        row_means = np.repeat(self.user_means[users], np.diff(block.indptr))
        centred = np.clip(block.data - row_means, 0.0, None)

        # A user whose ratings are all equal has no within-user deviation; fall back to
        # "above neutral" so their genres still form a profile (the same rule the API uses).
        weights = block.copy()
        weights.data = centred
        totals = np.asarray(weights.sum(axis=1)).ravel()
        uniform = totals <= 0
        if uniform.any():
            fallback = np.clip(block.data - 3.0, 0.0, None)
            row_uniform = np.repeat(uniform, np.diff(block.indptr))
            weights.data = np.where(row_uniform, fallback, centred)

        profiles = sparse.csr_matrix(weights @ self.item_features, dtype="float64")
        norms = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
        inverse = np.zeros_like(norms)
        np.divide(1.0, norms, out=inverse, where=norms > 0)
        profiles = (sparse.diags(inverse) @ profiles).tocsr()

        scores = np.asarray((profiles @ self.item_features.T).toarray(), dtype="float32")
        predictions = np.clip(
            self.user_means[np.asarray(users, dtype="int64")][:, None] + self.slope * scores,
            0.5,
            5.0,
        )
        # A candidate with zero overlap is not evidence for this user; like the serving path,
        # only positively-scored items enter the ranking.
        ranking = np.where(scores > 0, scores, -np.inf).astype("float32")
        return predictions, None, ranking


class HybridScorer:
    """A tuned weighted blend of a collaborative scorer and the content scorer."""

    supports_ranking = True
    name = "hybrid"

    def __init__(
        self,
        collaborative: Scorer,
        content: Scorer,
        *,
        weight: float,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        self.collaborative = collaborative
        self.content = content
        self.weight = float(weight)
        self.n_items = collaborative.n_items
        self.params: dict[str, object] = {
            "collaborative": collaborative.name,
            "content": content.name,
            "collaborative_weight": self.weight,
        }

    def blocks(
        self, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        collaborative_predictions = self._predicted(self.collaborative, users)
        content_predictions = self._predicted(self.content, users)
        _, _, collaborative_ranking = self.collaborative.blocks(users)
        _, _, content_ranking = self.content.blocks(users)

        ranking = blend_matrix(
            collaborative_ranking,
            content_ranking,
            weight=self.weight,
        )
        # A blend of two zeros is not support; the API has the same rule.
        ranking = np.where(ranking > 0, ranking, -np.inf).astype("float32")
        predictions = combine_predictions(
            collaborative_predictions,
            content_predictions,
            weight=self.weight,
            available_collaborative=np.isfinite(collaborative_ranking),
            available_content=np.isfinite(content_ranking),
        )
        return predictions, None, ranking

    @staticmethod
    def _predicted(scorer: Scorer, users: np.ndarray) -> np.ndarray:
        method = getattr(scorer, "predicted_for", None)
        if callable(method):
            return np.asarray(method(users), dtype="float32")
        return np.asarray(scorer.blocks(users)[0], dtype="float32")


@dataclass(frozen=True)
class MethodOutcome:
    """Everything the report needs about one method."""

    method: str
    params: dict[str, object]
    rmse: float
    mae: float
    rated_pairs: int
    fallback_rate: float
    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None
    catalogue_coverage: float | None
    evaluated_users: int
    skipped_users: int
    k: int
    runtime_seconds: float
    peak_memory_mb: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _top_k(row: np.ndarray, k: int) -> list[int]:
    """Indexes of the ``k`` highest values, ties broken by lower index.

    ``-inf`` marks excluded items and is never returned, even if the list comes up short.
    """
    available = int(np.isfinite(row).sum())
    if available == 0:
        return []
    take = min(k, available)
    partial = np.argpartition(-row, take - 1)[:take]
    order = partial[np.lexsort((partial, -row[partial]))]
    return [int(index) for index in order if np.isfinite(row[index])]


def _score_pairs(
    split: SplitData,
    chunk: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
    damping: float,
    fallback_predictions: np.ndarray,
    predicted: np.ndarray,
    fell_back: np.ndarray,
) -> None:
    index = np.concatenate([split.pairs_by_user[int(user)] for user in chunk])
    if index.size == 0:
        return
    pair_users = split.test_users[index]
    items = split.test_items[index]
    # ``chunk`` is ascending, so searchsorted maps pair users back to their block row.
    rows = np.searchsorted(chunk, pair_users)
    if weights is None:
        predicted[index] = scores[rows, items]
        return
    support = weights[rows, items]
    means = split.user_means[pair_users]
    filled = np.clip(means + scores[rows, items] / (support + damping), 0.5, 5.0)
    unsupported = support <= 0
    predicted[index] = np.where(unsupported, fallback_predictions[items], filled)
    fell_back[index] = unsupported


def _top_k_lists(
    split: SplitData,
    chunk: np.ndarray,
    ranking: np.ndarray,
    weights: np.ndarray | None,
    k: int,
) -> list[list[int]]:
    lists: list[list[int]] = []
    for row_index, user in enumerate(chunk):
        row = np.array(ranking[row_index], dtype="float64", copy=True)
        if weights is not None:
            # Same rule as the serving path: a candidate needs positive evidence from the
            # user's own profile, not merely a nonzero similarity.
            supported = (weights[row_index] > 0) & (row > 0)
            row = np.where(supported, row, -np.inf)
        excluded = split.train_items_by_user[int(user)]
        if excluded.size:
            row[excluded] = -np.inf
        lists.append(_top_k(row, k))
    return lists


def evaluate_method(
    scorer: Scorer,
    split: SplitData,
    *,
    fallback_predictions: np.ndarray,
    k: int = 10,
    chunk_size: int = 512,
    track_memory: bool = True,
) -> MethodOutcome:
    """Score one method on the split: rating error on every pair, ranking for the sampled users."""
    if k < 1:
        raise ValueError("k must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    damping = float(getattr(scorer, "damping", 0.0))
    started = time.perf_counter()
    if track_memory:
        tracemalloc.start()
        memory_before = tracemalloc.get_traced_memory()[0]
    else:
        memory_before = 0

    predicted = np.empty(split.n_pairs, dtype="float64")
    fell_back = np.zeros(split.n_pairs, dtype=bool)
    recommendations: list[list[int]] = []

    for start in range(0, split.evaluated_users.size, chunk_size):
        chunk = split.evaluated_users[start : start + chunk_size]
        scores, weights, ranking = scorer.blocks(chunk)
        _score_pairs(
            split, chunk, scores, weights, damping, fallback_predictions, predicted, fell_back
        )
        if scorer.supports_ranking and ranking is not None:
            recommendations.extend(_top_k_lists(split, chunk, ranking, weights, k))

    ranking_metrics: RankingMetrics | None = None
    if scorer.supports_ranking:
        relevant_sets = [
            split.relevant_by_user[int(user)] for user in split.evaluated_users
        ]
        ranking_metrics = average_ranking_metrics(
            recommendations, relevant_sets, k=k, n_items=split.n_items
        )

    peak_memory_mb = 0.0
    if track_memory:
        peak_memory_mb = max(tracemalloc.get_traced_memory()[1] - memory_before, 0) / 2**20
        tracemalloc.stop()

    return MethodOutcome(
        method=scorer.name,
        params=dict(scorer.params),
        rmse=rmse(split.test_ratings, predicted),
        mae=mae(split.test_ratings, predicted),
        rated_pairs=split.n_pairs,
        fallback_rate=float(fell_back.mean()) if fell_back.size else 0.0,
        precision_at_k=ranking_metrics.precision_at_k if ranking_metrics else None,
        recall_at_k=ranking_metrics.recall_at_k if ranking_metrics else None,
        ndcg_at_k=ranking_metrics.ndcg_at_k if ranking_metrics else None,
        catalogue_coverage=ranking_metrics.catalogue_coverage if ranking_metrics else None,
        evaluated_users=ranking_metrics.evaluated_users if ranking_metrics else 0,
        skipped_users=ranking_metrics.skipped_users if ranking_metrics else 0,
        k=k,
        runtime_seconds=time.perf_counter() - started,
        peak_memory_mb=peak_memory_mb,
    )


def evaluate_split(
    split: SplitData,
    scorers: Sequence[Scorer],
    *,
    k: int = 10,
    chunk_size: int = 512,
    track_memory: bool = True,
) -> list[MethodOutcome]:
    """Run every method on the same split and return their outcomes in order."""
    fallback_predictions = damped_item_mean(split.train)
    return [
        evaluate_method(
            scorer,
            split,
            fallback_predictions=fallback_predictions,
            k=k,
            chunk_size=chunk_size,
            track_memory=track_memory,
        )
        for scorer in scorers
    ]


def default_scorers(
    train: sparse.csr_matrix,
    *,
    item_neighbors: ItemNeighbors | None = None,
    user_neighbors: UserNeighbors | None = None,
    item_damping: float = 4.0,
    user_damping: float = 4.0,
    item_mean_damping: float = 10.0,
    svd_model: SvdModel | None = None,
    content_features: ContentFeatures | None = None,
    model_movie_ids: np.ndarray | None = None,
    hybrid_weight: float = 0.65,
    hybrid_base: str = "item_cf",
) -> list[Scorer]:
    """Every method the plan asks about, in report order.

    SVD and content are only added when their artifacts are supplied; the hybrid needs item-CF
    and raw content features together. ``model_movie_ids`` aligns the catalog feature rows to the
    model matrix's item order and is required with ``content_features``.
    """
    counts, _ = item_counts_and_sums(train)
    mu = global_mean(train)
    damped = damped_item_mean(train, item_mean_damping)
    # Items nobody rated in training still need a defensible prediction for RMSE.
    popularity_prediction = np.where(counts > 0, item_means(train), mu)
    scorers: list[Scorer] = [
        GlobalMeanScorer(train),
        ItemVectorScorer(
            "damped_item_mean",
            prediction_vector=damped,
            ranking_vector=damped,
            counts=counts,
            params={"damping": item_mean_damping},
        ),
        ItemVectorScorer(
            "popularity",
            prediction_vector=popularity_prediction,
            ranking_vector=counts.astype("float64"),
            counts=counts,
            params={"prediction": "item_mean", "ranking": "rating_count"},
        ),
    ]

    if hybrid_base not in {"item_cf", "user_cf"}:
        raise ValueError("hybrid_base must be 'item_cf' or 'user_cf'")

    train_counts = np.diff(train.indptr)
    train_sums = np.asarray(train.sum(axis=1)).ravel()
    user_means = np.zeros(train.shape[0], dtype="float64")
    np.divide(train_sums, train_counts, out=user_means, where=train_counts > 0)
    fallback = damped_item_mean(train, item_mean_damping)

    item_scorer: FoldInScorer | None = None
    user_scorer: FoldInScorer | None = None
    if item_neighbors is not None or user_neighbors is not None:
        centered, _ = mean_center(train)
        mask = train.copy()
        mask.data = np.ones_like(mask.data)
        mask = mask.astype("float64")
        if item_neighbors is not None:
            item_scorer = FoldInScorer(
                "item_cf",
                kind="item",
                centered=centered,
                mask=mask,
                structure=item_neighbors.similarity_matrix(),
                damping=item_damping,
                params={
                    "k": int(item_neighbors.indices.shape[1]),
                    "damping": item_damping,
                },
                user_means=user_means,
                fallback_predictions=fallback,
            )
            scorers.append(item_scorer)
        if user_neighbors is not None:
            user_scorer = FoldInScorer(
                "user_cf",
                kind="user",
                centered=centered,
                mask=mask,
                structure=user_neighbors.similarity_matrix(),
                damping=user_damping,
                params={
                    "k": int(user_neighbors.indices.shape[1]),
                    "damping": user_damping,
                },
                user_means=user_means,
                fallback_predictions=fallback,
            )
            scorers.append(user_scorer)

    if svd_model is not None:
        if svd_model.n_users != train.shape[0]:
            raise ValueError("the SVD model was trained on a different number of users")
        scorers.append(SvdScorer(svd_model))

    content_scorer: ContentScorer | None = None
    if content_features is not None:
        if model_movie_ids is None:
            raise ValueError("model_movie_ids is required with content_features")
        item_features = content_features.rows_for(np.asarray(model_movie_ids, dtype="int64"))
        content_scorer = ContentScorer(train, item_features, features=content_features)
        scorers.append(content_scorer)

    collaborative_base = item_scorer if hybrid_base == "item_cf" else user_scorer
    if collaborative_base is not None and content_scorer is not None:
        scorers.append(HybridScorer(collaborative_base, content_scorer, weight=hybrid_weight))
    return scorers


def tune_hybrid_weight(
    split: SplitData,
    *,
    collaborative: Scorer,
    content: Scorer,
    weights: Sequence[float],
    k: int = 10,
    chunk_size: int = 512,
) -> tuple[float, list[MethodOutcome]]:
    """Score the blend at each candidate weight on a validation split and pick the best NDCG.

    Call this with a split carved out of the *training* data, never the test split: the tuned
    weight must not see the pairs the final report will score.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    fallback = damped_item_mean(split.train)
    outcomes: list[MethodOutcome] = []
    for weight in weights:
        scorer = HybridScorer(collaborative, content, weight=weight)
        outcomes.append(
            evaluate_method(
                scorer,
                split,
                fallback_predictions=fallback,
                k=k,
                chunk_size=chunk_size,
                track_memory=False,
            )
        )
    best = max(
        outcomes,
        key=lambda outcome: (outcome.ndcg_at_k or 0.0, outcome.precision_at_k or 0.0),
    )
    return float(best.params["collaborative_weight"]), outcomes


def load_validation_split(
    processed_dir: str | Path,
    *,
    users: int = 1_500,
    validation_fraction: float = 0.15,
    seed: int = 42,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> SplitData:
    """Carve a chronological validation slice out of the *training* ratings.

    The test split stays sealed. Each sampled user's most recent ``validation_fraction`` of their
    training ratings becomes the target, and the matrix is rebuilt without them, so the weight is
    tuned on pairs the final evaluation will never touch.
    """
    processed = Path(processed_dir)
    train_path = processed / "ratings_train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"missing {train_path.name}; run scripts/prepare_data.py first"
        )
    if users < 1:
        raise ValueError("users must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    columns = ["user_index", "movie_index", "rating", "timestamp"]
    ratings = pd.read_parquet(train_path, columns=columns)
    if ratings.empty:
        raise ValueError("the training split is empty")
    candidates = np.unique(ratings["user_index"].to_numpy(dtype="int64"))
    if users < candidates.size:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(candidates, size=users, replace=False))
    else:
        chosen = candidates
    ratings = ratings[ratings["user_index"].isin(chosen)].reset_index(drop=True)

    # ``chronological_split`` works on raw MovieLens column names; the processed split keeps both
    # raw ids and compact indexes, so the rename is only for the helper's benefit.
    ordered = ratings.rename(columns={"user_index": "userId", "movie_index": "movieId"})
    earlier, later = chronological_split(ordered, validation_fraction)
    earlier = earlier.reset_index(drop=True)
    later = later.reset_index(drop=True)

    # Keep the full model dimensions when the id maps are present: SVD factors and content rows
    # are indexed against the whole universe, not just the sampled users' slice of it.
    user_map = processed / "user_id_map.parquet"
    movie_map = processed / "movie_id_map.parquet"
    n_users = (
        len(pd.read_parquet(user_map, columns=["user_index"]))
        if user_map.is_file()
        else int(ratings["user_index"].max()) + 1
    )
    n_items = (
        len(pd.read_parquet(movie_map, columns=["movie_index"]))
        if movie_map.is_file()
        else int(ratings["movie_index"].max()) + 1
    )
    training = earlier.rename(
        columns={"userId": "user_index", "movieId": "movie_index"}
    )
    validation = later.rename(
        columns={"userId": "user_index", "movieId": "movie_index"}
    )
    matrix = build_csr(training, n_users=n_users, n_movies=n_items)
    return split_from_arrays(
        matrix,
        validation["user_index"].to_numpy(dtype="int64"),
        validation["movie_index"].to_numpy(dtype="int64"),
        validation["rating"].to_numpy(dtype="float64"),
        evaluated_users=chosen,
        relevance_threshold=relevance_threshold,
    )


def render_markdown_table(outcomes: Sequence[MethodOutcome]) -> str:
    """A compact results table for the README."""

    def cell(value: float | None, digits: int = 4) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "| Method | RMSE | MAE | P@10 | R@10 | NDCG@10 | Coverage | Eval. users |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for outcome in outcomes:
        lines.append(
            "| {method} | {rmse} | {mae} | {p} | {r} | {ndcg} | {coverage} | {users:,} |".format(
                method=outcome.method,
                rmse=cell(outcome.rmse),
                mae=cell(outcome.mae),
                p=cell(outcome.precision_at_k),
                r=cell(outcome.recall_at_k),
                ndcg=cell(outcome.ndcg_at_k),
                coverage=(
                    "—"
                    if outcome.catalogue_coverage is None
                    else f"{outcome.catalogue_coverage:.2%}"
                ),
                users=outcome.evaluated_users,
            )
        )
    return "\n".join(lines)
