"""Offline evaluation metrics: rating error and top-K ranking quality.

Conventions come from ``PLAN.md`` §9: precision@K, recall@K and NDCG@K with K=10, and a held-out
rating of 4.0 or higher counting as *relevant*. Averages cover only users who have at least one
relevant held-out item; users with none carry no ranking signal and are reported as skipped rather
than silently scored as zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")


def _paired_arrays(actual: object, predicted: object) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype="float64")
    predicted_array = np.asarray(predicted, dtype="float64")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    return actual_array, predicted_array


def rmse(actual: object, predicted: object) -> float:
    """Root mean squared error over held-out rating pairs."""
    actual_array, predicted_array = _paired_arrays(actual, predicted)
    if actual_array.size == 0:
        raise ValueError("rmse needs at least one pair")
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def mae(actual: object, predicted: object) -> float:
    """Mean absolute error over held-out rating pairs."""
    actual_array, predicted_array = _paired_arrays(actual, predicted)
    if actual_array.size == 0:
        raise ValueError("mae needs at least one pair")
    return float(np.mean(np.abs(actual_array - predicted_array)))


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top-k list that is relevant; a short list is scored against ``k``."""
    _check_k(k)
    if not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the user's relevant held-out items that appear in the top-k list."""
    _check_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@k with the ideal ranking truncated at ``min(k, len(relevant))``."""
    _check_k(k)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


@dataclass(frozen=True)
class RankingMetrics:
    """Averaged ranking quality plus the sample sizes behind it."""

    k: int
    evaluated_users: int
    skipped_users: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    catalogue_coverage: float
    mean_list_length: float


def average_ranking_metrics(
    recommendations: Sequence[Sequence[int]],
    relevant_sets: Sequence[set[int]],
    *,
    k: int,
    n_items: int,
) -> RankingMetrics:
    """Average precision/recall/NDCG@k over users, and catalogue coverage of the served lists."""
    _check_k(k)
    if n_items < 1:
        raise ValueError("n_items must be positive")
    if len(recommendations) != len(relevant_sets):
        raise ValueError("recommendations and relevant_sets must be the same length")

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    served: set[int] = set()
    lengths: list[int] = []
    skipped = 0

    for recommended, relevant in zip(recommendations, relevant_sets, strict=True):
        list_length = min(len(recommended), k)
        lengths.append(list_length)
        served.update(int(item) for item in recommended[:k])
        if not relevant:
            skipped += 1
            continue
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    evaluated = len(precisions)
    return RankingMetrics(
        k=k,
        evaluated_users=evaluated,
        skipped_users=skipped,
        precision_at_k=float(np.mean(precisions)) if evaluated else 0.0,
        recall_at_k=float(np.mean(recalls)) if evaluated else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if evaluated else 0.0,
        catalogue_coverage=len(served) / n_items,
        mean_list_length=float(np.mean(lengths)) if lengths else 0.0,
    )
