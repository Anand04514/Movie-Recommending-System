"""Phase 3.1: rating and ranking metrics."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.metrics import (
    average_ranking_metrics,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_rmse_and_mae_match_hand_computed_values() -> None:
    actual = [5.0, 3.0, 1.0]
    predicted = [4.5, 3.0, 2.0]
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((0.25 + 0.0 + 1.0) / 3))
    assert mae(actual, predicted) == pytest.approx((0.5 + 0.0 + 1.0) / 3)


def test_rating_metrics_reject_empty_or_mismatched_input() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        rmse([], [])
    with pytest.raises(ValueError, match="at least one pair"):
        mae([], [])
    with pytest.raises(ValueError, match="same shape"):
        mae([1.0, 2.0], [1.0])


def test_precision_recall_and_ndcg_on_a_known_list() -> None:
    recommended = [10, 11, 12, 13]
    relevant = {11, 13, 99}

    assert precision_at_k(recommended, relevant, 4) == pytest.approx(2 / 4)
    assert precision_at_k(recommended, relevant, 2) == pytest.approx(1 / 2)
    assert recall_at_k(recommended, relevant, 4) == pytest.approx(2 / 3)
    # Hits sit at ranks 2 and 4; the ideal list holds three relevant items.
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    ideal = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(recommended, relevant, 4) == pytest.approx(dcg / ideal)


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_when_hits_move_down() -> None:
    relevant = {1, 2}
    assert ndcg_at_k([1, 2, 3], relevant, 3) == pytest.approx(1.0)
    assert ndcg_at_k([3, 1, 2], relevant, 3) < 1.0


def test_k_and_empty_inputs_are_handled() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k([1], {1}, 0)
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k([1], {1}, 0)
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k([1], {1}, 0)
    assert precision_at_k([], {1}, 10) == 0.0
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0


def test_average_metrics_skip_users_without_relevant_items_and_track_coverage() -> None:
    report = average_ranking_metrics(
        recommendations=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        relevant_sets=[{1, 4}, {4, 5, 6}, set()],
        k=3,
        n_items=10,
    )

    assert report.evaluated_users == 2
    assert report.skipped_users == 1
    assert report.precision_at_k == pytest.approx((1 / 3 + 3 / 3) / 2)
    assert report.recall_at_k == pytest.approx((1 / 2 + 3 / 3) / 2)
    assert 0.0 < report.ndcg_at_k <= 1.0
    assert report.catalogue_coverage == pytest.approx(0.9)  # nine distinct items of ten
    assert report.mean_list_length == 3.0


def test_average_metrics_handle_an_empty_sample() -> None:
    report = average_ranking_metrics(recommendations=[], relevant_sets=[], k=10, n_items=5)
    assert report.evaluated_users == 0
    assert report.precision_at_k == 0.0
    assert report.catalogue_coverage == 0.0
    assert report.mean_list_length == 0.0


def test_average_metrics_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        average_ranking_metrics([[1]], [], k=1, n_items=5)
    with pytest.raises(ValueError, match="n_items must be positive"):
        average_ranking_metrics([[1]], [{1}], k=1, n_items=0)


def test_coverage_counts_only_the_served_top_k() -> None:
    report = average_ranking_metrics(
        recommendations=[[1, 2, 3, 4]],
        relevant_sets=[{1}],
        k=2,
        n_items=4,
    )
    assert report.catalogue_coverage == pytest.approx(0.5)  # items 1 and 2 only
    assert report.mean_list_length == 2.0
