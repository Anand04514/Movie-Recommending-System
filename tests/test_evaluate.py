"""Phase 3.2/3.3: the evaluation harness, on small matrices with hand-checkable numbers.

The synthetic train matrix below leaves each user a gap on purpose:

- u0, u1 rated items 0-2 and never item 3 (their held-out pair).
- u2 rated everything, so it has no candidate left; its one held-out rating is 2.0 and therefore
  not relevant, which exercises the "skip users without relevant items" rule.
- u6 has a single rating, so nothing in the neighbor table can support its held-out pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.baselines import damped_item_mean, global_mean, item_counts_and_sums, item_means
from recsys.evaluate import (
    GlobalMeanScorer,
    ItemVectorScorer,
    _score_pairs,
    _top_k,
    _top_k_lists,
    default_scorers,
    evaluate_method,
    evaluate_split,
    load_split,
    render_markdown_table,
    split_from_arrays,
)
from recsys.item_cf import build_item_neighbors
from recsys.metrics import rmse
from recsys.user_cf import build_user_neighbors

TRAIN_ROWS = [
    [5.0, 4.0, 1.0, 0.0],
    [4.0, 5.0, 2.0, 0.0],
    [5.0, 4.0, 1.0, 2.0],
    [1.0, 2.0, 5.0, 4.0],
    [1.0, 1.0, 4.0, 5.0],
    [2.0, 1.0, 5.0, 4.0],
    [5.0, 0.0, 0.0, 0.0],
]
TRAIN_TOTAL = 73.0  # 10 + 11 + 12 + 12 + 11 + 12 + 5
TRAIN_NNZ = 23
TEST_PAIRS = [
    (0, 3, 5.0),
    (1, 3, 5.0),
    (2, 3, 2.0),
    (6, 3, 4.0),
]
EVALUATED = [0, 1, 6]  # u2 has no relevant held-out item


def build_train(rows: list[list[float]] = TRAIN_ROWS) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.asarray(rows, dtype="float32"))


def build_split(rows: list[list[float]] = TRAIN_ROWS):
    test_users = np.array([user for user, _, _ in TEST_PAIRS], dtype="int64")
    test_items = np.array([item for _, item, _ in TEST_PAIRS], dtype="int64")
    test_ratings = np.array([rating for _, _, rating in TEST_PAIRS], dtype="float64")
    return split_from_arrays(build_train(rows), test_users, test_items, test_ratings)


def test_global_mean_predicts_the_training_mean_everywhere() -> None:
    split = build_split()
    outcome = evaluate_method(
        GlobalMeanScorer(split.train),
        split,
        fallback_predictions=damped_item_mean(split.train),
        k=2,
        track_memory=False,
    )

    assert global_mean(split.train) == pytest.approx(TRAIN_TOTAL / TRAIN_NNZ)
    assert outcome.rmse == pytest.approx(
        rmse(split.test_ratings, np.full(split.n_pairs, TRAIN_TOTAL / TRAIN_NNZ))
    )
    assert outcome.rated_pairs == 4
    # The trivial predictor cannot rank anything, so its ranking columns stay empty.
    assert outcome.precision_at_k is None
    assert outcome.recall_at_k is None
    assert outcome.ndcg_at_k is None
    assert outcome.catalogue_coverage is None
    assert outcome.fallback_rate == 0.0


def test_damped_item_mean_matches_hand_computed_ranking_metrics() -> None:
    split = build_split()
    counts, _ = item_counts_and_sums(split.train)
    damped = damped_item_mean(split.train, damping=10.0)
    scorer = ItemVectorScorer(
        "damped_item_mean",
        prediction_vector=damped,
        ranking_vector=damped,
        counts=counts,
        params={"damping": 10.0},
    )
    outcome = evaluate_method(
        scorer, split, fallback_predictions=damped, k=2, track_memory=False
    )

    # u2's held-out rating is 2.0, so it is not relevant and the user is skipped.
    assert outcome.evaluated_users == 3
    assert outcome.skipped_users == 1
    # Held-out item 3 is every user's best candidate, so each list hits at rank 1.
    assert outcome.precision_at_k == pytest.approx((0.5 + 0.5 + 0.5) / 3)
    assert outcome.recall_at_k == pytest.approx(1.0)
    assert outcome.ndcg_at_k == pytest.approx(1.0)
    assert outcome.catalogue_coverage == pytest.approx(0.5)  # items 2 and 3 were served
    assert outcome.rmse == pytest.approx(rmse(split.test_ratings, damped[split.test_items]))


def test_popularity_ranks_by_count_and_predicts_item_means() -> None:
    split = build_split()
    counts, _ = item_counts_and_sums(split.train)
    means = item_means(split.train)
    scorer = ItemVectorScorer(
        "popularity",
        prediction_vector=means,
        ranking_vector=counts.astype("float64"),
        counts=counts,
        params={"ranking": "rating_count"},
    )
    outcome = evaluate_method(
        scorer, split, fallback_predictions=damped_item_mean(split.train), k=2, track_memory=False
    )

    assert counts.tolist() == [7, 6, 6, 4]
    assert means[3] == pytest.approx(3.75)
    assert outcome.rmse == pytest.approx(rmse(split.test_ratings, np.full(4, 3.75)))
    # u6's single rating leaves items 1-3 as candidates and item 1 wins on count, so only the
    # users whose list leads with item 3 score a hit.
    assert outcome.precision_at_k == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    assert outcome.recall_at_k == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert outcome.evaluated_users == 3


def test_top_k_prefers_higher_scores_and_breaks_ties_by_index() -> None:
    row = np.array([1.0, -np.inf, 3.0, 2.0])
    assert _top_k(row, 2) == [2, 3]
    assert _top_k(row, 10) == [2, 3, 0]
    assert _top_k(np.full(4, -np.inf), 2) == []
    assert _top_k(np.array([1.0, 1.0, 1.0, 1.0]), 2) == [0, 1]


def test_top_k_lists_exclude_train_items_and_unsupported_candidates() -> None:
    split = build_split()
    users = np.array([0, 6])
    # Ranking and weights arrive block-shaped: row i belongs to users[i], not to user id i.
    ranking = np.broadcast_to(np.array([4.0, 3.0, 2.0, 1.0], dtype="float32"), (2, 4))
    weights = np.ones((2, 4), dtype="float32")
    weights[1, 1] = 0.0  # u6's single rating cannot support item 1

    supported = _top_k_lists(split, users, ranking, weights, k=3)
    assert supported[0] == [3]  # u0 rated items 0-2, leaving item 3 as the only candidate
    assert supported[1] == [2, 3]  # item 1 is dropped for lack of support

    # Without support weights, only the user's own train items are excluded.
    plain = _top_k_lists(split, users, ranking, None, k=4)
    assert plain[0] == [3]
    assert plain[1] == [1, 2, 3]

    # Collaborative methods only serve candidates with positive evidence, like the API does.
    negative = np.array([[4.0, 3.0, -1.0, -2.0]], dtype="float32")
    served = _top_k_lists(split, np.array([6]), negative, np.ones((1, 4), dtype="float32"), k=3)
    assert served[0] == [1]  # items 2 and 3 score below zero for u6


def test_unsupported_pairs_fall_back_to_the_item_mean() -> None:
    split = build_split()
    scores = np.zeros((7, 4), dtype="float32")
    weights = np.zeros((7, 4), dtype="float32")
    weights[1, 3] = 0.5  # only u1's held-out pair finds any support
    scores[1, 3] = 0.9
    fallback = np.array([1.0, 2.0, 3.0, 4.0], dtype="float64")
    predicted = np.full(split.n_pairs, -1.0)
    fell_back = np.zeros(split.n_pairs, dtype=bool)

    _score_pairs(
        split, split.evaluated_users, scores, weights, 4.0, fallback, predicted, fell_back
    )

    assert predicted[0] == pytest.approx(4.0)  # u0 has no support, item 3 falls back
    assert predicted[1] == pytest.approx(11 / 3 + 0.9 / 4.5)  # u1's mean plus the fold-in
    assert predicted[2] == pytest.approx(4.0)
    assert predicted[3] == pytest.approx(4.0)
    assert fell_back.tolist() == [True, False, True, True]


def test_default_scorers_cover_the_five_methods_and_fall_back_when_unsupported() -> None:
    split = build_split()
    item_neighbors = build_item_neighbors(split.train, k=2, block_size=3)
    user_neighbors = build_user_neighbors(split.train, k=2, block_size=3)

    scorers = default_scorers(
        split.train, item_neighbors=item_neighbors, user_neighbors=user_neighbors
    )
    assert [scorer.name for scorer in scorers] == [
        "global_mean",
        "damped_item_mean",
        "popularity",
        "item_cf",
        "user_cf",
    ]

    outcomes = evaluate_split(split, scorers, k=2, chunk_size=2, track_memory=False)
    assert [outcome.method for outcome in outcomes] == [scorer.name for scorer in scorers]
    for outcome in outcomes:
        assert outcome.rated_pairs == split.n_pairs
        assert 0.0 <= outcome.rmse <= 4.0
        assert outcome.peak_memory_mb == 0.0  # tracking disabled

    item_cf, user_cf = outcomes[3], outcomes[4]
    # u6's single rating supports nothing, so both collaborative methods must fall back there.
    assert item_cf.fallback_rate == pytest.approx(1 / 4)
    assert user_cf.fallback_rate > 0.0
    assert item_cf.precision_at_k is not None
    assert user_cf.precision_at_k is not None
    assert item_cf.params["damping"] == 4.0


def test_memory_tracking_reports_a_positive_peak() -> None:
    split = build_split()
    outcome = evaluate_method(
        GlobalMeanScorer(split.train),
        split,
        fallback_predictions=damped_item_mean(split.train),
        k=2,
        track_memory=True,
    )
    assert outcome.peak_memory_mb > 0.0


def test_markdown_table_lists_every_method_and_dashes_missing_metrics() -> None:
    split = build_split()
    outcomes = evaluate_split(split, default_scorers(split.train), k=2, track_memory=False)
    table = render_markdown_table(outcomes)

    lines = table.splitlines()
    assert len(lines) == 2 + len(outcomes)
    assert lines[0].startswith("| Method |")
    assert "| popularity |" in table
    global_mean_row = next(line for line in lines if line.startswith("| global_mean |"))
    assert global_mean_row.count("—") == 4  # only RMSE, MAE and the user count are meaningful


def test_load_split_reads_artifacts_and_limits_the_sample(tmp_path: Path) -> None:
    train = build_train(TRAIN_ROWS[:3])
    sparse.save_npz(tmp_path / "train_matrix.npz", train.tocsr())
    pd.DataFrame(
        {
            "user_index": [0, 0, 1, 2],
            "movie_index": [2, 3, 2, 0],
            "rating": [4.0, 2.0, 5.0, 1.0],
        }
    ).to_parquet(tmp_path / "ratings_test.parquet", index=False)

    split = load_split(tmp_path, users=2, seed=1)
    assert split.evaluated_users.size == 2
    assert 2 <= split.n_pairs < 4
    assert set(split.test_users.tolist()) <= set(split.evaluated_users.tolist())
    assert split.train.shape == (3, 4)
    assert split.user_means.tolist() == pytest.approx([10 / 3, 11 / 3, 3.0])
    assert split.pairs_by_user.keys() == set(split.evaluated_users.tolist())

    everyone = load_split(tmp_path, users=99)
    assert everyone.evaluated_users.size == 3
    assert everyone.n_pairs == 4


def test_load_split_validates_its_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_split(tmp_path)

    sparse.save_npz(tmp_path / "train_matrix.npz", build_train(TRAIN_ROWS[:3]).tocsr())
    pd.DataFrame(
        {"user_index": [0], "movie_index": [2], "rating": [4.0]}
    ).to_parquet(tmp_path / "ratings_test.parquet", index=False)

    with pytest.raises(ValueError, match="users must be positive"):
        load_split(tmp_path, users=0)


def test_split_from_arrays_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        split_from_arrays(
            build_train(TRAIN_ROWS[:3]),
            np.array([0, 1]),
            np.array([2]),
            np.array([4.0, 5.0]),
        )
