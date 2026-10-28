"""Serving-side user CF: folding a user who is not in the training matrix into it."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from recsys.user_cf import (
    build_fold_in_model,
    fold_in,
    recommend_for_folded_user,
    score_items,
)

# u0 and u2 agree with the folded user below; u1 is deliberately inverted.
TRAIN = [
    [5.0, 4.0, 5.0, 0.0, 0.0],  # u0, mean 4.667
    [4.0, 5.0, 4.0, 0.0, 0.0],  # u1, mean 4.333 (inverted relative to the new user)
    [5.0, 4.5, 0.0, 5.0, 4.0],  # u2, mean 4.625
]


def build_train() -> sparse.csr_matrix:
    return sparse.csr_matrix(np.asarray(TRAIN, dtype="float32"))


def folded_user():
    model = build_fold_in_model(build_train())
    return model, fold_in(model, np.array([0, 1]), np.array([5.0, 4.5]))


def test_fold_in_keeps_agreeing_users_and_drops_inverted_ones() -> None:
    _, folded = folded_user()

    assert folded.rated_count == 2
    assert folded.mean == pytest.approx(4.75)
    assert folded.variance == pytest.approx(0.125)
    assert set(folded.neighbor_indexes.tolist()) == {0, 2}
    assert 1 not in folded.neighbor_indexes  # u1 correlates negatively and must not be a neighbour
    assert np.all(folded.similarities > 0)


def test_fold_in_scores_items_from_the_neighbours_own_ratings() -> None:
    model, folded = folded_user()
    scores = score_items(model, folded)

    # u0 (highly similar) rated item 2 a 5; u2 rated items 3 and 4.
    assert scores.scores[2] > 0 and scores.weights[2] > 0
    assert scores.scores[3] > 0
    assert scores.scores[4] < 0  # u2 rated item 4 below their own mean
    assert scores.predicted[2] > 4.5
    assert scores.predicted[4] < scores.predicted[2]
    assert scores.rater_counts[2] == 1
    assert scores.rater_counts[3] == 1
    assert scores.rater_counts[0] == 2  # both neighbours rated item 0


def test_recommendations_require_positive_support_and_exclude_rated_items() -> None:
    model, folded = folded_user()
    recs = recommend_for_folded_user(model, folded, limit=5)

    indexes = {rec.item_index for rec in recs}
    # Item 4 has negative support, and the user's own rated items come back excluded by default.
    assert indexes == {2, 3}
    assert all(rec.neighbors_rated == 1 for rec in recs)
    assert {rec.item_index for rec in recommend_for_folded_user(model, folded, exclude={2})} == {3}
    assert all(0.5 <= rec.predicted_rating <= 5.0 for rec in recs)


def test_a_user_with_no_deviation_gets_no_neighbours() -> None:
    model = build_fold_in_model(build_train())

    flat = fold_in(model, np.array([0, 1, 2]), np.array([4.0, 4.0, 4.0]))
    assert flat.has_neighbors is False
    assert flat.variance == 0.0

    scores = score_items(model, flat)
    # With nothing to compare, every item falls back to the user's own mean.
    assert np.allclose(scores.predicted, 4.0)
    assert np.all(scores.weights == 0)
    assert recommend_for_folded_user(model, flat) == []


def test_empty_and_out_of_range_input_is_handled() -> None:
    model = build_fold_in_model(build_train())

    empty = fold_in(model, np.array([], dtype="int64"), np.array([], dtype="float64"))
    assert empty.rated_count == 0
    assert empty.has_neighbors is False
    assert np.allclose(score_items(model, empty).predicted, 0.5)  # unclipped default mean 0 -> 0.5

    with pytest.raises(IndexError, match="outside the fold-in model"):
        fold_in(model, np.array([99]), np.array([5.0]))
    with pytest.raises(ValueError, match="same shape"):
        fold_in(model, np.array([0, 1]), np.array([5.0]))
    with pytest.raises(ValueError, match="k must be positive"):
        fold_in(model, np.array([0]), np.array([5.0]), k=0)
    with pytest.raises(ValueError, match="limit must be positive"):
        recommend_for_folded_user(model, empty, limit=0)
    with pytest.raises(ValueError, match="damping must be non-negative"):
        score_items(model, empty, damping=-1.0)
    assert recommend_for_folded_user(model, empty) == []
