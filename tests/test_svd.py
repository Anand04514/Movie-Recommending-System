"""SVD training, fold-in and explanation behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from recsys.svd import (
    factor_gram,
    fold_in,
    load_model,
    recommend_for_folded_user,
    save_model,
    train_svd,
)


def two_cluster_matrix() -> sparse.csr_matrix:
    """Twelve users, eight films, two clear taste clusters and a little noise."""
    rng = np.random.default_rng(7)
    rows: list[list[float]] = []
    for _ in range(6):
        rows.append([4.5, 4.0, 5.0, 4.5, 1.5, 2.0, 1.0, 1.5])
    for _ in range(6):
        rows.append([1.0, 1.5, 2.0, 1.5, 4.5, 4.0, 5.0, 4.5])
    values = np.asarray(rows, dtype="float64")
    values += rng.normal(0.0, 0.15, size=values.shape)
    values = np.clip(np.round(values * 2) / 2, 0.5, 5.0)
    return sparse.csr_matrix(values, dtype="float32")


@pytest.fixture(scope="module")
def model():
    return train_svd(two_cluster_matrix(), rank=3, seed=3)


def test_training_beats_the_global_mean_on_observed_ratings(model) -> None:
    matrix = two_cluster_matrix().tocoo()
    predicted = np.asarray(model.predict_for_users(np.arange(12)))
    errors = predicted[matrix.row, matrix.col] - matrix.data
    global_errors = matrix.data - model.global_mean

    assert np.sqrt(np.mean(errors**2)) < np.sqrt(np.mean(global_errors**2))
    assert 0.5 <= model.predict_pair(0, 0) <= 5.0
    assert model.rank == 3
    assert model.user_factors.shape == (12, 3)
    assert model.item_factors.shape == (8, 3)
    assert np.isfinite(model.user_factors).all()


def test_rank_is_clamped_for_small_matrices() -> None:
    model = train_svd(two_cluster_matrix(), rank=50)
    assert model.rank == 7  # min(users, items) - 1


def test_fold_in_ranks_the_users_own_cluster_first(model) -> None:
    folded = fold_in(model, np.array([0, 1]), np.array([5.0, 4.5]))
    recs = recommend_for_folded_user(model, folded, limit=4)

    # Items 2 and 3 belong to the cluster this user has shown a taste for; 4-7 do not.
    assert {rec.item_index for rec in recs[:2]} <= {2, 3}
    assert recs[0].predicted_rating > recs[-1].predicted_rating
    assert all(0.5 <= rec.predicted_rating <= 5.0 for rec in recs)
    assert len(recs) == 4


def test_fold_in_never_recommends_a_rated_or_excluded_item(model) -> None:
    folded = fold_in(model, np.array([0, 1, 2]), np.array([5.0, 4.5, 4.0]))
    recs = recommend_for_folded_user(model, folded, limit=10, exclude={3})
    served = {rec.item_index for rec in recs}

    assert not served & {0, 1, 2}
    assert 3 not in served
    assert served == {4, 5, 6, 7}


def test_a_cold_user_gets_no_svd_picks(model) -> None:
    folded = fold_in(model, np.empty(0, dtype="int64"), np.empty(0, dtype="float64"))
    assert folded.rated_count == 0
    assert recommend_for_folded_user(model, folded, limit=5) == []


def test_explanations_come_only_from_liked_ratings(model) -> None:
    # One loved film and one disliked film from the *other* cluster. The disliked rating must not
    # be offered as a reason to watch its neighbours.
    folded = fold_in(model, np.array([0, 4]), np.array([5.0, 1.0]))
    recs = recommend_for_folded_user(model, folded, limit=3, explain=3)
    assert recs
    for rec in recs:
        for contribution in rec.contributions:
            assert contribution.item_index in {0, 4}
            assert contribution.strength > 0
        if rec.contributions:
            assert rec.source_index == rec.contributions[0].item_index


def test_gram_is_regularised_and_fold_in_matches_the_default_path(model) -> None:
    gram = factor_gram(model)
    assert gram.shape == (model.rank, model.rank)
    assert np.allclose(gram, gram.T)

    indices = np.array([0, 1])
    values = np.array([5.0, 4.5])
    cached = fold_in(model, indices, values, gram=gram)
    fresh = fold_in(model, indices, values)
    assert np.allclose(cached.factors, fresh.factors)


def test_model_round_trips_through_disk(tmp_path: Path, model) -> None:
    path = tmp_path / "svd_model.npz"
    save_model(model, path)
    loaded = load_model(path)

    assert loaded.rank == model.rank
    assert np.allclose(loaded.user_factors, model.user_factors)
    assert np.allclose(loaded.item_factors, model.item_factors)
    folded = fold_in(loaded, np.array([0, 1]), np.array([5.0, 4.5]))
    a = recommend_for_folded_user(loaded, folded, limit=3)
    b = recommend_for_folded_user(model, folded, limit=3)
    assert [rec.item_index for rec in a] == [rec.item_index for rec in b]


def test_missing_model_file_says_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_and_precompute"):
        load_model(tmp_path / "absent.npz")


def test_biases_stay_bounded_on_a_large_sparse_matrix() -> None:
    """Regression: iterating the coupled bias equations used to diverge to 1e9 on real data."""
    rng = np.random.default_rng(11)
    rows, cols = 600, 120
    dense = rng.choice([0.0, 1.0], size=(rows, cols), p=[0.9, 0.1])
    values = np.where(dense > 0, rng.integers(1, 11, size=(rows, cols)) / 2.0, 0.0)
    matrix = sparse.csr_matrix(values)

    model = train_svd(matrix, rank=4)
    assert np.abs(model.user_bias).max() < 5.0
    assert np.abs(model.item_bias).max() < 5.0
    predictions = model.predict_for_users(np.arange(rows))
    assert np.isfinite(predictions).all()
    assert np.abs(predictions).max() < 50.0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="rank"):
        train_svd(two_cluster_matrix(), rank=0)
    with pytest.raises(ValueError, match="no ratings"):
        train_svd(sparse.csr_matrix((4, 4)))
    with pytest.raises(ValueError, match="positive"):
        train_svd(two_cluster_matrix(), factor_regularisation=0.0)

    model = train_svd(two_cluster_matrix(), rank=2)
    with pytest.raises(IndexError):
        fold_in(model, np.array([99]), np.array([4.0]))
    with pytest.raises(ValueError, match="same shape"):
        fold_in(model, np.array([0, 1]), np.array([4.0]))
    with pytest.raises(ValueError, match="positive"):
        recommend_for_folded_user(model, fold_in(model, np.array([0]), np.array([4.0])), limit=0)
