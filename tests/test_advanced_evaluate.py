"""SVD, content and hybrid entries in the offline evaluation harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.content import build_content_features
from recsys.evaluate import (
    ContentScorer,
    SvdScorer,
    default_scorers,
    evaluate_split,
    load_validation_split,
    tune_hybrid_weight,
)
from recsys.item_cf import build_item_neighbors
from recsys.svd import train_svd

TRAIN_ROWS = [
    [5.0, 4.0, 1.0, 0.0],
    [4.0, 5.0, 2.0, 0.0],
    [5.0, 4.0, 1.0, 2.0],
    [1.0, 2.0, 5.0, 4.0],
    [1.0, 1.0, 4.0, 5.0],
    [2.0, 1.0, 5.0, 4.0],
    [5.0, 0.0, 0.0, 0.0],
]
MOVIE_IDS = [100, 101, 102, 103]


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": MOVIE_IDS,
            "genres": ["Action|Sci-Fi", "Action", "Romance|Drama", "Drama"],
            "release_year": [1999, 2003, 2004, 1995],
        }
    )


def build_split():
    from recsys.evaluate import split_from_arrays

    return split_from_arrays(
        sparse.csr_matrix(np.asarray(TRAIN_ROWS, dtype="float32")),
        np.array([0, 1, 6]),
        np.array([3, 3, 3]),
        np.array([5.0, 5.0, 4.0]),
    )


def build_scorers(split, weight: float = 0.5):
    neighbors = build_item_neighbors(split.train, k=2, block_size=3)
    model = train_svd(split.train, rank=2)
    features = build_content_features(catalog())
    scorers = default_scorers(
        split.train,
        item_neighbors=neighbors,
        svd_model=model,
        content_features=features,
        model_movie_ids=np.array(MOVIE_IDS),
        hybrid_weight=weight,
    )
    return scorers, model, features


def test_all_advanced_methods_join_the_report() -> None:
    split = build_split()
    scorers, model, _features = build_scorers(split)

    assert [scorer.name for scorer in scorers] == [
        "global_mean",
        "damped_item_mean",
        "popularity",
        "item_cf",
        "svd",
        "content",
        "hybrid",
    ]
    assert isinstance(scorers[4], SvdScorer)
    assert isinstance(scorers[5], ContentScorer)
    assert scorers[6].params["collaborative_weight"] == 0.5
    assert scorers[4].params["rank"] == model.rank

    outcomes = evaluate_split(split, scorers, k=2, chunk_size=2, track_memory=False)
    assert [outcome.method for outcome in outcomes] == [scorer.name for scorer in scorers]
    for outcome in outcomes:
        assert 0.0 <= outcome.rmse <= 4.0
        assert outcome.rated_pairs == 3
    for outcome in outcomes[3:]:
        assert outcome.precision_at_k is not None
        assert outcome.ndcg_at_k is not None


def test_svd_scorer_predictions_stay_in_the_rating_range() -> None:
    split = build_split()
    model = train_svd(split.train, rank=2)
    predictions, weights, ranking = SvdScorer(model).blocks(np.array([0, 3, 6]))
    assert weights is None
    assert predictions.shape == ranking.shape == (3, 4)
    assert np.all((predictions >= 0.5) & (predictions <= 5.0))
    assert np.isfinite(ranking).all()


def test_content_scorer_marks_zero_overlap_items_unrankable() -> None:
    split = build_split()
    # A documentary from the 2010s shares no genre or decade with the action films u0 loved.
    separated = pd.DataFrame(
        {
            "movie_id": MOVIE_IDS,
            "genres": ["Action|Sci-Fi", "Action", "Romance|Drama", "Documentary"],
            "release_year": [1999, 2003, 2004, 2019],
        }
    )
    features = build_content_features(separated)
    scorer = ContentScorer(
        split.train,
        features.rows_for(np.array(MOVIE_IDS)),
        features=features,
    )
    _predictions, weights, ranking = scorer.blocks(np.array([0]))
    assert weights is None
    # u0 loved the action films and disliked the romance one. Item 3 shares neither a genre nor a
    # decade with anything they liked, so it cannot be ranked; item 2 still matches on the decade.
    assert np.isneginf(ranking[0, 3])
    assert 0 < ranking[0, 2] < ranking[0, 0]
    assert np.isfinite(ranking[0, 0]) and np.isfinite(ranking[0, 1])
    assert scorer.params["feature_kinds"] == {"genres": 5, "decades": 3, "tags": 0}


def test_hybrid_weight_is_tuned_without_touching_the_test_split() -> None:
    split = build_split()
    scorers, _model, _features = build_scorers(split)
    collaborative, content = scorers[3], scorers[5]

    weight, outcomes = tune_hybrid_weight(
        split,
        collaborative=collaborative,
        content=content,
        weights=[0.0, 0.5, 1.0],
        k=2,
        chunk_size=7,
    )
    assert weight in {0.0, 0.5, 1.0}
    assert [outcome.params["collaborative_weight"] for outcome in outcomes] == [0.0, 0.5, 1.0]
    assert all(outcome.method == "hybrid" for outcome in outcomes)
    best = max(
        outcomes,
        key=lambda outcome: (outcome.ndcg_at_k or 0.0, outcome.precision_at_k or 0.0),
    )
    assert best.params["collaborative_weight"] == pytest.approx(weight)


def test_validation_split_rebuilds_a_smaller_training_matrix(tmp_path: Path) -> None:
    rows = []
    for user in (0, 1):
        for offset, item in enumerate((0, 1, 2, 3)):
            rows.append((user, item, 5.0 - offset * 0.5, 1_000 + offset + user * 100))
    pd.DataFrame(
        rows,
        columns=["user_index", "movie_index", "rating", "timestamp"],
    ).to_parquet(tmp_path / "ratings_train.parquet", index=False)
    pd.DataFrame(
        {"userId": [1, 2, 3], "user_index": [0, 1, 2]}
    ).to_parquet(tmp_path / "user_id_map.parquet", index=False)
    pd.DataFrame(
        {"movieId": [10, 11, 12, 13, 14], "movie_index": [0, 1, 2, 3, 4]}
    ).to_parquet(tmp_path / "movie_id_map.parquet", index=False)

    split = load_validation_split(tmp_path, users=2, validation_fraction=0.25)
    assert split.train.shape == (3, 5)
    assert split.n_pairs == 2  # the most recent rating of each user
    assert split.evaluated_users.tolist() == [0, 1]
    assert set(split.test_items.tolist()) == {3}
    assert all(split.train.getrow(user).nnz == 3 for user in (0, 1))

    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_validation_split(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="users must be positive"):
        load_validation_split(tmp_path, users=0)
