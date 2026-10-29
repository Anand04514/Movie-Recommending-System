"""Content-based features and recommendations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.content import (
    build_content_features,
    build_profile,
    extract_tag_features,
    load_features,
    recommend_content,
    save_features,
    score_items,
)


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [10, 11, 12, 13, 14, 15, 16],
            "title": [
                "Alpha",
                "Beta",
                "Gamma",
                "Delta",
                "Epsilon",
                "Zeta",
                "Eta",
            ],
            "genres": [
                "Action|Sci-Fi",
                "Action|Adventure",
                "Romance",
                "Romance|Drama",
                "Documentary",
                "Action",
                "Romance",
            ],
            "release_year": [1999, 2001, 2004, 2018, 2022, 2015, 1995],
        }
    )


@pytest.fixture
def features():
    return build_content_features(catalog())


def test_features_are_unit_length_and_named(features) -> None:
    norms = np.sqrt(np.asarray(features.matrix.multiply(features.matrix).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)
    assert "genre:Action" in features.feature_names
    assert "decade:1990" in features.feature_names
    assert features.matrix.shape[0] == 7
    assert features.row_for(10) == 0
    assert features.row_for(999) is None


def test_rows_for_aligns_to_the_requested_order(features) -> None:
    rows = features.rows_for(np.array([16, 10]))
    assert rows.shape[0] == 2
    assert np.allclose(rows.toarray()[0], features.matrix[features.index[16]].toarray()[0])
    with pytest.raises(KeyError):
        features.rows_for(np.array([10, 4242]))


def test_profile_ranks_unseen_films_from_the_liked_genre(features) -> None:
    # Alpha and Beta are action films; Gamma was disliked.
    rated = np.array([features.index[10], features.index[11], features.index[12]])
    recs = recommend_content(
        features,
        rated,
        np.array([5.0, 4.0, 1.0]),
        limit=3,
    )
    titles = [catalog().iloc[rec.item_index]["title"] for rec in recs]

    assert titles[0] == "Zeta"  # action, not yet rated
    assert "Epsilon" not in titles  # no overlap with the profile
    assert set(rec.item_index for rec in recs).isdisjoint({0, 1, 2})
    assert all(0.5 <= rec.predicted_rating <= 5.0 for rec in recs)


def test_a_single_rating_still_builds_a_profile(features) -> None:
    """One onboarding rating must be enough: there is no mean to rise above yet."""
    profile = build_profile(features, np.array([features.index[10]]), np.array([5.0]))
    assert profile.has_signal
    assert profile.weights.tolist() == [2.0]
    assert recommend_content(features, np.array([features.index[10]]), np.array([5.0]), limit=1)


def test_profile_ignores_films_the_user_did_not_enjoy(features) -> None:
    rated = np.array([features.index[10], features.index[13]])
    profile = build_profile(features, rated, np.array([1.0, 1.0]))
    assert not profile.has_signal
    assert not np.any(score_items(features, profile))
    assert recommend_content(features, rated, np.array([1.0, 1.0])) == []


def test_catalog_films_outside_the_model_can_be_recommended(features) -> None:
    """Features know nothing about rating counts, which is the point for rare and new films."""
    recs = recommend_content(features, np.array([features.index[10]]), np.array([5.0]), limit=7)
    assert features.index[15] in {rec.item_index for rec in recs}


def test_exclusions_and_explanations_are_honest(features) -> None:
    rated = np.array([features.index[10], features.index[12]])
    recs = recommend_content(
        features, rated, np.array([5.0, 1.0]), limit=5, exclude={features.index[15]}
    )
    assert features.index[15] not in {rec.item_index for rec in recs}

    for rec in recs:
        for contribution in rec.contributions:
            assert contribution.item_index in {features.index[10], features.index[12]}
            assert contribution.strength > 0
        if rec.contributions:
            # Gamma was disliked and has no positive weight, so only Alpha can be cited.
            assert rec.source_index == features.index[10]


def test_tags_shift_similarity(features) -> None:
    tagged = build_content_features(
        pd.DataFrame(
            {
                "movie_id": [1, 2, 3],
                "genres": ["Drama", "Drama", "Drama"],
                "release_year": [2000, 2000, 2000],
            }
        ),
        tags=pd.DataFrame(
            {
                "movie_id": [1, 2, 3],
                "tag": ["space", "space", "noir"],
                "count": [10, 8, 9],
            }
        ),
    )
    # Row 1 was rated, so the space-tagged row 0 should beat the noir row 2.
    recs = recommend_content(tagged, np.array([1]), np.array([5.0]), limit=2)
    assert [rec.item_index for rec in recs] == [0, 2]


def test_tag_extraction_keeps_frequent_tags_only(tmp_path: Path) -> None:
    path = tmp_path / "tags.csv"
    pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 3, 3, 4],
            "movieId": [10, 10, 11, 10, 10, 11, 12],
            "tag": ["Space", "space", "noir", "space", "space", "noir", "typo"],
            "timestamp": [1, 2, 3, 4, 5, 6, 7],
        }
    ).to_csv(path, index=False)

    tags = extract_tag_features(path, chunksize=2, top_tags=2, min_count=2)
    counts = {
        (row.movie_id, row.tag): row.count for row in tags.itertuples(index=False)
    }
    assert counts == {(10, "space"): 4, (11, "noir"): 2}

    top_one = extract_tag_features(path, chunksize=1, top_tags=1, min_count=1)
    assert set(top_one["tag"]) == {"space"}
    assert set(top_one["movie_id"]) == {10}

    assert extract_tag_features(path, min_count=99).empty


def test_feature_round_trip(tmp_path: Path, features) -> None:
    path = tmp_path / "content_features.npz"
    save_features(features, path)
    loaded = load_features(path)

    assert loaded.movie_ids.tolist() == features.movie_ids.tolist()
    assert loaded.feature_names == features.feature_names
    assert np.allclose(loaded.matrix.toarray(), features.matrix.toarray())

    recs = recommend_content(loaded, np.array([loaded.index[10]]), np.array([5.0]), limit=3)
    assert recs and recs[0].item_index == loaded.index[15]


def test_missing_feature_file_says_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_and_precompute"):
        load_features(tmp_path / "absent.npz")


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="missing column"):
        build_content_features(pd.DataFrame({"movie_id": [1], "title": ["x"]}))
    with pytest.raises(ValueError, match="duplicate"):
        build_content_features(
            pd.DataFrame(
                {"movie_id": [1, 1], "genres": ["Drama", "Comedy"], "release_year": [2000, 2001]}
            )
        )

    features = build_content_features(catalog())
    with pytest.raises(ValueError, match="same shape"):
        build_profile(features, np.array([0, 1]), np.array([4.0]))
    with pytest.raises(IndexError):
        build_profile(features, np.array([99]), np.array([4.0]))
    with pytest.raises(ValueError, match="positive"):
        recommend_content(features, np.array([0]), np.array([4.0]), limit=0)
