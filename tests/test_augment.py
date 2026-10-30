"""PLAN 5.11: app ratings join the training data, and a film joins once it has support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.augment import (
    SOURCE_APP,
    SOURCE_ORIGINAL,
    USER_ID_OFFSET,
    merge_app_ratings,
    strip_app_rows,
    validate_app_ratings,
)


def base_artifacts() -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Two MovieLens users on two films, mirroring what prepare_data.py writes."""
    user_map = pd.DataFrame({"userId": [1, 2], "user_index": [0, 1]})
    movie_map = pd.DataFrame({"movieId": [10, 20], "movie_index": [0, 1]})
    matrix = sparse.csr_matrix(
        np.array([[4.0, 0.0], [0.0, 3.5]], dtype="float32"),
    )
    return matrix, user_map, movie_map


def app_ratings(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def test_a_known_film_gains_a_user_row_without_moving_existing_indexes() -> None:
    matrix, user_map, movie_map = base_artifacts()
    merged = merge_app_ratings(matrix, user_map, movie_map, app_ratings([(7, 10, 5.0)]))

    assert merged.matrix.shape == (3, 2)
    # The MovieLens block is untouched: same values, same coordinates.
    assert merged.matrix[:2].toarray().tolist() == [[4.0, 0.0], [0.0, 3.5]]
    assert merged.matrix[2].toarray().tolist() == [[5.0, 0.0]]
    assert merged.movie_map["movieId"].tolist() == [10, 20]
    assert merged.user_map["user_index"].tolist() == [0, 1, 2]
    # A database id must never be written as if it were a MovieLens id.
    assert int(merged.user_map.loc[2, "userId"]) == USER_ID_OFFSET + 7
    assert merged.user_map.loc[2, "source"] == SOURCE_APP
    assert merged.user_map.loc[0, "source"] == SOURCE_ORIGINAL
    assert merged.stats.ratings_merged == 1
    assert merged.stats.users_added == 1
    assert merged.stats.movies_added == 0


def test_a_new_film_promotes_only_when_it_reaches_the_threshold() -> None:
    matrix, user_map, movie_map = base_artifacts()
    ratings = app_ratings([(7, 99, 4.5), (8, 99, 5.0), (9, 99, 3.0)])

    merged = merge_app_ratings(
        matrix, user_map, movie_map, ratings, min_item_ratings=3
    )

    assert merged.movie_map["movieId"].tolist() == [10, 20, 99]
    assert merged.movie_map.loc[2, "source"] == SOURCE_APP
    assert merged.matrix.shape == (5, 3)
    # Each new user's rating sits in the new column and nowhere else.
    assert merged.matrix[2:, 2].toarray().ravel().tolist() == [4.5, 5.0, 3.0]
    assert merged.stats.promoted == (99,)
    assert merged.stats.below_threshold == ()


def test_a_film_without_enough_support_keeps_its_ratings_out_of_the_model() -> None:
    matrix, user_map, movie_map = base_artifacts()

    merged = merge_app_ratings(
        matrix, user_map, movie_map, app_ratings([(7, 99, 4.5)]), min_item_ratings=20
    )

    # Nothing is added at all: the only rating was on a film the collaborative models cannot
    # learn from yet, and the content model already covers it.
    assert merged.matrix.shape == (2, 2)
    assert merged.user_map["userId"].tolist() == [1, 2]
    assert merged.stats.below_threshold == (99,)
    assert merged.stats.ratings_merged == 0


def test_prior_catalog_ratings_count_towards_the_threshold() -> None:
    matrix, user_map, movie_map = base_artifacts()

    # A catalogue film with 18 MovieLens ratings needs two app ratings, not twenty.
    merged = merge_app_ratings(
        matrix,
        user_map,
        movie_map,
        app_ratings([(7, 99, 4.0), (8, 99, 4.0)]),
        min_item_ratings=20,
        prior_counts={99: 18},
    )

    assert merged.stats.promoted == (99,)
    assert merged.movie_map["movieId"].tolist() == [10, 20, 99]


def test_duplicate_ratings_collapse_and_are_reported() -> None:
    matrix, user_map, movie_map = base_artifacts()

    merged = merge_app_ratings(
        matrix,
        user_map,
        movie_map,
        app_ratings([(7, 10, 5.0), (7, 10, 2.0)]),
    )

    assert merged.stats.ratings_merged == 1
    assert merged.stats.duplicates == 1
    # The last write wins, exactly as the API's upsert would have left it.
    assert merged.matrix[2, 0] == pytest.approx(2.0)


def test_the_merge_is_idempotent_so_a_retrain_can_run_daily() -> None:
    matrix, user_map, movie_map = base_artifacts()
    ratings = app_ratings([(7, 10, 5.0), (7, 99, 3.0), (8, 99, 4.0), (9, 99, 4.5)])
    first = merge_app_ratings(
        matrix, user_map, movie_map, ratings, min_item_ratings=3
    )

    second = merge_app_ratings(
        first.matrix, first.user_map, first.movie_map, ratings, min_item_ratings=3
    )

    assert second.matrix.shape == first.matrix.shape
    assert second.matrix.nnz == first.matrix.nnz
    assert second.movie_map["movieId"].tolist() == [10, 20, 99]
    assert second.user_map["userId"].tolist() == first.user_map["userId"].tolist()
    assert second.stats.previous_users_stripped == 3
    assert second.stats.previous_movies_stripped == 1
    # And the numbers do not creep: the same rows are merged again, not added again.
    assert second.stats.ratings_merged == first.stats.ratings_merged
    assert second.matrix.toarray().tolist() == first.matrix.toarray().tolist()


def test_an_empty_app_table_removes_rows_a_previous_merge_added() -> None:
    matrix, user_map, movie_map = base_artifacts()
    merged = merge_app_ratings(matrix, user_map, movie_map, app_ratings([(7, 10, 5.0)]))

    cleared = merge_app_ratings(
        merged.matrix, merged.user_map, merged.movie_map, app_ratings([])
    )

    assert cleared.matrix.shape == (2, 2)
    assert cleared.movie_map["source"].tolist() == [SOURCE_ORIGINAL, SOURCE_ORIGINAL]
    assert cleared.stats.previous_users_stripped == 1


def test_strip_app_rows_leaves_a_map_without_provenance_alone() -> None:
    matrix, user_map, movie_map = base_artifacts()

    stripped_matrix, users, movies, users_stripped, movies_stripped = strip_app_rows(
        matrix, user_map, movie_map
    )

    assert users_stripped == 0
    assert movies_stripped == 0
    assert stripped_matrix.shape == matrix.shape
    assert users["source"].tolist() == [SOURCE_ORIGINAL, SOURCE_ORIGINAL]
    assert movies["source"].tolist() == [SOURCE_ORIGINAL, SOURCE_ORIGINAL]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(7, 10, 4.2)], "half-stars"),
        ([(7, 10, 6.0)], "between 0.5 and 5.0"),
        ([(7, 10, 0.0)], "between 0.5 and 5.0"),
        ([(7, 10, float("nan"))], "finite"),
    ],
)
def test_ratings_the_api_would_have_rejected_are_refused(
    rows: list[tuple[int, int, float]], message: str
) -> None:
    matrix, user_map, movie_map = base_artifacts()
    with pytest.raises(ValueError, match=message):
        merge_app_ratings(matrix, user_map, movie_map, app_ratings(rows))


def test_missing_columns_are_named() -> None:
    with pytest.raises(ValueError, match="user_id, movie_id, rating"):
        validate_app_ratings(pd.DataFrame({"user_id": [1], "movie_id": [10]}))


def test_a_user_id_that_would_collide_with_the_offset_is_refused() -> None:
    matrix, user_map, movie_map = base_artifacts()
    with pytest.raises(ValueError, match="collides with the synthetic offset"):
        merge_app_ratings(matrix, user_map, movie_map, app_ratings([(USER_ID_OFFSET, 10, 4.0)]))


def test_maps_and_matrix_must_agree() -> None:
    matrix, user_map, movie_map = base_artifacts()
    with pytest.raises(ValueError, match="column counts"):
        merge_app_ratings(matrix, user_map, movie_map.iloc[:1], app_ratings([(7, 10, 4.0)]))
    with pytest.raises(ValueError, match="row counts"):
        merge_app_ratings(matrix, user_map.iloc[:1], movie_map, app_ratings([(7, 10, 4.0)]))


def test_a_zero_threshold_is_rejected() -> None:
    matrix, user_map, movie_map = base_artifacts()
    with pytest.raises(ValueError, match="positive"):
        merge_app_ratings(
            matrix, user_map, movie_map, app_ratings([(7, 10, 4.0)]), min_item_ratings=0
        )
