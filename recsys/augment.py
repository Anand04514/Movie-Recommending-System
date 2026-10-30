"""Merge the app's own ratings into the trained artifacts (PLAN 5.11).

The training matrix is built from MovieLens. A film ingested from TMDB has no MovieLens ratings,
so it stays outside the collaborative model however much the app's users rate it — which is why
this module exists: a retrain can fold the application's ratings into the training data and let
a film join the model once it has enough of them.

Three rules make the merge safe to run repeatedly:

* **Synthetic user ids.** A database user id is unrelated to a MovieLens user id, so merging one as
  if it were the other would attribute a stranger's ratings to a MovieLens profile. App users are
  therefore written as ``userId + USER_ID_OFFSET``, above every real id.
* **Provenance.** Rows and columns added by a previous merge are marked in the id maps and stripped
  before the new merge, so re-running after a retrain cannot count the same rating twice or invent
  a duplicate user.
* **Threshold.** A film outside the model universe joins only when its MovieLens rating count (from
  the catalog) plus its app ratings reaches ``min_item_ratings``. Films below it keep their ratings
  in the database, stay recommendable through the content model, and are reported instead of being
  silently dropped.

This module is pure: it takes frames and returns frames, so it is testable without a database.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

# Above every MovieLens user id (200,948 in ml-32m) and every plausible database id, so the two
# namespaces cannot collide in a merged map.
USER_ID_OFFSET = 10_000_000

SOURCE_ORIGINAL = "movielens"
SOURCE_APP = "app"

_APP_RATING_COLUMNS = ("user_id", "movie_id", "rating")


@dataclass(frozen=True)
class MergeStats:
    """What one merge did, in numbers a log line or a JSON sidecar can report."""

    ratings_merged: int = 0
    users_added: int = 0
    movies_added: int = 0
    duplicates: int = 0
    promoted: tuple[int, ...] = ()
    below_threshold: tuple[int, ...] = ()
    previous_users_stripped: int = 0
    previous_movies_stripped: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "ratings_merged": self.ratings_merged,
            "users_added": self.users_added,
            "movies_added": self.movies_added,
            "duplicates": self.duplicates,
            "promoted": list(self.promoted),
            "below_threshold": list(self.below_threshold),
            "previous_users_stripped": self.previous_users_stripped,
            "previous_movies_stripped": self.previous_movies_stripped,
        }


@dataclass(frozen=True)
class MergedArtifacts:
    """A training matrix and its id maps, ready for the artifact builders."""

    matrix: sparse.csr_matrix
    user_map: pd.DataFrame
    movie_map: pd.DataFrame
    stats: MergeStats = field(default_factory=MergeStats)


def validate_app_ratings(ratings: pd.DataFrame) -> None:
    """Reject anything the API would not have accepted, rather than training on it."""
    missing = [column for column in _APP_RATING_COLUMNS if column not in ratings.columns]
    if missing:
        raise ValueError(f"app ratings need the columns {', '.join(_APP_RATING_COLUMNS)}")
    values = ratings["rating"].to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        raise ValueError("app ratings must be finite numbers")
    if np.any((values < 0.5) | (values > 5.0)):
        raise ValueError("app ratings must sit between 0.5 and 5.0")
    halves = values * 2
    if not np.allclose(halves, np.round(halves), atol=1e-9):
        raise ValueError("app ratings must be whole half-stars")


def _clean(ratings: pd.DataFrame, user_id_offset: int) -> tuple[pd.DataFrame, int]:
    """Validate, deduplicate and type the incoming ratings; return them and the duplicate count."""
    if ratings.empty:
        empty = pd.DataFrame(
            {
                "user_id": pd.Series(dtype="int64"),
                "movie_id": pd.Series(dtype="int64"),
                "rating": pd.Series(dtype="float64"),
            }
        )
        return empty, 0

    validate_app_ratings(ratings)
    frame = ratings.loc[:, list(_APP_RATING_COLUMNS)].copy()
    frame["user_id"] = frame["user_id"].astype("int64")
    frame["movie_id"] = frame["movie_id"].astype("int64")
    frame["rating"] = frame["rating"].astype("float64")
    if frame["user_id"].min() < 1:
        raise ValueError("user ids start at 1; an id of 0 cannot be a database row")
    if frame["user_id"].max() >= user_id_offset:
        raise ValueError(
            f"database user id {int(frame['user_id'].max())} collides with the synthetic "
            f"offset {user_id_offset}; raise USER_ID_OFFSET"
        )

    # The database enforces one rating per (user, movie); a caller assembling frames by hand
    # might not, and a duplicate would silently double an entry in the matrix.
    before = len(frame)
    frame = frame.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    duplicates = before - len(frame)
    return frame.sort_values(["user_id", "movie_id"], kind="mergesort").reset_index(drop=True), (
        duplicates
    )


def _with_source_columns(
    user_map: pd.DataFrame, movie_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure both maps say where each row or column came from."""
    users = user_map.copy()
    movies = movie_map.copy()
    if "source" not in users.columns:
        users["source"] = SOURCE_ORIGINAL
    if "source" not in movies.columns:
        movies["source"] = SOURCE_ORIGINAL
    return users, movies


def strip_app_rows(
    matrix: sparse.csr_matrix,
    user_map: pd.DataFrame,
    movie_map: pd.DataFrame,
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame, int, int]:
    """Drop rows and columns a previous merge added, so a re-merge starts from the base data."""
    users, movies = _with_source_columns(user_map, movie_map)
    keep_users = users["source"].ne(SOURCE_APP).to_numpy()
    keep_movies = movies["source"].ne(SOURCE_APP).to_numpy()
    stripped_users = int((~keep_users).sum())
    stripped_movies = int((~keep_movies).sum())
    if stripped_users == 0 and stripped_movies == 0:
        return matrix, users, movies, 0, 0

    base_matrix = sparse.csr_matrix(matrix)[keep_users][:, keep_movies].tocsr()
    base_matrix.sort_indices()
    base_users = users.loc[keep_users].reset_index(drop=True)
    base_movies = movies.loc[keep_movies].reset_index(drop=True)
    base_users["user_index"] = np.arange(len(base_users), dtype="int32")
    base_movies["movie_index"] = np.arange(len(base_movies), dtype="int32")
    return base_matrix, base_users, base_movies, stripped_users, stripped_movies


def merge_app_ratings(
    matrix: sparse.csr_matrix,
    user_map: pd.DataFrame,
    movie_map: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    min_item_ratings: int = 20,
    prior_counts: Mapping[int, int] | None = None,
    user_id_offset: int = USER_ID_OFFSET,
) -> MergedArtifacts:
    """Fold application ratings into a training matrix, promoting films that reach the threshold.

    ``prior_counts`` maps a movie id to its MovieLens rating count, which is what lets a rarely
    rated catalogue film cross the threshold with a handful of app ratings. Films already in the
    matrix do not need it.
    """
    if min_item_ratings < 1:
        raise ValueError("min_item_ratings must be positive")
    matrix = sparse.csr_matrix(matrix)
    matrix.sort_indices()
    if len(movie_map) != matrix.shape[1]:
        raise ValueError("the movie map and the matrix have different column counts")
    if len(user_map) != matrix.shape[0]:
        raise ValueError("the user map and the matrix have different row counts")

    base_matrix, base_users, base_movies, stripped_users, stripped_movies = strip_app_rows(
        matrix, user_map, movie_map
    )
    frame, duplicates = _clean(ratings, user_id_offset)
    if frame.empty:
        return MergedArtifacts(
            matrix=base_matrix,
            user_map=base_users,
            movie_map=base_movies,
            stats=MergeStats(
                previous_users_stripped=stripped_users,
                previous_movies_stripped=stripped_movies,
            ),
        )

    known_movies = {int(movie_id) for movie_id in base_movies["movieId"]}
    app_counts = frame.groupby("movie_id", sort=True).size()
    promoted: list[int] = []
    below_threshold: list[int] = []
    for movie_id, count in app_counts.items():
        movie_id = int(movie_id)
        if movie_id in known_movies:
            continue
        prior = int((prior_counts or {}).get(movie_id, 0))
        if prior + int(count) >= min_item_ratings:
            promoted.append(movie_id)
        else:
            below_threshold.append(movie_id)
    promoted.sort()
    below_threshold.sort()

    mapped_ids = known_movies | set(promoted)
    frame = frame[frame["movie_id"].isin(mapped_ids)].reset_index(drop=True)
    if frame.empty:
        return MergedArtifacts(
            matrix=base_matrix,
            user_map=base_users,
            movie_map=base_movies,
            stats=MergeStats(
                duplicates=duplicates,
                promoted=tuple(promoted),
                below_threshold=tuple(below_threshold),
                previous_users_stripped=stripped_users,
                previous_movies_stripped=stripped_movies,
            ),
        )

    user_ids = sorted(int(user_id) for user_id in frame["user_id"].unique())
    n_base_users, n_base_movies = base_matrix.shape
    movie_index = {int(movie_id): index for index, movie_id in enumerate(base_movies["movieId"])}
    for position, movie_id in enumerate(promoted):
        movie_index[movie_id] = n_base_movies + position
    # The new block owns rows 0..n-1 of its own; global row numbers are assigned below when it is
    # stacked under the base matrix.
    local_rows = {user_id: position for position, user_id in enumerate(user_ids)}

    rows = frame["user_id"].map(local_rows).to_numpy(dtype="int64")
    columns = frame["movie_id"].map(movie_index).to_numpy(dtype="int64")
    values = frame["rating"].to_numpy(dtype="float32")
    app_block = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(user_ids), n_base_movies + len(promoted)),
        dtype="float32",
    )

    # Promoted films widen the matrix for every existing user too, so their rows must be padded
    # before the two blocks can be stacked.
    if promoted:
        filler = sparse.csr_matrix((n_base_users, len(promoted)), dtype="float32")
        base_matrix = sparse.hstack([base_matrix, filler], format="csr", dtype="float32")

    merged_matrix = sparse.vstack([base_matrix, app_block], format="csr", dtype="float32")
    merged_matrix.sort_indices()

    new_movies = pd.DataFrame(
        {
            "movieId": np.asarray(promoted, dtype="int64"),
            "movie_index": np.arange(n_base_movies, n_base_movies + len(promoted), dtype="int32"),
            "source": SOURCE_APP,
        }
    )
    new_users = pd.DataFrame(
        {
            "userId": np.asarray([user_id_offset + user_id for user_id in user_ids], dtype="int64"),
            "user_index": np.arange(n_base_users, n_base_users + len(user_ids), dtype="int32"),
            "source": SOURCE_APP,
        }
    )
    merged_movies = pd.concat([base_movies, new_movies], ignore_index=True)
    merged_users = pd.concat([base_users, new_users], ignore_index=True)

    return MergedArtifacts(
        matrix=merged_matrix,
        user_map=merged_users,
        movie_map=merged_movies,
        stats=MergeStats(
            ratings_merged=len(frame),
            users_added=len(user_ids),
            movies_added=len(promoted),
            duplicates=duplicates,
            promoted=tuple(promoted),
            below_threshold=tuple(below_threshold),
            previous_users_stripped=stripped_users,
            previous_movies_stripped=stripped_movies,
        ),
    )
