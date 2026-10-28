"""Recommendation serving: precomputed neighbors, folded in per user at request time.

Two algorithms are servable for a signed-in app user:

- ``item_cf`` reads the trained item neighbor table and folds the user's ratings into it.
- ``user_cf`` folds the user into the *training users* on the fly (they have no row in the
  matrix), so it needs the compact training matrix rather than a neighbor table.

The folding happens per request; nothing is retrained, and no similarity is recomputed for the
item-based path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from backend.app.config import get_settings
from recsys.baselines import item_counts_and_sums
from recsys.item_cf import (
    ItemNeighbors,
    load_neighbors,
    popularity_recommendations,
    recommend_from_ratings,
)
from recsys.user_cf import (
    FoldInModel,
    build_fold_in_model,
    fold_in,
    recommend_for_folded_user,
)


class ArtifactsUnavailable(RuntimeError):
    """Raised when the offline training artifacts have not been generated."""


@dataclass(frozen=True)
class ScoredItem:
    movie_id: int
    score: float
    predicted_rating: float
    source_movie_id: int | None
    source_rating: float | None
    popularity_count: int | None = None
    # (movie_id, weight) of the user's own ratings that pushed this pick, strongest first.
    contributions: tuple[tuple[int, float], ...] = ()
    # For user-based CF: how many of the user's nearest neighbours rated it.
    neighbors_rated: int | None = None


@dataclass(frozen=True)
class RecommendationBatch:
    algorithm: str
    cold_start: bool
    items: list[ScoredItem]


class RecommendationService:
    """Serves recommendations from fixed item neighbors.

    Neighbors are computed once by ``scripts/train_and_precompute.py``; a request only folds the
    signed-in user's ratings into them, so no similarity is ever computed at request time.
    """

    def __init__(self, processed_dir: str | Path):
        self.processed_dir = Path(processed_dir)
        self._neighbors: ItemNeighbors | None = None
        self._movie_ids: np.ndarray | None = None
        self._id_to_index: dict[int, int] = {}
        self._lock = threading.Lock()

    def _artifacts(self) -> tuple[ItemNeighbors, np.ndarray, dict[int, int]]:
        if self._neighbors is not None and self._movie_ids is not None:
            return self._neighbors, self._movie_ids, self._id_to_index
        with self._lock:
            if self._neighbors is None or self._movie_ids is None:
                neighbors_path = self.processed_dir / "item_neighbors.npz"
                map_path = self.processed_dir / "movie_id_map.parquet"
                try:
                    neighbors = load_neighbors(neighbors_path)
                    mapping = pd.read_parquet(map_path, columns=["movieId", "movie_index"])
                except FileNotFoundError as exc:
                    raise ArtifactsUnavailable(
                        "Recommendations are not trained yet. Run "
                        "`uv run python scripts/train_and_precompute.py`."
                    ) from exc
                mapping = mapping.sort_values("movie_index")
                movie_ids = mapping["movieId"].to_numpy(dtype="int64")
                if movie_ids.size != neighbors.counts.size:
                    raise ArtifactsUnavailable(
                        "Item neighbors and the movie id map disagree; retrain the artifacts."
                    )
                self._movie_ids = movie_ids
                self._id_to_index = {
                    int(movie_id): index for index, movie_id in enumerate(movie_ids)
                }
                self._neighbors = neighbors
        return self._neighbors, self._movie_ids, self._id_to_index

    def recommend(
        self,
        ratings: list[tuple[int, float]],
        *,
        limit: int = 10,
        explain: int = 3,
    ) -> RecommendationBatch:
        neighbors, movie_ids, id_to_index = self._artifacts()

        known = [
            (movie_id, float(rating)) for movie_id, rating in ratings if movie_id in id_to_index
        ]
        rated_ids = {int(movie_id) for movie_id, _ in ratings}

        if known:
            indices = np.asarray([id_to_index[movie_id] for movie_id, _ in known], dtype="int64")
            values = np.asarray([rating for _, rating in known], dtype="float64")
            ratings_by_index = {
                index: rating for index, (_, rating) in zip(indices, known, strict=True)
            }
            recs = recommend_from_ratings(neighbors, indices, values, limit=limit, explain=explain)
            if recs:
                items = []
                for rec in recs:
                    source_movie_id = (
                        int(movie_ids[rec.source_index]) if rec.source_index is not None else None
                    )
                    items.append(
                        ScoredItem(
                            movie_id=int(movie_ids[rec.item_index]),
                            score=rec.score,
                            predicted_rating=rec.predicted_rating,
                            source_movie_id=source_movie_id,
                            source_rating=(
                                ratings_by_index[rec.source_index]
                                if rec.source_index is not None
                                else None
                            ),
                            contributions=tuple(
                                (int(movie_ids[entry.item_index]), float(entry.strength))
                                for entry in rec.contributions
                            ),
                        )
                    )
                return RecommendationBatch(algorithm="item_cf", cold_start=False, items=items)

        # No ratings, a single rating, or a user whose ratings are all equal their mean: nothing
        # survives mean-centering, so the honest answer is the popularity baseline. It is not
        # personalized, hence ``cold_start`` is always true on this path.
        exclude = {id_to_index[movie_id] for movie_id in rated_ids if movie_id in id_to_index}
        candidates = popularity_recommendations(
            neighbors, limit=limit + len(rated_ids), exclude=exclude
        )
        items = []
        for rec in candidates:
            served = ScoredItem(
                movie_id=int(movie_ids[rec.item_index]),
                score=rec.score,
                predicted_rating=rec.predicted_rating,
                source_movie_id=None,
                source_rating=None,
                popularity_count=int(neighbors.counts[rec.item_index]),
            )
            if served.movie_id in rated_ids:
                continue
            items.append(served)
            if len(items) >= limit:
                break
        return RecommendationBatch(algorithm="popularity", cold_start=True, items=items)


class UserCfService:
    """User-based CF for a signed-in user, folded into the training users per request.

    The offline neighbour table cannot serve app users (they have no training row), so this loads
    the compact training matrix instead and computes the nearest training users from the items the
    request's user actually rated.
    """

    def __init__(self, processed_dir: str | Path):
        self.processed_dir = Path(processed_dir)
        self._model: FoldInModel | None = None
        self._movie_ids: np.ndarray | None = None
        self._id_to_index: dict[int, int] = {}
        self._counts: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._lock = threading.Lock()

    def _artifacts(
        self,
    ) -> tuple[FoldInModel, np.ndarray, dict[int, int], np.ndarray, np.ndarray]:
        if self._model is not None and self._movie_ids is not None:
            return (
                self._model,
                self._movie_ids,
                self._id_to_index,
                self._counts,
                self._means,
            )
        with self._lock:
            if self._model is None or self._movie_ids is None:
                matrix_path = self.processed_dir / "train_matrix.npz"
                map_path = self.processed_dir / "movie_id_map.parquet"
                try:
                    matrix = sparse.load_npz(matrix_path).tocsr()
                    mapping = pd.read_parquet(map_path, columns=["movieId", "movie_index"])
                except FileNotFoundError as exc:
                    raise ArtifactsUnavailable(
                        "Recommendations are not trained yet. Run "
                        "`uv run python scripts/prepare_data.py`."
                    ) from exc
                mapping = mapping.sort_values("movie_index")
                movie_ids = mapping["movieId"].to_numpy(dtype="int64")
                if movie_ids.size != matrix.shape[1]:
                    raise ArtifactsUnavailable(
                        "The training matrix and the movie id map disagree; re-run "
                        "scripts/prepare_data.py."
                    )
                counts, sums = item_counts_and_sums(matrix)
                means = np.zeros_like(sums, dtype="float64")
                np.divide(sums, counts, out=means, where=counts > 0)
                self._model = build_fold_in_model(matrix)
                self._movie_ids = movie_ids
                self._id_to_index = {
                    int(movie_id): index for index, movie_id in enumerate(movie_ids)
                }
                self._counts = counts
                self._means = means
        return self._model, self._movie_ids, self._id_to_index, self._counts, self._means

    def recommend(
        self,
        ratings: list[tuple[int, float]],
        *,
        limit: int = 10,
    ) -> RecommendationBatch:
        model, movie_ids, id_to_index, counts, means = self._artifacts()
        known = [
            (movie_id, float(rating)) for movie_id, rating in ratings if movie_id in id_to_index
        ]
        rated_ids = {int(movie_id) for movie_id, _ in ratings}

        if known:
            indices = np.asarray([id_to_index[movie_id] for movie_id, _ in known], dtype="int64")
            values = np.asarray([rating for _, rating in known], dtype="float64")
            folded = fold_in(model, indices, values)
            recs = recommend_for_folded_user(model, folded, limit=limit)
            if recs:
                return RecommendationBatch(
                    algorithm="user_cf",
                    cold_start=False,
                    items=[
                        ScoredItem(
                            movie_id=int(movie_ids[rec.item_index]),
                            score=rec.score,
                            predicted_rating=rec.predicted_rating,
                            source_movie_id=None,
                            source_rating=None,
                            neighbors_rated=int(rec.neighbors_rated),
                        )
                        for rec in recs
                    ],
                )

        # A user with no variance in their ratings (or none at all) has nothing to correlate with.
        exclude = {id_to_index[movie_id] for movie_id in rated_ids if movie_id in id_to_index}
        order = np.lexsort((np.arange(counts.size), -counts))
        items: list[ScoredItem] = []
        for index in order:
            if index in exclude or counts[index] == 0:
                continue
            movie_id = int(movie_ids[index])
            if movie_id in rated_ids:
                continue
            items.append(
                ScoredItem(
                    movie_id=movie_id,
                    score=float(counts[index]),
                    predicted_rating=float(np.clip(means[index], 0.5, 5.0)),
                    source_movie_id=None,
                    source_rating=None,
                    popularity_count=int(counts[index]),
                )
            )
            if len(items) >= limit:
                break
        return RecommendationBatch(algorithm="popularity", cold_start=True, items=items)


@lru_cache
def get_recommender() -> RecommendationService:
    return RecommendationService(get_settings().processed_dir)


@lru_cache
def get_user_cf_recommender() -> UserCfService:
    return UserCfService(get_settings().processed_dir)
