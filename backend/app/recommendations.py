"""Recommendation serving: precomputed neighbors, folded in per user at request time."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import get_settings
from recsys.item_cf import (
    ItemNeighbors,
    load_neighbors,
    popularity_recommendations,
    recommend_from_ratings,
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
            recs = recommend_from_ratings(neighbors, indices, values, limit=limit)
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


@lru_cache
def get_recommender() -> RecommendationService:
    return RecommendationService(get_settings().processed_dir)
