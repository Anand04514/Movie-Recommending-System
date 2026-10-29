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
from recsys.content import (
    ContentFeatures,
    load_features,
    recommend_content,
)
from recsys.hybrid import (
    load_collaborative_base,
    load_weights,
)
from recsys.item_cf import (
    ItemNeighbors,
    load_neighbors,
    popularity_recommendations,
    recommend_from_ratings,
)
from recsys.svd import (
    factor_gram,
    load_model,
)
from recsys.svd import (
    fold_in as fold_in_svd,
)
from recsys.svd import (
    recommend_for_folded_user as recommend_for_svd_user,
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


def _merge_contributions(
    *groups: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    """Union of several evidence lists, strongest weight per movie."""
    best: dict[int, float] = {}
    for group in groups:
        for movie_id, strength in group:
            if strength > best.get(movie_id, float("-inf")):
                best[movie_id] = float(strength)
    return tuple(sorted(best.items(), key=lambda pair: -pair[1]))


class ContentService:
    """Genres, decades and tags scored over the whole catalog, rare films included.

    The feature artifact covers every catalog movie, not just the model universe, so a film with
    four ratings can still be recommended to someone whose taste it matches.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        fallback: RecommendationService | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.fallback = fallback
        self._features: ContentFeatures | None = None
        self._lock = threading.Lock()

    def _artifacts(self) -> ContentFeatures:
        if self._features is not None:
            return self._features
        with self._lock:
            if self._features is None:
                try:
                    self._features = load_features(
                        self.processed_dir / "content_features.npz"
                    )
                except FileNotFoundError as exc:
                    raise ArtifactsUnavailable(
                        "Content recommendations are not trained yet. Run "
                        "`uv run python scripts/train_and_precompute.py`."
                    ) from exc
        return self._features

    def recommend(
        self,
        ratings: list[tuple[int, float]],
        *,
        limit: int = 10,
        explain: int = 3,
    ) -> RecommendationBatch:
        features = self._artifacts()
        ratings_by_movie = {int(movie_id): float(rating) for movie_id, rating in ratings}
        known = [
            (movie_id, rating)
            for movie_id, rating in ratings_by_movie.items()
            if features.row_for(movie_id) is not None
        ]

        items: list[ScoredItem] = []
        if known:
            indices = np.asarray(
                [features.index[movie_id] for movie_id, _ in known], dtype="int64"
            )
            values = np.asarray([rating for _, rating in known], dtype="float64")
            for rec in recommend_content(
                features, indices, values, limit=limit, explain=explain
            ):
                source_id = (
                    int(features.movie_ids[rec.source_index])
                    if rec.source_index is not None
                    else None
                )
                items.append(
                    ScoredItem(
                        movie_id=int(features.movie_ids[rec.item_index]),
                        score=rec.score,
                        predicted_rating=rec.predicted_rating,
                        source_movie_id=source_id,
                        source_rating=(
                            ratings_by_movie.get(source_id)
                            if source_id is not None
                            else None
                        ),
                        contributions=tuple(
                            (int(features.movie_ids[entry.item_index]), float(entry.strength))
                            for entry in rec.contributions
                        ),
                    )
                )

        if items:
            return RecommendationBatch(algorithm="content", cold_start=False, items=items)
        if self.fallback is not None:
            return self.fallback.recommend(ratings, limit=limit)
        return RecommendationBatch(algorithm="content", cold_start=True, items=[])


class SvdService:
    """Latent factors with ridge fold-in for the signed-in user.

    The model's user factors only cover training users; app users are folded in from their own
    ratings, which is why recommendations update instantly without retraining.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        fallback: RecommendationService | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.fallback = fallback
        self._model = None
        self._movie_ids: np.ndarray | None = None
        self._id_to_index: dict[int, int] = {}
        self._gram: np.ndarray | None = None
        self._lock = threading.Lock()

    def _artifacts(self):
        if self._model is not None and self._movie_ids is not None and self._gram is not None:
            return self._model, self._movie_ids, self._id_to_index, self._gram
        with self._lock:
            if self._model is None or self._movie_ids is None or self._gram is None:
                try:
                    model = load_model(self.processed_dir / "svd_model.npz")
                    mapping = pd.read_parquet(
                        self.processed_dir / "movie_id_map.parquet",
                        columns=["movieId", "movie_index"],
                    )
                except FileNotFoundError as exc:
                    raise ArtifactsUnavailable(
                        "SVD recommendations are not trained yet. Run "
                        "`uv run python scripts/train_and_precompute.py`."
                    ) from exc
                mapping = mapping.sort_values("movie_index")
                movie_ids = mapping["movieId"].to_numpy(dtype="int64")
                if movie_ids.size != model.n_items:
                    raise ArtifactsUnavailable(
                        "The SVD factors and the movie id map disagree; retrain the artifacts."
                    )
                self._model = model
                self._movie_ids = movie_ids
                self._id_to_index = {
                    int(movie_id): index for index, movie_id in enumerate(movie_ids)
                }
                self._gram = factor_gram(model)
        return self._model, self._movie_ids, self._id_to_index, self._gram

    def recommend(
        self,
        ratings: list[tuple[int, float]],
        *,
        limit: int = 10,
        explain: int = 3,
    ) -> RecommendationBatch:
        model, movie_ids, id_to_index, gram = self._artifacts()
        ratings_by_movie = {int(movie_id): float(rating) for movie_id, rating in ratings}
        known = [
            (movie_id, rating)
            for movie_id, rating in ratings_by_movie.items()
            if movie_id in id_to_index
        ]

        if known:
            indices = np.asarray(
                [id_to_index[movie_id] for movie_id, _ in known], dtype="int64"
            )
            values = np.asarray([rating for _, rating in known], dtype="float64")
            folded = fold_in_svd(model, indices, values, gram=gram)
            recs = recommend_for_svd_user(
                model, folded, limit=limit, explain=explain
            )
            if recs:
                return RecommendationBatch(
                    algorithm="svd",
                    cold_start=False,
                    items=[
                        ScoredItem(
                            movie_id=int(movie_ids[rec.item_index]),
                            score=rec.score,
                            predicted_rating=rec.predicted_rating,
                            source_movie_id=(
                                int(movie_ids[rec.source_index])
                                if rec.source_index is not None
                                else None
                            ),
                            source_rating=(
                                ratings_by_movie.get(int(movie_ids[rec.source_index]))
                                if rec.source_index is not None
                                else None
                            ),
                            contributions=tuple(
                                (
                                    int(movie_ids[entry.item_index]),
                                    float(entry.strength),
                                )
                                for entry in rec.contributions
                            ),
                        )
                        for rec in recs
                    ],
                )

        if self.fallback is not None:
            return self.fallback.recommend(ratings, limit=limit)
        return RecommendationBatch(algorithm="svd", cold_start=True, items=[])


class HybridService:
    """Tuned blend of a collaborative model and the content model.

    Both candidate lists are normalised on their own scale before the weighted blend (see
    ``recsys.hybrid``), and the collaborative base and its weight come from the validation-tuned
    artifact. With no artifact the defaults are user-CF at 0.65, which still beats the baselines.
    """

    def __init__(
        self,
        item_service: RecommendationService,
        user_service: UserCfService,
        content_service: ContentService,
        *,
        processed_dir: str | Path | None = None,
        base: str | None = None,
        weight: float | None = None,
        explain: int = 3,
    ) -> None:
        self.item_service = item_service
        self.user_service = user_service
        self.content_service = content_service
        self.explain = explain
        directory = Path(processed_dir) if processed_dir else Path(get_settings().processed_dir)
        settings_path = directory / "hybrid_weights.json"
        self.base = base or load_collaborative_base(settings_path)
        if self.base not in {"item_cf", "user_cf"}:
            raise ValueError("base must be 'item_cf' or 'user_cf'")
        self.weight = float(load_weights(settings_path) if weight is None else weight)
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")

    def recommend(
        self,
        ratings: list[tuple[int, float]],
        *,
        limit: int = 10,
    ) -> RecommendationBatch:
        if not ratings:
            return self.item_service.recommend(ratings, limit=limit)

        pool = min(max(limit * 6, limit), 300)
        collaborative = self.user_service if self.base == "user_cf" else self.item_service
        base_batch = collaborative.recommend(ratings, limit=pool)
        if base_batch.cold_start or base_batch.algorithm == "popularity":
            return base_batch

        try:
            content_batch = self.content_service.recommend(
                ratings, limit=pool, explain=self.explain
            )
        except ArtifactsUnavailable:
            # The hybrid is usable without content features; serving item/user CF is better than
            # failing the page because one optional artifact is missing.
            return base_batch
        if not content_batch.items or content_batch.algorithm == "popularity":
            return base_batch

        ratings_by_movie = {int(movie_id): float(rating) for movie_id, rating in ratings}
        base_scale = max((item.score for item in base_batch.items), default=0.0) or 1.0
        content_scale = max((item.score for item in content_batch.items), default=0.0) or 1.0

        merged: dict[int, dict[str, object]] = {}
        for item in base_batch.items:
            merged[item.movie_id] = {
                "base_score": max(item.score, 0.0) / base_scale,
                "content_score": 0.0,
                "base_prediction": item.predicted_rating,
                "content_prediction": None,
                "neighbors_rated": item.neighbors_rated,
                "contributions": dict(item.contributions),
            }
        for item in content_batch.items:
            entry = merged.setdefault(
                item.movie_id,
                {
                    "base_score": 0.0,
                    "content_score": 0.0,
                    "base_prediction": None,
                    "content_prediction": None,
                    "neighbors_rated": None,
                    "contributions": {},
                },
            )
            entry["content_score"] = max(item.score, 0.0) / content_scale
            entry["content_prediction"] = item.predicted_rating
            contributions = entry["contributions"]
            assert isinstance(contributions, dict)
            for movie_id, strength in item.contributions:
                previous = contributions.get(movie_id)
                if previous is None or strength > previous:
                    contributions[movie_id] = strength

        ranked: list[tuple[float, int, dict[str, object]]] = []
        for movie_id, entry in merged.items():
            combined = self.weight * float(entry["base_score"]) + (
                1.0 - self.weight
            ) * float(entry["content_score"])
            if combined <= 0.0:
                continue
            base_prediction = entry["base_prediction"]
            content_prediction = entry["content_prediction"]
            if base_prediction is not None and content_prediction is not None:
                predicted = (
                    self.weight * float(base_prediction)
                    + (1.0 - self.weight) * float(content_prediction)
                )
            elif base_prediction is not None:
                predicted = float(base_prediction)
            else:
                predicted = float(content_prediction)
            ranked.append((combined, movie_id, {**entry, "predicted_rating": predicted}))

        ranked.sort(key=lambda row: (-row[0], row[1]))
        items: list[ScoredItem] = []
        for combined, movie_id, entry in ranked[:limit]:
            contributions = entry["contributions"]
            assert isinstance(contributions, dict)
            ordered = _merge_contributions(
                tuple(
                    sorted(
                        ((int(mid), float(strength)) for mid, strength in contributions.items()),
                        key=lambda pair: -pair[1],
                    )
                )
            )[: self.explain]
            source = ordered[0][0] if ordered else None
            items.append(
                ScoredItem(
                    movie_id=movie_id,
                    score=combined,
                    predicted_rating=float(np.clip(entry["predicted_rating"], 0.5, 5.0)),
                    source_movie_id=source,
                    source_rating=ratings_by_movie.get(source) if source is not None else None,
                    popularity_count=None,
                    contributions=ordered,
                    neighbors_rated=entry["neighbors_rated"],
                )
            )
        return RecommendationBatch(algorithm="hybrid", cold_start=False, items=items)


@lru_cache
def get_recommender() -> RecommendationService:
    return RecommendationService(get_settings().processed_dir)


@lru_cache
def get_user_cf_recommender() -> UserCfService:
    return UserCfService(get_settings().processed_dir)


@lru_cache
def get_content_recommender() -> ContentService:
    return ContentService(get_settings().processed_dir, fallback=get_recommender())


@lru_cache
def get_svd_recommender() -> SvdService:
    return SvdService(get_settings().processed_dir, fallback=get_recommender())


@lru_cache
def get_hybrid_recommender() -> HybridService:
    return HybridService(
        get_recommender(),
        get_user_cf_recommender(),
        get_content_recommender(),
    )
