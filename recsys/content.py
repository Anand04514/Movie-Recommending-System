"""Content-based features and recommendations.

Every catalog film gets a sparse feature row built from what we know about it without any
ratings: its genres, its release decade and (when the tag vocabulary has been extracted) its
MovieLens tags. Feature rows are L2-normalised, so a dot product is a cosine similarity and a
user's profile is a weighted average of the films they liked.

This model is what gives a film outside the collaborative model universe a chance: a 2023 release
with four ratings still has genres and a year, and can still be recommended to someone who loves
that kind of film. It is also the second half of the hybrid score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.item_cf import Contribution, Recommendation
from recsys.titles import split_genres

DEFAULT_GENRE_WEIGHT = 1.0
DEFAULT_DECADE_WEIGHT = 0.55
DEFAULT_TAG_WEIGHT = 0.35
# Cosine similarity is not a rating, so it is mapped into rating space with a conservative slope
# before it is shown as "Predicted 4.1 stars".
DEFAULT_PREDICTION_SLOPE = 1.4
MIN_YEAR = 1880
MAX_YEAR = 2100


@dataclass(frozen=True)
class ContentFeatures:
    """One L2-normalised feature row per catalog movie."""

    movie_ids: np.ndarray  # (n_movies,) int64, row order
    matrix: sparse.csr_matrix  # (n_movies, n_features) float32
    feature_names: tuple[str, ...]
    index: dict[int, int]  # movie id -> row

    @property
    def n_movies(self) -> int:
        return int(self.movie_ids.size)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def row_for(self, movie_id: int) -> int | None:
        return self.index.get(int(movie_id))

    def rows_for(self, movie_ids: np.ndarray) -> sparse.csr_matrix:
        """Feature rows aligned to ``movie_ids``; raises if one is missing from the catalog."""
        wanted = np.asarray(movie_ids, dtype="int64")
        missing = [int(movie_id) for movie_id in wanted if int(movie_id) not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} movie(s) have no content features: {missing[:5]}")
        rows = np.asarray([self.index[int(movie_id)] for movie_id in wanted], dtype="int64")
        return self.matrix[rows]


def _decade_label(year: int | None) -> str | None:
    if year is None or not MIN_YEAR <= int(year) <= MAX_YEAR:
        return None
    return f"decade:{int(year) // 10 * 10}"


def build_content_features(
    catalog: pd.DataFrame,
    *,
    tags: pd.DataFrame | None = None,
    genre_weight: float = DEFAULT_GENRE_WEIGHT,
    decade_weight: float = DEFAULT_DECADE_WEIGHT,
    tag_weight: float = DEFAULT_TAG_WEIGHT,
) -> ContentFeatures:
    """Build the feature matrix from ``movie_id``, ``genres`` and ``release_year`` columns.

    ``tags`` is optional long-form data with ``movie_id``, ``tag`` and ``count`` columns. Tags are
    weighted by ``log1p(count)`` so one obsessive user tagging a film does not outweigh the genre.
    """
    required = {"movie_id", "genres"}
    missing_columns = required - set(catalog.columns)
    if missing_columns:
        raise ValueError(f"catalog is missing column(s): {sorted(missing_columns)}")
    if catalog["movie_id"].duplicated().any():
        raise ValueError("catalog contains duplicate movie_id values")

    movie_ids = catalog["movie_id"].to_numpy(dtype="int64")
    feature_index: dict[str, int] = {}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def add(row: int, name: str, weight: float) -> None:
        if weight <= 0:
            return
        column = feature_index.setdefault(name, len(feature_index))
        rows.append(row)
        columns.append(column)
        values.append(float(weight))

    years = (
        catalog["release_year"].to_numpy(dtype="float64")
        if "release_year" in catalog.columns
        else np.full(len(catalog), np.nan)
    )
    for row, (genres, year) in enumerate(
        zip(catalog["genres"].fillna("").astype(str), years, strict=True)
    ):
        for genre in split_genres(genres):
            if genre == "(no genres listed)":
                continue
            add(row, f"genre:{genre}", genre_weight)
        if not np.isnan(year):
            label = _decade_label(int(year))
            if label:
                add(row, label, decade_weight)

    if tags is not None and not tags.empty:
        missing_tag_columns = {"movie_id", "tag", "count"} - set(tags.columns)
        if missing_tag_columns:
            raise ValueError(f"tags is missing column(s): {sorted(missing_tag_columns)}")
        position = {int(movie_id): row for row, movie_id in enumerate(movie_ids)}
        cleaned = tags.copy()
        cleaned["movie_id"] = cleaned["movie_id"].astype("int64")
        cleaned["count"] = pd.to_numeric(cleaned["count"], errors="coerce").fillna(0.0)
        cleaned = cleaned[cleaned["movie_id"].isin(position) & (cleaned["count"] > 0)]
        for movie_id, tag, count in zip(
            cleaned["movie_id"], cleaned["tag"].astype(str), cleaned["count"], strict=True
        ):
            label = tag.strip().lower()
            if not label:
                continue
            add(
                position[int(movie_id)],
                f"tag:{label}",
                tag_weight * float(np.log1p(count)),
            )

    n_movies = len(movie_ids)
    n_features = max(1, len(feature_index))
    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype="float32"), (np.asarray(rows), np.asarray(columns))),
        shape=(n_movies, n_features),
        dtype="float32",
    )
    # Duplicate (movie, feature) pairs can only come from repeated genres or tags; summing them
    # is correct, then each row is unit length so cosine is a plain dot product.
    matrix.sum_duplicates()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms, dtype="float32")
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    matrix = (sparse.diags(inverse) @ matrix).tocsr()
    matrix.sort_indices()

    names = tuple(
        name for name, _ in sorted(feature_index.items(), key=lambda item: item[1])
    )
    return ContentFeatures(
        movie_ids=movie_ids,
        matrix=matrix,
        feature_names=names,
        index={int(movie_id): row for row, movie_id in enumerate(movie_ids)},
    )


def extract_tag_features(
    tags_path: Path,
    *,
    chunksize: int = 1_000_000,
    top_tags: int = 300,
    min_count: int = 5,
) -> pd.DataFrame:
    """Aggregate MovieLens tags into ``movie_id``/``tag``/``count`` rows.

    ``tags.csv`` is read in chunks and never held whole. Only the ``top_tags`` most frequent tags
    survive, which keeps the feature matrix small and drops one-off typos.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if top_tags < 1:
        raise ValueError("top_tags must be positive")
    tags_path = Path(tags_path)
    if not tags_path.is_file():
        raise FileNotFoundError(f"tags file not found: {tags_path}")

    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        tags_path, usecols=["movieId", "tag"], dtype={"movieId": "int64", "tag": "string"},
        chunksize=chunksize,
    ):
        cleaned = chunk.dropna(subset=["tag"]).copy()
        cleaned["tag"] = cleaned["tag"].str.strip().str.lower()
        cleaned = cleaned[cleaned["tag"] != ""]
        if cleaned.empty:
            continue
        pieces.append(
            cleaned.groupby(["movieId", "tag"], observed=True).size().rename("count").reset_index()
        )
    if not pieces:
        raise ValueError("tags.csv contained no usable tags")

    combined = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["movieId", "tag"], observed=True)["count"]
        .sum()
        .reset_index()
    )
    combined = combined[combined["count"] >= min_count]
    totals = combined.groupby("tag", observed=True)["count"].sum()
    keep = totals.sort_values(ascending=False).head(top_tags).index
    combined = combined[combined["tag"].isin(keep)]
    return combined.rename(columns={"movieId": "movie_id"}).reset_index(drop=True)


@dataclass(frozen=True)
class ContentProfile:
    """A user's taste direction in feature space, plus the weights that built it."""

    vector: np.ndarray  # (n_features,) float64
    user_mean: float
    rated_indexes: np.ndarray
    rated_values: np.ndarray
    weights: np.ndarray  # (n_rated,) float64, non-negative

    @property
    def has_signal(self) -> bool:
        return bool(np.any(self.vector))


def build_profile(
    features: ContentFeatures,
    rated_indices: np.ndarray,
    rated_values: np.ndarray,
) -> ContentProfile:
    """Average the features of liked films, weighted by how far above the mean each rating was.

    A user who rates everything the same has no content signal; the profile is empty and the
    caller falls back rather than inventing taste out of uniformity.
    """
    rated_indices = np.asarray(rated_indices, dtype="int64")
    rated_values = np.asarray(rated_values, dtype="float64")
    if rated_indices.shape != rated_values.shape:
        raise ValueError("rated_indices and rated_values must have the same shape")
    if rated_indices.ndim != 1:
        raise ValueError("rated_indices must be one-dimensional")
    if rated_values.size and (rated_indices.min() < 0 or rated_indices.max() >= features.n_movies):
        raise IndexError("a rated row index is outside the feature matrix")

    user_mean = float(rated_values.mean()) if rated_values.size else 0.0
    weights = np.zeros(rated_values.size, dtype="float64")
    if rated_values.size:
        # Above the user's own average is the strongest signal, but a single rating (or a user who
        # rates everything the same) has no average to rise above. Fall back to "above neutral"
        # so one onboarding rating still produces a taste profile. Everything at or below three
        # stars is not a preference we should build a profile from.
        weights = np.clip(rated_values - user_mean, 0.0, None)
        if weights.sum() <= 0:
            weights = np.clip(rated_values - 3.0, 0.0, None)

    vector = np.zeros(features.n_features, dtype="float64")
    if rated_indices.size and weights.sum() > 0:
        vector = np.asarray(
            weights @ features.matrix[rated_indices], dtype="float64"
        ).ravel()
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
    return ContentProfile(
        vector=vector,
        user_mean=user_mean,
        rated_indexes=rated_indices,
        rated_values=rated_values,
        weights=weights,
    )


def score_items(features: ContentFeatures, profile: ContentProfile) -> np.ndarray:
    """Cosine similarity of every catalog film to the profile."""
    if not profile.has_signal:
        return np.zeros(features.n_movies, dtype="float64")
    return np.asarray(features.matrix @ profile.vector, dtype="float64").ravel()


def recommend_content(
    features: ContentFeatures,
    rated_indices: np.ndarray,
    rated_values: np.ndarray,
    *,
    limit: int = 10,
    exclude: set[int] | None = None,
    explain: int = 3,
    slope: float = DEFAULT_PREDICTION_SLOPE,
) -> list[Recommendation]:
    """Rank the whole catalog for a user, including films outside the collaborative universe.

    ``item_index`` here is a *content row*, not a model matrix index; the caller maps it back to a
    MovieLens id through ``features.movie_ids``.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if explain < 0:
        raise ValueError("explain must be non-negative")
    if slope <= 0:
        raise ValueError("slope must be positive")

    profile = build_profile(features, rated_indices, rated_values)
    if not profile.has_signal:
        return []

    scores = score_items(features, profile)
    predicted = np.clip(profile.user_mean + slope * scores, 0.5, 5.0)

    blocked = {int(index) for index in rated_indices}
    if exclude:
        blocked |= {int(index) for index in exclude}
    if blocked:
        allowed = np.ones(features.n_movies, dtype=bool)
        blocked_array = np.fromiter(blocked, dtype="int64", count=len(blocked))
        blocked_array = blocked_array[
            (blocked_array >= 0) & (blocked_array < features.n_movies)
        ]
        allowed[blocked_array] = False
        scores = np.where(allowed, scores, -np.inf)

    candidates = np.flatnonzero(scores > 0)
    if candidates.size == 0:
        return []
    take = min(limit, candidates.size)
    top = np.argpartition(-scores[candidates], take - 1)[:take]
    chosen = candidates[top[np.lexsort((candidates[top], -scores[candidates][top]))]]

    contributions: list[tuple[Contribution, ...]] = [() for _ in chosen]
    if explain and profile.rated_indexes.size:
        positive = profile.weights
        ranked = np.argsort(-positive, kind="stable")
        cited = profile.rated_indexes[ranked[: max(explain, 1) * 4]]
        cited_weights = positive[ranked[: max(explain, 1) * 4]]
        similarities = (features.matrix[chosen] @ features.matrix[cited].T).toarray()
        strengths = np.asarray(similarities, dtype="float64") * cited_weights[None, :]
        for row in range(chosen.size):
            order = np.argsort(-strengths[row], kind="stable")[:explain]
            contributions[row] = tuple(
                Contribution(item_index=int(cited[column]), strength=float(strengths[row, column]))
                for column in order
                if strengths[row, column] > 0
            )

    results: list[Recommendation] = []
    for row, index in enumerate(chosen):
        sources = contributions[row]
        results.append(
            Recommendation(
                item_index=int(index),
                score=float(scores[index]),
                predicted_rating=float(predicted[index]),
                source_index=sources[0].item_index if sources else None,
                contributions=sources,
            )
        )
    return results


def save_features(features: ContentFeatures, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        movie_ids=features.movie_ids.astype("int64"),
        data=features.matrix.data.astype("float32"),
        indices=features.matrix.indices.astype("int32"),
        indptr=features.matrix.indptr.astype("int64"),
        shape=np.asarray(features.matrix.shape, dtype="int64"),
        feature_names=np.asarray(features.feature_names, dtype="str"),
    )


def load_features(path: Path) -> ContentFeatures:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"content features not found at {path}; run scripts/train_and_precompute.py first"
        )
    with np.load(path) as data:
        shape = tuple(int(value) for value in data["shape"])
        matrix = sparse.csr_matrix(
            (
                data["data"].astype("float32"),
                data["indices"].astype("int32"),
                data["indptr"].astype("int64"),
            ),
            shape=shape,
        )
        movie_ids = data["movie_ids"].astype("int64")
        names = tuple(str(name) for name in data["feature_names"])
    return ContentFeatures(
        movie_ids=movie_ids,
        matrix=matrix,
        feature_names=names,
        index={int(movie_id): row for row, movie_id in enumerate(movie_ids)},
    )
