"""Content cold start for films the collaborative models have never seen (5.11).

A film ingested from TMDB has no ratings at all, so every neighbourhood model is blind to it. It
does, however, have genres, a release year and keywords, which is exactly what the content model
reads. This service scores those films against the same taste profile the content model builds for
a signed-in user, using the feature vocabulary frozen at training time.

Two consequences worth stating plainly:

- A cold film can be recommended on day one, but only through content. It cannot enter the trained
  model until a retrain gives it enough ratings (``in_model`` stays false until then).
- If the user has no positive content signal, the row is an honest "new in the catalog" list rather
  than a pretend personalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy import sparse

from backend.app.freshness import is_new_release
from backend.app.models import Movie
from backend.app.recommendations import ArtifactsUnavailable, ContentService
from recsys.content import (
    DEFAULT_PREDICTION_SLOPE,
    ContentProfile,
    build_profile,
    feature_row,
    score_rows,
)
from recsys.titles import split_genres

# How many catalogue rows the row considers before ranking; a page is 6-8 films.
CANDIDATE_LIMIT = 200


@dataclass(frozen=True)
class ColdStartPick:
    movie: Movie
    similarity: float
    predicted_rating: float
    reason: str
    is_new: bool


def _keywords(movie: Movie) -> list[str]:
    raw = (movie.tmdb_keywords or "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()] if raw else []


def build_rows(features, movies: list[Movie]) -> sparse.csr_matrix:
    """Stack one feature row per film, in the order given."""
    rows = [
        feature_row(
            features,
            genres=split_genres(movie.genres or ""),
            release_year=movie.release_year,
            tags=_keywords(movie),
        )
        for movie in movies
    ]
    if not rows:
        return sparse.csr_matrix((0, features.n_features), dtype="float32")
    return sparse.vstack(rows).tocsr()


class NewReleaseService:
    """Rank TMDB-era films against a live user's content profile."""

    def __init__(self, processed_dir: str | Path, content: ContentService) -> None:
        self.processed_dir = Path(processed_dir)
        self.content = content

    def _profile(self, ratings: dict[int, float]) -> tuple[ContentProfile | None, float]:
        features = self.content.features()
        known = [
            (movie_id, float(rating))
            for movie_id, rating in ratings.items()
            if features.row_for(movie_id) is not None
        ]
        if not known:
            return None, 0.0
        indices = np.asarray([features.index[movie_id] for movie_id, _ in known], dtype="int64")
        values = np.asarray([rating for _, rating in known], dtype="float64")
        profile = build_profile(features, indices, values)
        return (profile if profile.has_signal else None), profile.user_mean

    def picks(
        self,
        ratings: dict[int, float],
        movies: list[Movie],
        *,
        favourite_genres: set[str],
        limit: int = 6,
        today: date | None = None,
    ) -> list[ColdStartPick]:
        if limit < 1 or not movies:
            return []
        today = today or date.today()

        try:
            features = self.content.features()
        except ArtifactsUnavailable:
            # Without the vocabulary there is no cold-start path; the caller shows nothing rather
            # than a list ordered by nothing in particular.
            return []

        profile, user_mean = self._profile(ratings)
        if profile is None:
            ordered = sorted(
                movies,
                key=lambda movie: (movie.release_year or 0, movie.rating_count),
                reverse=True,
            )[:limit]
            return [
                ColdStartPick(
                    movie=movie,
                    similarity=0.0,
                    predicted_rating=3.0,
                    reason="New in the catalog — no personal taste signal yet",
                    is_new=is_new_release(movie.release_year, today=today),
                )
                for movie in ordered
            ]

        scores = score_rows(profile, build_rows(features, movies))
        ranked = np.argsort(-scores, kind="stable")[:limit]
        picks: list[ColdStartPick] = []
        for position in ranked:
            movie = movies[int(position)]
            similarity = float(scores[int(position)])
            if similarity <= 0:
                continue
            # Favourite genres are lower-cased by ``favourite_genres``; compare like with like and
            # then name the film's own label, which is what the catalog displays.
            shared = sorted(
                genre
                for genre in split_genres(movie.genres or "")
                if genre.lower() in favourite_genres
            )
            picks.append(
                ColdStartPick(
                    movie=movie,
                    similarity=similarity,
                    predicted_rating=float(
                        np.clip(user_mean + DEFAULT_PREDICTION_SLOPE * similarity, 0.5, 5.0)
                    ),
                    reason=(
                        f"New release matching your taste in {', '.join(shared[:2])}"
                        if shared
                        else "New release close to the films you rate highly"
                    ),
                    is_new=is_new_release(movie.release_year, today=today),
                )
            )
        return picks
