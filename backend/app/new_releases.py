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
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.freshness import is_new_release
from backend.app.models import Feedback, Movie, Rating, RecommendationLog
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

# The algorithm label impressions from the "New releases for you" row are logged under.
NEW_RELEASE_ALGORITHM = "new_releases"


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


@dataclass(frozen=True)
class ReleaseEngagement:
    """What the new-release row has actually achieved, from impressions and their outcomes.

    PLAN 5.12 asks for the share of new films in lists and their like rates; this is that check,
    computed from ``recommendation_logs`` rather than read off the boost's formula.
    """

    impressions: int
    total_impressions: int
    users: int
    films: int
    likes: int
    rejects: int
    clicks: int
    rated_after: int

    @property
    def share_of_impressions(self) -> float | None:
        """How much of what the app shows comes from the recency row."""
        return self.impressions / self.total_impressions if self.total_impressions else None

    @property
    def like_rate(self) -> float | None:
        verdicts = self.likes + self.rejects
        return self.likes / verdicts if verdicts else None

    @property
    def rated_rate(self) -> float | None:
        return self.rated_after / self.impressions if self.impressions else None


def _utc(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; every timestamp in this app is UTC."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def engagement(db: Session) -> ReleaseEngagement:
    """Count the row's impressions and the signals that followed them.

    An outcome counts only when it happened at or after the impression of that film, so a rating
    the user already had cannot be read as a response to the recommendation.
    """
    rows = db.execute(
        select(
            RecommendationLog.user_id,
            RecommendationLog.movie_id,
            RecommendationLog.shown_at,
        ).where(RecommendationLog.algorithm == NEW_RELEASE_ALGORITHM)
    ).all()
    total = db.scalar(select(func.count()).select_from(RecommendationLog)) or 0
    if not rows:
        return ReleaseEngagement(0, int(total), 0, 0, 0, 0, 0, 0)

    user_ids = {int(row[0]) for row in rows}
    verdicts: dict[tuple[int, int], list[tuple[datetime | None, str]]] = {}
    for user_id, movie_id, action, created_at in db.execute(
        select(Feedback.user_id, Feedback.movie_id, Feedback.action, Feedback.created_at).where(
            Feedback.user_id.in_(user_ids)
        )
    ).all():
        verdicts.setdefault((int(user_id), int(movie_id)), []).append((_utc(created_at), action))

    rated_at: dict[tuple[int, int], datetime | None] = {
        (int(user_id), int(movie_id)): _utc(updated_at)
        for user_id, movie_id, updated_at in db.execute(
            select(Rating.user_id, Rating.movie_id, Rating.updated_at).where(
                Rating.user_id.in_(user_ids)
            )
        ).all()
    }

    likes = rejects = clicks = rated_after = 0
    users: set[int] = set()
    films: set[int] = set()
    for user_id, movie_id, shown_at in rows:
        user_id, movie_id = int(user_id), int(movie_id)
        users.add(user_id)
        films.add(movie_id)
        shown = _utc(shown_at)
        for when, action in verdicts.get((user_id, movie_id), ()):
            if shown is not None and when is not None and when < shown:
                continue
            if action == "liked":
                likes += 1
            elif action == "rejected":
                rejects += 1
            elif action == "clicked":
                clicks += 1
        when = rated_at.get((user_id, movie_id))
        if when is not None and shown is not None and when >= shown:
            rated_after += 1

    return ReleaseEngagement(
        impressions=len(rows),
        total_impressions=int(total),
        users=len(users),
        films=len(films),
        likes=likes,
        rejects=rejects,
        clicks=clicks,
        rated_after=rated_after,
    )
