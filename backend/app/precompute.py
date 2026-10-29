"""Precomputed recommendation candidates: the read/write side of 5.9.

The cache stores **scored candidates** rather than rendered cards. Serving still resolves movie
rows, drops anything the user has rated or rejected since, applies filters and writes the reason
lines, so a cache hit can never resurrect a rated film or show a stale explanation. What it skips
is the expensive part: neighbour and factor maths over the whole item space.

Validity is deliberately conservative:

- the rows must be newer than the serving artifacts, so a retrain invalidates them;
- they must be newer than ``PRECOMPUTE_TTL``, so a cache left alone for a day goes cold;
- the pool must be large enough to fill the requested limit after live exclusions, otherwise the
  request falls back to the model and rewrites the cache.

Write-through is best effort: a cache that cannot be written is a slow request, never a failed one.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.models import PrecomputedRec, Rating
from backend.app.recommendations import RecommendationBatch, ScoredItem

logger = logging.getLogger(__name__)

# Serving artifacts whose retraining makes a cached candidate stale. user_neighbors.npz is left
# out on purpose: it is built by the evaluation harness and never read while serving.
ARTIFACT_FILES = (
    "item_neighbors.npz",
    "svd_model.npz",
    "content_features.npz",
    "hybrid_weights.json",
)


def as_utc(value: datetime | None) -> datetime | None:
    """Normalise a timestamp: SQLite returns naive datetimes, and everything here is UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def artifact_version(processed_dir: str | Path) -> datetime | None:
    """Newest modification time among the serving artifacts, or ``None`` when none exist yet."""
    directory = Path(processed_dir)
    newest: float | None = None
    for name in ARTIFACT_FILES:
        path = directory / name
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        newest = modified if newest is None else max(newest, modified)
    if newest is None:
        return None
    return datetime.fromtimestamp(newest, UTC)


def load_batch(
    db: Session,
    user_id: int,
    algorithm: str,
    *,
    processed_dir: str | Path,
    ttl: timedelta,
    limit: int,
) -> RecommendationBatch | None:
    """A cached candidate pool for this user and method, or ``None`` when it cannot be trusted."""
    rows = db.scalars(
        select(PrecomputedRec)
        .where(PrecomputedRec.user_id == int(user_id), PrecomputedRec.algorithm == algorithm)
        .order_by(PrecomputedRec.rank)
    ).all()
    if len(rows) < limit:
        return None

    computed_at = as_utc(rows[0].computed_at)
    if computed_at is None:
        return None
    if computed_at + ttl < datetime.now(UTC):
        return None
    version = artifact_version(processed_dir)
    if version is not None and computed_at < version:
        return None

    items = [
        ScoredItem(
            movie_id=int(row.movie_id),
            score=float(row.score),
            predicted_rating=float(row.predicted_rating),
            source_movie_id=int(row.source_movie_id) if row.source_movie_id is not None else None,
            source_rating=float(row.source_rating) if row.source_rating is not None else None,
            contributions=_load_contributions(row.contributions),
            neighbors_rated=row.neighbors_rated,
        )
        for row in rows
    ]
    return RecommendationBatch(algorithm=algorithm, cold_start=False, items=items)


def _load_contributions(raw: str | None) -> tuple[tuple[int, float], ...]:
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return ()
    pairs: list[tuple[int, float]] = []
    for entry in parsed if isinstance(parsed, list) else ():
        try:
            movie_id, strength = entry
            pairs.append((int(movie_id), float(strength)))
        except (TypeError, ValueError):
            continue
    return tuple(pairs)


def _dump_contributions(item: ScoredItem) -> str:
    pairs = [
        [int(movie_id), round(float(strength), 6)] for movie_id, strength in item.contributions
    ]
    return json.dumps(pairs)


def store_batch(
    db: Session, user_id: int, batch: RecommendationBatch, *, computed_at: datetime | None = None
) -> int:
    """Replace this user's cached candidates for the batch's method; returns rows written."""
    if batch.cold_start or not batch.items:
        return 0
    stamp = computed_at or datetime.now(UTC)
    try:
        db.execute(
            delete(PrecomputedRec).where(
                PrecomputedRec.user_id == int(user_id),
                PrecomputedRec.algorithm == batch.algorithm,
            )
        )
        db.add_all(
            [
                PrecomputedRec(
                    user_id=int(user_id),
                    algorithm=batch.algorithm,
                    movie_id=int(item.movie_id),
                    score=float(item.score),
                    rank=rank,
                    predicted_rating=float(item.predicted_rating),
                    source_movie_id=(
                        int(item.source_movie_id) if item.source_movie_id is not None else None
                    ),
                    source_rating=(
                        float(item.source_rating) if item.source_rating is not None else None
                    ),
                    neighbors_rated=item.neighbors_rated,
                    contributions=_dump_contributions(item),
                    computed_at=stamp,
                )
                for rank, item in enumerate(batch.items)
            ]
        )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.warning("Could not precompute recommendations for user %s", user_id, exc_info=True)
        return 0
    return len(batch.items)


def warm_user(
    db: Session,
    user_id: int,
    services: dict[str, object],
    *,
    algorithms: tuple[str, ...],
    limit: int,
) -> dict[str, int]:
    """Warm one user's candidates for each method; returns rows written per algorithm.

    Services are passed in so the warm-up job and the API share exactly one implementation of
    "run the model" and the job can be tested without a running server.
    """
    ratings = {
        int(movie_id): float(value)
        for movie_id, value in db.execute(
            select(Rating.movie_id, Rating.rating).where(Rating.user_id == int(user_id))
        ).all()
    }
    if not ratings:
        return {}

    written: dict[str, int] = {}
    for name in algorithms:
        service = services.get(name)
        if service is None:
            continue
        try:
            batch = service.recommend(list(ratings.items()), limit=limit)
        except Exception:  # noqa: BLE001 - one failed model must not stop the warm-up
            logger.warning("Warm-up failed for user %s with %s", user_id, name, exc_info=True)
            continue
        written[name] = store_batch(db, user_id, batch)
    return written


def cached_users(db: Session, algorithm: str) -> int:
    """How many users have at least one cached candidate for this method."""
    return int(
        db.scalar(
            select(func.count(func.distinct(PrecomputedRec.user_id))).where(
                PrecomputedRec.algorithm == algorithm
            )
        )
        or 0
    )


def clear(db: Session, user_id: int | None = None) -> int:
    """Drop cached candidates, for one user or for everyone; returns rows removed."""
    statement = delete(PrecomputedRec)
    if user_id is not None:
        statement = statement.where(PrecomputedRec.user_id == int(user_id))
    result = db.execute(statement)
    db.commit()
    return int(result.rowcount or 0)
