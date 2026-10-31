"""A/B experiments: sticky variant assignment and engagement reporting.

The first experiment decides which model a user sees *by default*. It is deliberately small and
honest about what it measures:

- Assignment is a hash of ``(experiment, user_id)``, so it is stable before any row is written and
  reproducible from the user id alone; the row in ``experiment_assignments`` makes it auditable.
- Only traffic the experiment actually assigned counts. A user who picks a model in the UI sends an
  explicit algorithm, that list is logged with no variant, and it never enters the comparison.
- An outcome is attributed to an impression when the same user liked, rejected or rated that film
  *at or after* the moment it was shown. Nothing else is counted, so a rating that predates the
  recommendation cannot flatter the arm that happened to show it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.models import (
    ExperimentAssignment,
    Feedback,
    Rating,
    RecommendationLog,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    name: str
    variants: tuple[str, ...]
    description: str


DEFAULT_MODEL_EXPERIMENT = "default_model_v1"

EXPERIMENTS: dict[str, Experiment] = {
    DEFAULT_MODEL_EXPERIMENT: Experiment(
        name=DEFAULT_MODEL_EXPERIMENT,
        variants=("hybrid", "user_cf"),
        description=(
            "Which model a signed-in user gets when they have not chosen one: the tuned hybrid "
            "or the best-ranking single model."
        ),
    ),
}


def get_experiment(name: str | None = None) -> Experiment:
    """The named experiment, or the default one when ``name`` is omitted."""
    key = name or DEFAULT_MODEL_EXPERIMENT
    try:
        return EXPERIMENTS[key]
    except KeyError:
        raise ValueError(
            f"unknown experiment {key!r}; known experiments: {sorted(EXPERIMENTS)}"
        ) from None


def bucket_for(experiment: Experiment, user_id: int) -> str:
    """Deterministic, evenly spread variant for a user.

    A hash keeps the split reproducible without storing anything, which matters for tests and for
    reasoning about a backfill: the same user always lands in the same arm.
    """
    digest = hashlib.sha256(f"{experiment.name}:{int(user_id)}".encode()).hexdigest()
    return experiment.variants[int(digest[:8], 16) % len(experiment.variants)]


def assign(db: Session, user_id: int, experiment: Experiment | None = None) -> ExperimentAssignment:
    """Return the user's assignment, creating it on first sight.

    The row is the source of truth once written, so changing the variant list in code never moves
    a user who is already assigned.
    """
    experiment = experiment or get_experiment()
    existing = db.scalar(
        select(ExperimentAssignment).where(
            ExperimentAssignment.user_id == int(user_id),
            ExperimentAssignment.experiment == experiment.name,
        )
    )
    if existing is not None:
        return existing

    row = ExperimentAssignment(
        user_id=int(user_id),
        experiment=experiment.name,
        variant=bucket_for(experiment, user_id),
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # Two concurrent first requests: the other one won, so use the row it wrote.
        db.rollback()
        row = db.scalar(
            select(ExperimentAssignment).where(
                ExperimentAssignment.user_id == int(user_id),
                ExperimentAssignment.experiment == experiment.name,
            )
        )
        if row is None:  # pragma: no cover - only if the competing row vanished
            raise
        return row
    except SQLAlchemyError:
        # Analytics must never break the page; fall back to the deterministic bucket.
        db.rollback()
        logger.warning("Could not persist experiment assignment", exc_info=True)
        return ExperimentAssignment(
            user_id=int(user_id),
            experiment=experiment.name,
            variant=bucket_for(experiment, user_id),
        )
    return row


@dataclass(frozen=True)
class VariantReport:
    variant: str
    users: int
    impressions: int
    likes: int
    rejects: int
    clicks: int
    rated_after: int

    @property
    def positive_rate(self) -> float | None:
        """Likes as a share of the like-or-reject verdicts. Clicks are not a verdict."""
        verdicts = self.likes + self.rejects
        return self.likes / verdicts if verdicts else None

    @property
    def rated_rate(self) -> float | None:
        """Share of shown picks the user went on to rate."""
        return self.rated_after / self.impressions if self.impressions else None

    @property
    def engagement_rate(self) -> float | None:
        """Share of shown picks that produced any signal at all: a verdict, a click or a rating."""
        if not self.impressions:
            return None
        return (self.likes + self.rejects + self.clicks + self.rated_after) / self.impressions


@dataclass(frozen=True)
class ArmComparison:
    """The gap between two arms, with the uncertainty that a small sample leaves behind.

    A bare percentage difference invites reading noise as a result: 3 of 10 picks against 2 of 10 is
    "10 points" of nothing. The interval is Newcombe's square-and-add method for a difference of
    proportions, which behaves well at the small counts an experiment actually has on day one.
    """

    metric: str
    a_variant: str
    b_variant: str
    difference: float
    low: float
    high: float
    confidence: float
    significant: bool


Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, *, z: float = Z_95) -> tuple[float, float] | None:
    """Score interval for one proportion. Not the textbook normal approximation.

    The normal approximation collapses to zero width when a rate is 0% or 100%, which is exactly
    where a new experiment starts, so the score interval is used instead.
    """
    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def compare_arms(
    a: VariantReport, b: VariantReport, *, metric: str = "engagement_rate", z: float = Z_95
) -> ArmComparison | None:
    """Effect of arm ``b`` over arm ``a`` on one rate, or ``None`` when either has no traffic."""
    rate_a = getattr(a, metric)
    rate_b = getattr(b, metric)
    if rate_a is None or rate_b is None:
        return None
    interval_a = wilson_interval(
        a.likes + a.rejects + a.clicks + a.rated_after, a.impressions, z=z
    )
    interval_b = wilson_interval(
        b.likes + b.rejects + b.clicks + b.rated_after, b.impressions, z=z
    )
    if interval_a is None or interval_b is None:  # pragma: no cover - rates exist only with traffic
        return None
    low_a, high_a = interval_a
    low_b, high_b = interval_b
    # Newcombe's square-and-add: the two arms are independent, so their errors add in quadrature.
    spread = (((rate_b - low_b) ** 2) + ((high_a - rate_a) ** 2)) ** 0.5
    difference = rate_b - rate_a
    low, high = difference - spread, difference + spread
    return ArmComparison(
        metric=metric,
        a_variant=a.variant,
        b_variant=b.variant,
        difference=difference,
        low=low,
        high=high,
        confidence=0.95,
        significant=low > 0 or high < 0,
    )


@dataclass(frozen=True)
class ExperimentReport:
    name: str
    description: str
    variants: list[VariantReport]

    @property
    def comparison(self) -> ArmComparison | None:
        """The first two arms compared on end-to-end engagement, or ``None`` under three arms."""
        if len(self.variants) != 2:
            return None
        return compare_arms(self.variants[0], self.variants[1])


def _verdicts_by_movie(
    db: Session, user_ids: set[int]
) -> dict[tuple[int, int], list[tuple[datetime, str]]]:
    if not user_ids:
        return {}
    rows = db.execute(
        select(Feedback.user_id, Feedback.movie_id, Feedback.action, Feedback.created_at).where(
            Feedback.user_id.in_(user_ids)
        )
    ).all()
    result: dict[tuple[int, int], list[tuple[datetime, str]]] = {}
    for user_id, movie_id, action, created_at in rows:
        result.setdefault((int(user_id), int(movie_id)), []).append((created_at, action))
    return result


def _ratings_by_movie(db: Session, user_ids: set[int]) -> dict[tuple[int, int], datetime]:
    if not user_ids:
        return {}
    rows = db.execute(
        select(Rating.user_id, Rating.movie_id, Rating.updated_at).where(
            Rating.user_id.in_(user_ids)
        )
    ).all()
    return {(int(user_id), int(movie_id)): updated_at for user_id, movie_id, updated_at in rows}


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; every timestamp in this app is UTC."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def report(db: Session, experiment: Experiment | None = None) -> ExperimentReport:
    """Per-arm counts, computed from impressions and the outcomes that followed them."""
    experiment = experiment or get_experiment()

    assigned: dict[str, set[int]] = {variant: set() for variant in experiment.variants}
    for variant, user_id in db.execute(
        select(ExperimentAssignment.variant, ExperimentAssignment.user_id).where(
            ExperimentAssignment.experiment == experiment.name
        )
    ).all():
        assigned.setdefault(variant, set()).add(int(user_id))

    impressions = db.execute(
        select(
            RecommendationLog.user_id,
            RecommendationLog.movie_id,
            RecommendationLog.variant,
            RecommendationLog.shown_at,
        ).where(RecommendationLog.experiment == experiment.name)
    ).all()

    user_ids = {int(row[0]) for row in impressions}
    verdicts = _verdicts_by_movie(db, user_ids)
    ratings = _ratings_by_movie(db, user_ids)

    counters = {
        variant: {"impressions": 0, "likes": 0, "rejects": 0, "clicks": 0, "rated_after": 0}
        for variant in experiment.variants
    }
    for user_id, movie_id, variant, shown_at in impressions:
        if variant not in counters:
            continue
        key = (int(user_id), int(movie_id))
        shown = _as_utc(shown_at)
        bucket = counters[variant]
        bucket["impressions"] += 1

        for created_at, action in verdicts.get(key, ()):
            when = _as_utc(created_at)
            if shown is not None and when is not None and when < shown:
                continue
            if action == "liked":
                bucket["likes"] += 1
            elif action == "rejected":
                bucket["rejects"] += 1
            elif action == "clicked":
                bucket["clicks"] += 1

        rated_at = _as_utc(ratings.get(key))
        if rated_at is not None and shown is not None and rated_at >= shown:
            bucket["rated_after"] += 1

    return ExperimentReport(
        name=experiment.name,
        description=experiment.description,
        variants=[
            VariantReport(variant=variant, users=len(assigned.get(variant, set())), **counts)
            for variant, counts in counters.items()
        ],
    )
