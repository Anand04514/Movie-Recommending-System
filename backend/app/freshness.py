"""Recency: is a film new, and how much does that lift it in a list? (5.12)

The catalog stores a release *year*, not a date, so age is approximated from 1 July of that year.
The halfway point is the least wrong single guess for a corpus of films spread across twelve months.

The boost is deliberately small and decays: ``weight * exp(-age_days / 365)``. On MovieLens ml-32m
the effect is close to zero by construction — the dataset ends in 2023, so every film in it is years
old — and its real home is films ingested from TMDB afterwards. That is stated in the README rather
than dressed up as a measured win.
"""

from __future__ import annotations

import math
from datetime import date

MIN_YEAR = 1880
MAX_YEAR = 2100
DAYS_PER_YEAR = 365.0


def release_date_for(release_year: int | None) -> date | None:
    """The mid-year stand-in for a release year, or ``None`` when the year is unusable."""
    if release_year is None:
        return None
    year = int(release_year)
    if not MIN_YEAR <= year <= MAX_YEAR:
        return None
    return date(year, 7, 1)


def age_in_days(release_year: int | None, *, today: date) -> int | None:
    released = release_date_for(release_year)
    if released is None:
        return None
    return (today - released).days


def is_new_release(
    release_year: int | None, *, today: date, years: int = 2
) -> bool:
    """True when the film is within the recency window. A future-scheduled film is not 'new'."""
    if years <= 0:
        return False
    age = age_in_days(release_year, today=today)
    if age is None:
        return False
    return 0 <= age <= years * DAYS_PER_YEAR


def boost_for(
    release_year: int | None,
    *,
    today: date,
    years: int = 2,
    weight: float = 0.0,
) -> float:
    """Freshness bonus added to a candidate's score before re-ranking.

    Zero for anything outside the window, so an old film's rank is untouched by the mechanism.
    """
    if weight <= 0 or not is_new_release(release_year, today=today, years=years):
        return 0.0
    age = age_in_days(release_year, today=today)
    if age is None:  # pragma: no cover - is_new_release already rejected this
        return 0.0
    return float(weight) * math.exp(-age / DAYS_PER_YEAR)
