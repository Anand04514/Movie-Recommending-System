"""Discovery picks: a small, labelled share of a recommendation list outside the usual taste.

Recommendation lists drift toward the same blockbusters and the same two genres. Reserving a
slice of the list for a film with weak genre overlap gives the user something to react to, while
still requiring positive evidence (the candidate must have been scored by the model, not pulled
from nowhere).

The rule is deliberately simple and testable: among the lower half of the ranked list, choose the
highest-scoring candidates whose genre overlap with the user's favourite genres is at most
``max_overlap``. Nothing is invented when no candidate qualifies.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_OVERLAP = 0.34


@dataclass(frozen=True)
class DiscoveryCandidate:
    """The bits of a ranked candidate the discovery rule needs."""

    movie_id: int
    score: float
    genres: tuple[str, ...]


def genre_overlap(genres: tuple[str, ...] | list[str], favourites: set[str]) -> float:
    """Share of the candidate's genres that are among the user's favourites."""
    if not genres:
        return 0.0
    normalised = {genre.strip().lower() for genre in genres if genre}
    if not normalised:
        return 0.0
    return len(normalised & {genre.lower() for genre in favourites}) / len(normalised)


def discovery_slots(limit: int, *, share: float = 0.15, ratings_count: int = 0) -> int:
    """How many of a list of ``limit`` items to reserve for discovery.

    Only full-length lists get the treatment, and only once the user has given the model enough
    ratings to have a "usual taste" to step outside of.
    """
    if limit < 8 or ratings_count < 5:
        return 0
    return max(1, min(2, round(limit * share)))


def pick_discovery(
    candidates: list[DiscoveryCandidate],
    *,
    favourite_genres: set[str],
    slots: int,
    max_overlap: float = DEFAULT_MAX_OVERLAP,
) -> list[DiscoveryCandidate]:
    """Choose up to ``slots`` low-overlap candidates, highest score first.

    ``candidates`` should already be in final ranking order, and only the tail of the list should
    be passed in: discovery is a label for the back of the list, not a reason to promote an
    arbitrary film above the model's best picks.
    """
    if slots < 1 or not candidates or not favourite_genres:
        return []
    if not 0.0 <= max_overlap <= 1.0:
        raise ValueError("max_overlap must be between 0 and 1")

    eligible = [
        candidate
        for candidate in candidates
        if genre_overlap(candidate.genres, favourite_genres) <= max_overlap
    ]
    eligible.sort(key=lambda candidate: (-candidate.score, candidate.movie_id))
    return eligible[:slots]


def favourite_genres(
    rated_genres: list[tuple[tuple[str, ...], float]],
    *,
    limit: int = 2,
) -> set[str]:
    """The genres the user rates most often, weighted by how much they liked those films.

    A genre's score is the number of rated films in it times the average rating above 2.5, so
    disliking a whole shelf of horror does not make horror a favourite.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    totals: dict[str, float] = {}
    for genres, rating in rated_genres:
        weight = max(float(rating) - 2.5, 0.0)
        if weight <= 0:
            continue
        for genre in {genre.strip().lower() for genre in genres if genre}:
            totals[genre] = totals.get(genre, 0.0) + weight
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return {genre for genre, _ in ordered[:limit]}
