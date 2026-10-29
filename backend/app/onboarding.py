"""Pick a small, diverse set of well-known films for a new user to rate.

Asking a new user to rate five films from the same superhero franchise teaches the model very
little. The selection below walks the most-rated catalog films and takes them genre by genre, with
a cap on films from the same decade, so the answers spread across taste rather than confirming one
corner of it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from backend.app.models import Movie
from recsys.titles import split_genres

OTHER = "other"
DECADE_CAP = 2


def _genre_label(movie: Movie) -> str:
    genres = split_genres(movie.genres or "")
    if not genres or genres[0] == "(no genres listed)":
        return OTHER
    return genres[0]


def _decade(movie: Movie) -> int | None:
    return None if movie.release_year is None else movie.release_year // 10 * 10


def diverse_picks(movies: Sequence[Movie], *, limit: int = 12) -> list[Movie]:
    """Round-robin over primary genres, capping each decade at two picks.

    ``movies`` is expected in descending popularity order. The tail is filled from whatever is
    left if the caps cannot be met, so a tiny catalog still returns something useful.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if not movies:
        return []

    grouped: dict[str, list[Movie]] = defaultdict(list)
    for movie in movies:
        grouped[_genre_label(movie)].append(movie)

    genre_order = sorted(
        grouped,
        key=lambda genre: (-grouped[genre][0].rating_count, genre),
    )
    positions = dict.fromkeys(genre_order, 0)
    decades: dict[int, int] = defaultdict(int)
    chosen: list[Movie] = []
    seen: set[int] = set()

    while len(chosen) < limit:
        picked_any = False
        for genre in genre_order:
            if len(chosen) >= limit:
                break
            bucket = grouped[genre]
            index = positions[genre]
            while index < len(bucket):
                movie = bucket[index]
                index += 1
                if movie.id in seen:
                    continue
                decade = _decade(movie)
                if genre != OTHER and decade is not None and decades[decade] >= DECADE_CAP:
                    continue
                positions[genre] = index
                chosen.append(movie)
                seen.add(movie.id)
                if decade is not None:
                    decades[decade] += 1
                picked_any = True
                break
            positions[genre] = index
        if not picked_any:
            break

    if len(chosen) < limit:
        for movie in movies:
            if len(chosen) >= limit:
                break
            if movie.id not in seen:
                chosen.append(movie)
                seen.add(movie.id)
    return chosen
