"""On-demand TMDB ingestion (5.10).

A searched film becomes a catalog row only when a user opens or rates it, never on search, so a
browsing session cannot flood the catalog with typos. Three rules keep it honest:

- **Match on ``tmdbId`` first.** A MovieLens film that links to the same TMDB id is reused, not
  duplicated; IMDb id is the second check, because TMDB occasionally has both.
- **Ids come from a reserved range.** ``TMDB_ID_BASE + tmdbId`` is deterministic, stable across
  restarts, and above every MovieLens id, so a new film can never collide with an old one.
- **TMDB is optional.** Without a key, with no network, or on a rate limit, ingestion returns
  ``None`` and the API says the film is unavailable; nothing else breaks.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import Movie
from backend.app.tmdb import TmdbClient, TmdbDetails
from recsys.titles import normalize_for_search

logger = logging.getLogger(__name__)

# Above every MovieLens movieId (ml-32m tops out in the low 300,000s). TMDB ids stay below this, so
# the sum is unique per film and stable across runs.
TMDB_ID_BASE = 10_000_000
NO_GENRES = "(no genres listed)"


def movie_id_for(tmdb_id: int) -> int:
    return TMDB_ID_BASE + int(tmdb_id)


def find_by_tmdb_id(db: Session, tmdb_id: int) -> Movie | None:
    return db.scalar(select(Movie).where(Movie.tmdb_id == int(tmdb_id)))


def _row_from_details(details: TmdbDetails) -> Movie:
    genres = "|".join(details.genres) if details.genres else NO_GENRES
    return Movie(
        id=movie_id_for(details.tmdb_id),
        title_display=details.title,
        title_search=normalize_for_search(details.title),
        release_year=details.release_year,
        genres=genres,
        imdb_id=details.imdb_id,
        tmdb_id=details.tmdb_id,
        poster_path=details.poster_path,
        overview=details.overview,
        tmdb_keywords=",".join(details.keywords) if details.keywords else None,
        # TMDB vote counts are not MovieLens ratings, and inventing a number here would corrupt
        # popularity ordering, so the row starts at zero and reports itself as new.
        rating_count=0,
        in_model=False,
        source="tmdb",
    )


def ingest(db: Session, tmdb: TmdbClient, tmdb_id: int) -> Movie | None:
    """Return the catalog row for a TMDB film, creating it once.

    ``None`` means TMDB could not supply the film (disabled, offline, unknown id); callers turn
    that into a 404 rather than an exception.
    """
    existing = find_by_tmdb_id(db, tmdb_id)
    if existing is not None:
        return existing

    details = tmdb.details(tmdb_id)
    if details is None:
        return None

    if details.imdb_id is not None:
        by_imdb = db.scalar(select(Movie).where(Movie.imdb_id == details.imdb_id))
        if by_imdb is not None:
            # Same film under a different link: remember the tmdbId instead of duplicating it.
            if by_imdb.tmdb_id is None:
                by_imdb.tmdb_id = details.tmdb_id
                db.commit()
                db.refresh(by_imdb)
            return by_imdb

    movie = _row_from_details(details)
    db.add(movie)
    try:
        db.commit()
    except IntegrityError:
        # Two users opened the same new film at once: keep the row the other request wrote.
        db.rollback()
        existing = find_by_tmdb_id(db, tmdb_id)
        if existing is None:
            logger.warning("Could not ingest TMDB film %s", tmdb_id, exc_info=True)
        return existing
    db.refresh(movie)
    return movie
