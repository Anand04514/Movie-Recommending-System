#!/usr/bin/env python
"""Cache TMDB posters and overviews for the catalog.

The API enriches a movie when its detail page is opened, which is right for one click and slow for
browsing a list. This walks the catalog once, most-rated first, and is safe to re-run: a movie that
already has a poster and an overview, or that is already in the on-disk cache, costs no request.

Usage:
    uv run python scripts/fetch_posters.py --limit 300
    uv run python scripts/fetch_posters.py --dry-run --min-rating-count 20
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

# Allow `python scripts/fetch_posters.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.db import get_session_factory, init_database  # noqa: E402
from backend.app.models import Movie  # noqa: E402
from backend.app.tmdb import TmdbClient, TmdbMovie, build_tmdb_client  # noqa: E402

ProgressCallback = Callable[[int, int, int], None]


def plan_fetch(
    session: Session, *, limit: int, min_rating_count: int = 0
) -> list[tuple[int, int]]:
    """``(movie id, tmdb id)`` pairs still missing a poster or an overview, most-rated first."""
    if limit < 1:
        raise ValueError("limit must be positive")
    statement = (
        select(Movie.id, Movie.tmdb_id)
        .where(
            Movie.tmdb_id.is_not(None),
            Movie.tmdb_id > 0,
            Movie.rating_count >= min_rating_count,
            or_(Movie.poster_path.is_(None), Movie.overview.is_(None)),
        )
        .order_by(Movie.rating_count.desc(), Movie.id)
        .limit(limit)
    )
    return [(int(movie_id), int(tmdb_id)) for movie_id, tmdb_id in session.execute(statement)]


def store_fetched(session: Session, movie_id: int, fetched: TmdbMovie | None) -> bool:
    """Write a fetched poster/overview onto its movie; returns whether the row changed."""
    if fetched is None or (fetched.poster_path is None and fetched.overview is None):
        return False
    movie = session.get(Movie, movie_id)
    if movie is None:
        return False
    changed = False
    if fetched.poster_path and movie.poster_path != fetched.poster_path:
        movie.poster_path = fetched.poster_path
        changed = True
    if fetched.overview and movie.overview != fetched.overview:
        movie.overview = fetched.overview
        changed = True
    return changed


def run_fetch(
    session: Session,
    client: TmdbClient,
    pairs: Sequence[tuple[int, int]],
    *,
    batch_size: int = 50,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """Fetch each pair and commit in batches.

    A movie TMDB does not know, or a request that fails, is counted and skipped so one bad id never
    ends the run. ``progress`` is called with ``(visited, stored, unavailable)`` after each movie.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    counters = {"planned": len(pairs), "stored": 0, "missing": 0}
    for index, (movie_id, tmdb_id) in enumerate(pairs, start=1):
        fetched = client.fetch(tmdb_id)
        if fetched is None or (not fetched.poster_path and not fetched.overview):
            counters["missing"] += 1
        elif store_fetched(session, movie_id, fetched):
            counters["stored"] += 1
        if index % batch_size == 0:
            session.commit()
        if progress is not None:
            progress(index, counters["stored"], counters["missing"])
    session.commit()
    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache TMDB posters and overviews for catalog movies."
    )
    parser.add_argument("--limit", type=int, default=500, help="How many movies to visit.")
    parser.add_argument(
        "--min-rating-count",
        type=int,
        default=0,
        help="Skip movies with fewer MovieLens ratings than this.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N movies.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report what would be fetched."
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], TmdbClient] = build_tmdb_client,
) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1 or args.batch_size < 1:
        print("error: --limit and --batch-size must be positive", file=sys.stderr)
        return 2

    init_database()
    with get_session_factory()() as session:
        pairs = plan_fetch(
            session, limit=args.limit, min_rating_count=args.min_rating_count
        )
        print(f"{len(pairs):,} movies are missing a poster, an overview, or both")
        if not pairs:
            return 0
        if args.dry_run:
            for movie_id, tmdb_id in pairs[:10]:
                print(f"  would fetch movie {movie_id} (tmdb {tmdb_id})")
            if len(pairs) > 10:
                print(f"  ... and {len(pairs) - 10:,} more")
            return 0

        client = client_factory()
        if not client.enabled:
            print(
                "error: TMDB is not configured. Set TMDB_READ_ACCESS_TOKEN (preferred) or "
                "TMDB_API_KEY in .env, then run this again.",
                file=sys.stderr,
            )
            return 1

        def progress(visited: int, stored: int, unavailable: int) -> None:
            if visited % 25 == 0 or visited == len(pairs):
                print(
                    f"  {visited:,}/{len(pairs):,} visited, {stored:,} stored, "
                    f"{unavailable:,} not on TMDB"
                )

        counters = run_fetch(
            session, client, pairs, batch_size=args.batch_size, progress=progress
        )

    print(
        f"Done: {counters['stored']:,} movies updated, {counters['missing']:,} unavailable. "
        "Posters are served from the database; the cache lives in the processed directory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
