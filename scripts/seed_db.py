#!/usr/bin/env python
"""Seed the catalog, plus optionally MovieLens demo users and their ratings.

Usage:
    uv run python scripts/seed_db.py                     # catalog + 500 demo users
    uv run python scripts/seed_db.py --demo-users 0      # catalog only
    uv run python scripts/seed_db.py --refresh-demo      # reseed demo users
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, insert, inspect, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

# Allow `python scripts/seed_db.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.db import get_engine, get_session_factory  # noqa: E402
from backend.app.models import Movie, Rating, User  # noqa: E402

MOVIE_COLUMNS = (
    "id",
    "title_display",
    "title_search",
    "release_year",
    "genres",
    "imdb_id",
    "tmdb_id",
    "poster_path",
    "overview",
    "tmdb_keywords",
    "rating_count",
    "in_model",
    "source",
    "added_at",
)


def _chunks(rows: list[dict], size: int = 5_000) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _upsert(
    session: Session,
    model: type[Movie],
    rows: list[dict],
    *,
    index_elements: list[str],
    update_columns: Iterable[str],
) -> None:
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(model)
    elif dialect == "sqlite":
        statement = sqlite_insert(model)
    else:
        raise RuntimeError(f"seed_db.py does not support the {dialect} dialect")
    if not rows:
        return

    # Executemany keeps each statement well under SQLite's bound-variable limit, which a single
    # wide VALUES list for 87k movies would exceed.
    updates = {column: getattr(statement.excluded, column) for column in update_columns}
    session.execute(
        statement.on_conflict_do_update(index_elements=index_elements, set_=updates),
        rows,
    )


def catalog_rows(processed_dir: Path) -> list[dict]:
    catalog = pd.read_parquet(processed_dir / "catalog.parquet")
    now = datetime.now(UTC)
    rows: list[dict] = []
    for row in catalog.itertuples(index=False):
        rows.append(
            {
                "id": int(row.movie_id),
                "title_display": str(row.title_display),
                "title_search": str(row.title_search),
                "release_year": None if pd.isna(row.release_year) else int(row.release_year),
                "genres": "" if pd.isna(row.genres) else str(row.genres),
                "imdb_id": None if pd.isna(row.imdb_id) else int(row.imdb_id),
                "tmdb_id": None if pd.isna(row.tmdb_id) else int(row.tmdb_id),
                "poster_path": None,
                "overview": None,
                "tmdb_keywords": None,
                "rating_count": int(row.rating_count),
                "in_model": bool(row.in_model),
                "source": str(row.source),
                "added_at": now,
            }
        )
    return rows


def seed_catalog(session: Session, processed_dir: Path) -> int:
    rows = catalog_rows(processed_dir)
    _upsert(
        session,
        Movie,
        rows,
        index_elements=["id"],
        update_columns=(
            "title_display",
            "title_search",
            "release_year",
            "genres",
            "imdb_id",
            "tmdb_id",
            "rating_count",
            "in_model",
            "source",
        ),
    )
    session.commit()
    return len(rows)


def seed_demo_users(
    session: Session, processed_dir: Path, count: int, *, refresh: bool
) -> tuple[int, int]:
    """Create seeded MovieLens users (no usable password) with their training ratings."""
    existing = session.scalar(
        select(func.count()).select_from(User).where(User.is_seeded.is_(True))
    )
    if existing and not refresh:
        print(f"  {existing:,} demo users already present; use --refresh-demo to rebuild")
        return 0, 0

    if refresh and existing:
        session.execute(delete(User).where(User.is_seeded.is_(True)))
        session.commit()
        print(f"  removed {existing:,} existing demo users")

    if count <= 0:
        return 0, 0

    ratings = pd.read_parquet(
        processed_dir / "ratings_train.parquet", columns=["userId", "movieId", "rating"]
    )
    known_movies = set(session.scalars(select(Movie.id)))
    ratings = ratings[ratings["movieId"].isin(known_movies)]
    eligible = ratings["userId"].drop_duplicates().sort_values().head(count).tolist()
    if not eligible:
        return 0, 0

    now = datetime.now(UTC)
    users = [
        User(
            email=f"demo-{int(user_id)}@movielens.invalid",
            display_name=f"MovieLens user {int(user_id)}",
            # "!" is not a valid bcrypt hash, so seeded accounts can never sign in.
            password_hash="!",
            is_seeded=True,
            created_at=now,
        )
        for user_id in eligible
    ]
    session.add_all(users)
    session.flush()

    by_movielens_id = dict(zip(eligible, users, strict=True))
    rating_rows: list[dict] = []
    for user_id, group in ratings[ratings["userId"].isin(eligible)].groupby("userId"):
        user = by_movielens_id[int(user_id)]
        for row in group.itertuples(index=False):
            rating_rows.append(
                {
                    "user_id": user.id,
                    "movie_id": int(row.movieId),
                    "rating": float(row.rating),
                    "created_at": now,
                    "updated_at": now,
                }
            )

    for chunk in _chunks(rating_rows):
        session.execute(insert(Rating), chunk)
    session.commit()
    return len(users), len(rating_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed the catalog and demo users.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--demo-users",
        type=int,
        default=500,
        help="MovieLens users to mirror into the database (0 disables).",
    )
    parser.add_argument(
        "--refresh-demo",
        action="store_true",
        help="Delete existing seeded users and reseed them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = get_engine()
    missing = [
        table
        for table in ("users", "movies", "ratings")
        if not inspect(engine).has_table(table)
    ]
    if missing:
        print(
            f"error: missing table(s) {', '.join(missing)}; run `uv run alembic upgrade head`",
            file=sys.stderr,
        )
        return 1
    if not (args.processed_dir / "catalog.parquet").is_file():
        print(
            f"error: {args.processed_dir / 'catalog.parquet'} not found; "
            "run scripts/prepare_data.py first",
            file=sys.stderr,
        )
        return 1

    started = time.perf_counter()
    with get_session_factory()() as session:
        movies = seed_catalog(session, args.processed_dir)
        print(f"Catalog: {movies:,} movies upserted")
        users, ratings = seed_demo_users(
            session, args.processed_dir, args.demo_users, refresh=args.refresh_demo
        )
        print(f"Demo users: {users:,} created with {ratings:,} ratings")
        totals = (
            session.scalar(select(func.count()).select_from(Movie)),
            session.scalar(select(func.count()).select_from(User)),
            session.scalar(select(func.count()).select_from(Rating)),
        )
    print(
        f"Database now holds {totals[0]:,} movies, {totals[1]:,} users, "
        f"{totals[2]:,} ratings ({time.perf_counter() - started:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
