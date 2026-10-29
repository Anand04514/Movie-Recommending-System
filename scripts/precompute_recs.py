"""Warm ``precomputed_recs`` for users who actually have a profile.

The repository's offline job (``train_and_precompute.py``) trains artifacts; this one turns those
artifacts into per-user candidate rows, which the API then serves without running a model. Run it
after every retrain, or on a schedule:

    uv run python scripts/precompute_recs.py            # 200 most recently active profiles
    uv run python scripts/precompute_recs.py --users 0  # every user with at least one rating
    uv run python scripts/precompute_recs.py --algorithms hybrid,item_cf --limit 50

It deliberately warms a *pool* (``--limit``, default 50) rather than one page, so page sizes from 1
to 50 are served from the same rows. Seeded MovieLens demo users are skipped by default: they have
no usable login and never make a request.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sqlalchemy import func, select

# Allow `python scripts/precompute_recs.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.db import get_session_factory
from backend.app.models import Rating, User
from backend.app.precompute import cached_users, clear, warm_user
from backend.app.recommendations import (
    ContentService,
    HybridService,
    RecommendationService,
    SvdService,
    UserCfService,
)

DEFAULT_ALGORITHMS = ("hybrid", "svd", "item_cf", "user_cf", "content")


def build_services(processed_dir: Path) -> dict[str, object]:
    """The same services the API uses, built once for the whole run."""
    item = RecommendationService(processed_dir)
    user = UserCfService(processed_dir)
    content = ContentService(processed_dir, fallback=item)
    svd = SvdService(processed_dir, fallback=item)
    # No explicit base or weight: the tuned hybrid_weights.json decides, exactly as in the API.
    hybrid = HybridService(item, user, content, processed_dir=processed_dir)
    return {
        "item_cf": item,
        "user_cf": user,
        "svd": svd,
        "content": content,
        "hybrid": hybrid,
    }


def active_user_ids(db, *, users: int) -> list[int]:
    """Users with at least one rating, most recently active first (``users=0`` means all)."""
    statement = (
        select(Rating.user_id, func.max(Rating.updated_at).label("last_seen"))
        .join(User, User.id == Rating.user_id)
        .where(User.is_seeded.is_(False))
        .group_by(Rating.user_id)
        .order_by(func.max(Rating.updated_at).desc())
    )
    if users > 0:
        statement = statement.limit(users)
    return [int(row[0]) for row in db.execute(statement).all()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--users", type=int, default=200, help="Profiles to warm; 0 means all.")
    parser.add_argument("--limit", type=int, default=50, help="Candidate pool per user and method.")
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated subset of hybrid, svd, item_cf, user_cf, content.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete every existing cached row before warming.",
    )
    args = parser.parse_args(argv)

    algorithms = tuple(name.strip() for name in args.algorithms.split(",") if name.strip())
    unknown = [name for name in algorithms if name not in DEFAULT_ALGORITHMS]
    if unknown:
        parser.error(f"unknown algorithm(s): {unknown}; choose from {list(DEFAULT_ALGORITHMS)}")
    if args.limit < 1:
        parser.error("--limit must be positive")

    services = build_services(args.processed_dir)
    session_factory = get_session_factory()
    started = time.perf_counter()

    with session_factory() as db:
        if args.clear:
            removed = clear(db)
            print(f"cleared {removed} cached rows")

        user_ids = active_user_ids(db, users=args.users)
        totals = {name: 0 for name in algorithms}
        warmed = 0
        for user_id in user_ids:
            written = warm_user(db, user_id, services, algorithms=algorithms, limit=args.limit)
            if written:
                warmed += 1
            for name, count in written.items():
                totals[name] += count

    elapsed = time.perf_counter() - started
    print(f"warmed {warmed}/{len(user_ids)} profiles in {elapsed:.1f}s (limit={args.limit})")
    for name, count in totals.items():
        with session_factory() as db:
            profiles = cached_users(db, name)
        print(f"  {name:<8} {count:>6} candidates across {profiles} profiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
