#!/usr/bin/env python
"""Evaluate every method on the chronological split and record the results.

Reads the processed artifacts, samples held-out users, scores the five methods, prints a table,
writes ``data/processed/evaluation.json`` and (unless ``--no-db``) refreshes the ``model_runs``
table so the numbers survive outside the README.

Usage:
    uv run python scripts/evaluate_models.py
    uv run python scripts/evaluate_models.py --users 5000 --k 10 --chunk-size 256
    uv run python scripts/evaluate_models.py --no-db --json /tmp/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Allow `python scripts/evaluate_models.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.evaluate import (  # noqa: E402
    DEFAULT_RELEVANCE_THRESHOLD,
    MethodOutcome,
    default_scorers,
    evaluate_split,
    load_split,
    render_markdown_table,
)
from recsys.item_cf import load_neighbors  # noqa: E402
from recsys.user_cf import (  # noqa: E402
    UserNeighbors,
    build_user_neighbors,
    load_user_neighbors,
    save_user_neighbors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate recommenders on the chronological split."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--users", type=int, default=10_000, help="Held-out users to evaluate.")
    parser.add_argument("--k", type=int, default=10, help="Ranking cut-off.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for the user sample.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Users scored per block.")
    parser.add_argument("--user-k", type=int, default=50, help="User neighbors kept per user.")
    parser.add_argument(
        "--json", type=Path, default=Path("data/processed/evaluation.json"), help="Report path."
    )
    parser.add_argument("--no-db", action="store_true", help="Do not touch the model_runs table.")
    parser.add_argument(
        "--rebuild-user-neighbors",
        action="store_true",
        help="Rebuild the user neighbor artifact even if it exists.",
    )
    return parser


def item_neighbor_metadata(processed: Path) -> dict[str, object]:
    """The training metadata ``train_and_precompute.py`` wrote alongside the artifact."""
    sidecar = processed / "item_neighbors.json"
    if not sidecar.is_file():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_or_build_user_neighbors(
    processed: Path, train, *, user_k: int, rebuild: bool
) -> tuple[UserNeighbors, float, bool]:
    """Reuse the saved user neighbor table, or build it once and save it."""
    path = processed / "user_neighbors.npz"
    if path.is_file() and not rebuild:
        return load_user_neighbors(path), 0.0, True

    started = time.perf_counter()
    neighbors = build_user_neighbors(train, k=user_k)
    seconds = time.perf_counter() - started
    save_user_neighbors(neighbors, path)
    return neighbors, seconds, False


def write_model_runs(
    outcomes: list[MethodOutcome], *, trained_at: datetime, sample_users: int
) -> int:
    """Refresh the rows for these algorithms; returns how many were written."""
    from sqlalchemy import delete

    from backend.app.db import get_session_factory
    from backend.app.models import ModelRun

    algorithms = [outcome.method for outcome in outcomes]
    with get_session_factory()() as session:
        session.execute(delete(ModelRun).where(ModelRun.algorithm.in_(algorithms)))
        session.add_all(
            ModelRun(
                algorithm=outcome.method,
                params=json.dumps(
                    {
                        **outcome.params,
                        "k": outcome.k,
                        "coverage": outcome.catalogue_coverage,
                        "fallback_rate": outcome.fallback_rate,
                        "evaluated_users": outcome.evaluated_users,
                        "skipped_users": outcome.skipped_users,
                        "rated_pairs": outcome.rated_pairs,
                        "sample_users": sample_users,
                        "runtime_seconds": round(outcome.runtime_seconds, 3),
                        "peak_memory_mb": round(outcome.peak_memory_mb, 1),
                    },
                    sort_keys=True,
                ),
                rmse=outcome.rmse,
                mae=outcome.mae,
                precision_at_k=outcome.precision_at_k,
                recall_at_k=outcome.recall_at_k,
                ndcg_at_k=outcome.ndcg_at_k,
                trained_at=trained_at,
            )
            for outcome in outcomes
        )
        session.commit()
    return len(outcomes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processed = args.processed_dir

    try:
        split = load_split(processed, users=args.users, seed=args.seed)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        item_neighbors = load_neighbors(processed / "item_neighbors.npz")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    user_neighbors, user_seconds, reused = load_or_build_user_neighbors(
        processed, split.train, user_k=args.user_k, rebuild=args.rebuild_user_neighbors
    )

    print(
        f"Evaluating {len(split.evaluated_users):,} users "
        f"({split.n_pairs:,} held-out ratings, {split.n_items:,} movies) "
        f"at k={args.k}, relevance >= {DEFAULT_RELEVANCE_THRESHOLD:g}"
    )
    print(
        f"  item neighbors: k={item_neighbors.indices.shape[1]}, "
        f"trained in {item_neighbor_metadata(processed).get('runtime_seconds', float('nan'))} s"
    )
    source = (
        f"reused from {processed / 'user_neighbors.npz'}"
        if reused
        else f"built in {user_seconds:.1f} s"
    )
    print(f"  user neighbors: k={user_neighbors.indices.shape[1]}, {source}")

    scorers = default_scorers(
        split.train,
        item_neighbors=item_neighbors,
        user_neighbors=user_neighbors,
    )

    started = time.perf_counter()
    outcomes = evaluate_split(split, scorers, k=args.k, chunk_size=args.chunk_size)
    runtime_seconds = time.perf_counter() - started

    print()
    print(render_markdown_table(outcomes))
    print()
    print("| Method | Fallback pairs | Runtime (s) | Peak traced memory (MB) |")
    print("| --- | --- | --- | --- |")
    for outcome in outcomes:
        print(
            f"| {outcome.method} | {outcome.fallback_rate:.2%} | "
            f"{outcome.runtime_seconds:.1f} | {outcome.peak_memory_mb:.0f} |"
        )
    print(f"\nTotal evaluation time: {runtime_seconds:.1f} s")

    trained_at = datetime.now(UTC)
    payload = {
        "generated_at": trained_at.isoformat(),
        "dataset": "MovieLens ml-32m",
        "protocol": {
            "split": "chronological per user, most recent 20% held out",
            "relevance_threshold": DEFAULT_RELEVANCE_THRESHOLD,
            "k": args.k,
            "sample_users": int(split.evaluated_users.size),
            "seed": args.seed,
            "rated_pairs": int(split.n_pairs),
            "model_users": split.n_users,
            "model_items": split.n_items,
        },
        "artifacts": {
            "item_neighbors": item_neighbor_metadata(processed),
            "user_neighbors": {
                "k": int(user_neighbors.indices.shape[1]),
                "reused": reused,
                "build_seconds": round(user_seconds, 2),
            },
        },
        "runtime_seconds": round(runtime_seconds, 2),
        "results": [outcome.as_dict() for outcome in outcomes],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.json}")

    if args.no_db:
        print("Skipped model_runs (--no-db).")
        return 0
    try:
        written = write_model_runs(
            outcomes, trained_at=trained_at, sample_users=int(split.evaluated_users.size)
        )
    except Exception as exc:  # noqa: BLE001 - the report matters more than the table
        print(
            f"warning: could not record model_runs ({type(exc).__name__}: {exc}); "
            "run `uv run alembic upgrade head` and retry",
            file=sys.stderr,
        )
        return 0
    print(f"Recorded {written} rows in model_runs.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
