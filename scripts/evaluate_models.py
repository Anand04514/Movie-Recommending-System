#!/usr/bin/env python
"""Evaluate every method on the chronological split and record the results.

The hybrid weight is chosen on a validation slice carved out of the *training* ratings, using item
neighbors rebuilt on that slice, so neither the weight nor the neighbours peek at the test split.
The final numbers are then measured once, with the chosen weight fixed, on the held-out test pairs.

Reads the processed artifacts, prints a table, writes ``data/processed/evaluation.json`` and
(unless ``--no-db``) refreshes the ``model_runs`` table so the numbers outlive the README.

Usage:
    uv run python scripts/evaluate_models.py
    uv run python scripts/evaluate_models.py --users 5000 --k 10 --chunk-size 256
    uv run python scripts/evaluate_models.py --no-tune --hybrid-weight 0.6
    uv run python scripts/evaluate_models.py --no-db --json /tmp/eval.json
    uv run python scripts/evaluate_models.py --import-only   # refresh model_runs from the report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python scripts/evaluate_models.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.content import load_features  # noqa: E402
from recsys.evaluate import (  # noqa: E402
    DEFAULT_RELEVANCE_THRESHOLD,
    MethodOutcome,
    default_scorers,
    evaluate_split,
    load_split,
    load_validation_split,
    render_markdown_table,
    tune_hybrid_weight,
)
from recsys.hybrid import (  # noqa: E402
    COLLABORATIVE_BASES,
    load_collaborative_base,
    load_weights,
    save_weights,
)
from recsys.item_cf import load_neighbors  # noqa: E402
from recsys.svd import load_model  # noqa: E402
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
        "--import-only",
        action="store_true",
        help=(
            "Refresh model_runs from the saved --json report and exit; nothing is recomputed. "
            "Used on container start, where the report ships as an artifact."
        ),
    )
    parser.add_argument(
        "--rebuild-user-neighbors",
        action="store_true",
        help="Rebuild the user neighbor artifact even if it exists.",
    )
    parser.add_argument(
        "--tune-users", type=int, default=1_500, help="Users in the validation tuning slice."
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Share of each user's training ratings held back for tuning.",
    )
    parser.add_argument("--tune-step", type=float, default=0.1, help="Hybrid weight grid step.")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip validation tuning and use the saved or default hybrid weight.",
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=None,
        help="Use this collaborative share instead of tuning (still evaluated on test).",
    )
    parser.add_argument(
        "--hybrid-base",
        choices=COLLABORATIVE_BASES,
        default=None,
        help="Collaborative model under the hybrid; default is the saved or tuned choice.",
    )
    parser.add_argument("--skip-svd", action="store_true")
    parser.add_argument("--skip-content", action="store_true")
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


def model_movie_ids(processed: Path, expected: int) -> np.ndarray | None:
    """MovieLens ids in model-column order, for aligning catalog content features."""
    path = processed / "movie_id_map.parquet"
    if not path.is_file():
        return None
    mapping = pd.read_parquet(path, columns=["movieId", "movie_index"]).sort_values("movie_index")
    ids = mapping["movieId"].to_numpy(dtype="int64")
    if ids.size != expected:
        print(
            f"warning: movie map has {ids.size} rows but the matrix has {expected} items; "
            "skipping content",
            file=sys.stderr,
        )
        return None
    return ids


def tune(
    processed: Path,
    *,
    item_neighbors,
    user_neighbors,
    content_features,
    movie_ids: np.ndarray,
    args,
) -> tuple[str, float, dict[str, object]]:
    """Pick the hybrid's collaborative base and weight on a validation slice of training.

    Both item-CF and user-CF are tried against the content model, because which one contributes
    more to a blend is an empirical question and it is cheap to answer here instead of guessing.
    The weight grid is scored on held-out training pairs; the test split is never loaded.
    """
    validation = load_validation_split(
        processed,
        users=args.tune_users,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        f"Tuning hybrid on {validation.evaluated_users.size:,} users "
        f"({validation.n_pairs:,} validation pairs, never the test split)"
    )

    started = time.perf_counter()
    scorers = default_scorers(
        validation.train,
        item_neighbors=item_neighbors,
        user_neighbors=user_neighbors,
        content_features=content_features,
        model_movie_ids=movie_ids,
    )
    by_name = {scorer.name: scorer for scorer in scorers}
    weights = np.round(np.arange(0.0, 1.0 + args.tune_step / 2, args.tune_step), 2)

    base_results: list[dict[str, object]] = []
    for base in COLLABORATIVE_BASES:
        if base not in by_name:
            continue
        chosen, trials = tune_hybrid_weight(
            validation,
            collaborative=by_name[base],
            content=by_name["content"],
            weights=[float(weight) for weight in weights],
            k=args.k,
            chunk_size=args.chunk_size,
        )
        best = max(trials, key=lambda outcome: outcome.ndcg_at_k or 0.0)
        base_results.append(
            {
                "base": base,
                "chosen_weight": chosen,
                "best_ndcg_at_k": best.ndcg_at_k,
                "best_precision_at_k": best.precision_at_k,
                "trials": [
                    {
                        "collaborative_weight": trial.params["collaborative_weight"],
                        "ndcg_at_k": trial.ndcg_at_k,
                        "precision_at_k": trial.precision_at_k,
                        "recall_at_k": trial.recall_at_k,
                    }
                    for trial in trials
                ],
            }
        )
        print(
            f"  {base} + content: best weight {chosen:.2f}, "
            f"NDCG@{args.k}={best.ndcg_at_k:.4f}"
        )

    winner = max(base_results, key=lambda result: result["best_ndcg_at_k"] or 0.0)
    payload = {
        "method": "validation slice of training ratings, chronological per user",
        "users": int(validation.evaluated_users.size),
        "rated_pairs": int(validation.n_pairs),
        "validation_fraction": args.validation_fraction,
        "k": args.k,
        "artifacts_from_full_training_split": True,
        "chosen_base": winner["base"],
        "chosen_weight": winner["chosen_weight"],
        "best_ndcg_at_k": winner["best_ndcg_at_k"],
        "best_precision_at_k": winner["best_precision_at_k"],
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "bases": base_results,
    }
    save_weights(
        processed / "hybrid_weights.json",
        weight=float(winner["chosen_weight"]),
        collaborative=str(winner["base"]),
        validation={
            "users": payload["users"],
            "rated_pairs": payload["rated_pairs"],
            "k": args.k,
            "ndcg_at_k": winner["best_ndcg_at_k"],
            "precision_at_k": winner["best_precision_at_k"],
        },
    )
    return str(winner["base"]), float(winner["chosen_weight"]), payload


def _store_model_runs(records: list[dict[str, object]], *, trained_at: datetime) -> int:
    """Replace the rows for these algorithms; returns how many were written."""
    from sqlalchemy import delete

    from backend.app.db import get_session_factory
    from backend.app.models import ModelRun

    algorithms = [str(record["algorithm"]) for record in records]
    with get_session_factory()() as session:
        session.execute(delete(ModelRun).where(ModelRun.algorithm.in_(algorithms)))
        session.add_all(
            ModelRun(
                algorithm=str(record["algorithm"]),
                params=json.dumps(record["params"], sort_keys=True),
                rmse=record["rmse"],
                mae=record["mae"],
                precision_at_k=record["precision_at_k"],
                recall_at_k=record["recall_at_k"],
                ndcg_at_k=record["ndcg_at_k"],
                trained_at=trained_at,
            )
            for record in records
        )
        session.commit()
    return len(records)


def write_model_runs(
    outcomes: list[MethodOutcome], *, trained_at: datetime, sample_users: int
) -> int:
    """Refresh the rows for these algorithms; returns how many were written."""
    return _store_model_runs(
        [
            {
                "algorithm": outcome.method,
                "params": {
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
                "rmse": outcome.rmse,
                "mae": outcome.mae,
                "precision_at_k": outcome.precision_at_k,
                "recall_at_k": outcome.recall_at_k,
                "ndcg_at_k": outcome.ndcg_at_k,
            }
            for outcome in outcomes
        ],
        trained_at=trained_at,
    )


def import_report(path: Path) -> int:
    """Refresh ``model_runs`` from a saved report, recomputing nothing.

    The report ships as an artifact, so a container can show the offline numbers on a database
    that has never run the evaluation itself. Rows are replaced per algorithm, which keeps the
    import idempotent across restarts.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol") or {}
        trained_at = datetime.fromisoformat(payload["generated_at"])
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=UTC)
        records = [
            {
                "algorithm": str(result["method"]),
                "params": {
                    **(result.get("params") or {}),
                    "k": result.get("k"),
                    "coverage": result.get("catalogue_coverage"),
                    "fallback_rate": result.get("fallback_rate"),
                    "evaluated_users": result.get("evaluated_users"),
                    "skipped_users": result.get("skipped_users"),
                    "rated_pairs": result.get("rated_pairs"),
                    "sample_users": protocol.get("sample_users"),
                    "runtime_seconds": round(float(result.get("runtime_seconds") or 0.0), 3),
                    "peak_memory_mb": round(float(result.get("peak_memory_mb") or 0.0), 1),
                    "imported_from": path.name,
                },
                "rmse": result.get("rmse"),
                "mae": result.get("mae"),
                "precision_at_k": result.get("precision_at_k"),
                "recall_at_k": result.get("recall_at_k"),
                "ndcg_at_k": result.get("ndcg_at_k"),
            }
            for result in payload["results"]
        ]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: cannot read {path} ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    if not records:
        print(f"error: {path} holds no results", file=sys.stderr)
        return 1
    try:
        written = _store_model_runs(records, trained_at=trained_at)
    except Exception as exc:  # noqa: BLE001 - a missing table must not stop the container
        print(
            f"error: could not write model_runs ({type(exc).__name__}: {exc}); "
            "run `uv run alembic upgrade head` and retry",
            file=sys.stderr,
        )
        return 1
    print(
        f"Imported {written} rows into model_runs from {path.name} "
        f"(trained {trained_at.date().isoformat()})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processed = args.processed_dir

    if args.hybrid_weight is not None and not 0.0 <= args.hybrid_weight <= 1.0:
        print("error: --hybrid-weight must be between 0 and 1", file=sys.stderr)
        return 1

    if args.import_only:
        return import_report(args.json)

    try:
        split = load_split(processed, users=args.users, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
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

    svd_model = None
    if not args.skip_svd:
        try:
            svd_model = load_model(processed / "svd_model.npz")
        except FileNotFoundError:
            print(
                "warning: no SVD artifact; run scripts/train_and_precompute.py "
                "to include it in the report",
                file=sys.stderr,
            )
        else:
            if svd_model.n_items != split.n_items or svd_model.n_users != split.n_users:
                print(
                    "warning: SVD artifact shape does not match the split; skipping it",
                    file=sys.stderr,
                )
                svd_model = None

    ids = model_movie_ids(processed, split.n_items)
    content_features = None
    if not args.skip_content and ids is not None:
        try:
            content_features = load_features(processed / "content_features.npz")
        except FileNotFoundError:
            print(
                "warning: no content artifact; run scripts/train_and_precompute.py "
                "to include content and hybrid in the report",
                file=sys.stderr,
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

    tuning: dict[str, object] | None = None
    saved_base = load_collaborative_base(processed / "hybrid_weights.json")
    hybrid_base = args.hybrid_base or saved_base
    if args.hybrid_weight is not None:
        hybrid_weight = float(args.hybrid_weight)
        print(
            f"  hybrid: {hybrid_base} + content at weight {hybrid_weight:.2f} "
            "(from --hybrid-weight)"
        )
    elif content_features is not None and not args.no_tune:
        hybrid_base, hybrid_weight, tuning = tune(
            processed,
            item_neighbors=item_neighbors,
            user_neighbors=user_neighbors,
            content_features=content_features,
            movie_ids=ids,
            args=args,
        )
        print(
            f"  hybrid: {hybrid_base} + content at weight {hybrid_weight:.2f} "
            f"(tuned on validation, NDCG@{args.k}={tuning['best_ndcg_at_k']:.4f})"
        )
    else:
        hybrid_weight = load_weights(processed / "hybrid_weights.json")
        origin = "saved" if (processed / "hybrid_weights.json").is_file() else "default"
        print(
            f"  hybrid: {hybrid_base} + content at weight {hybrid_weight:.2f} "
            f"({origin}; not tuned this run)"
        )

    scorers = default_scorers(
        split.train,
        item_neighbors=item_neighbors,
        user_neighbors=user_neighbors,
        svd_model=svd_model,
        content_features=content_features,
        model_movie_ids=ids,
        hybrid_weight=hybrid_weight,
        hybrid_base=hybrid_base,
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
            "hybrid_base": hybrid_base,
            "hybrid_weight": hybrid_weight,
        },
        "artifacts": {
            "item_neighbors": item_neighbor_metadata(processed),
            "svd": (
                json.loads((processed / "svd_model.json").read_text(encoding="utf-8"))
                if (processed / "svd_model.json").is_file()
                else {}
            ),
            "content": (
                json.loads((processed / "content_features.json").read_text(encoding="utf-8"))
                if (processed / "content_features.json").is_file()
                else {}
            ),
            "user_neighbors": {
                "k": int(user_neighbors.indices.shape[1]),
                "reused": reused,
                "build_seconds": round(user_seconds, 2),
            },
        },
        "tuning": tuning,
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
