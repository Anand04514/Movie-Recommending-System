#!/usr/bin/env python
"""Take raw MovieLens files to processed Parquet, ID maps, a sparse matrix, and a manifest.

Usage:
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --raw-dir data/raw/ml-32m --max-users 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/prepare_data.py` from the repo root without installing the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.data import PipelineConfig, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the MovieLens development sample and training artifacts.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ml-32m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-item-ratings", type=int, default=20)
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument(
        "--max-users",
        type=int,
        default=20_000,
        help="Development sample size; use 0 for every eligible user.",
    )
    parser.add_argument(
        "--max-ratings",
        type=int,
        default=3_000_000,
        help="Approximate rating budget; use 0 for no budget.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.out_dir,
        min_item_ratings=args.min_item_ratings,
        min_user_ratings=args.min_user_ratings,
        max_users=args.max_users or None,
        max_ratings=args.max_ratings or None,
        test_fraction=args.test_fraction,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    try:
        run_pipeline(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
