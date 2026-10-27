#!/usr/bin/env python
"""Precompute item-item neighbors from the processed training matrix.

The API never computes similarities on a request; it loads the artifact this script writes.

Usage:
    uv run python scripts/train_and_precompute.py
    uv run python scripts/train_and_precompute.py --k 50 --block-size 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from scipy import sparse

# Allow `python scripts/train_and_precompute.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.item_cf import build_item_neighbors, save_neighbors  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute item-CF neighbors.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k", type=int, default=50, help="Neighbors kept per movie.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--min-similarity", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processed = args.processed_dir
    matrix_path = processed / "train_matrix.npz"
    map_path = processed / "movie_id_map.parquet"
    if not matrix_path.is_file() or not map_path.is_file():
        print(
            f"error: missing {matrix_path} or {map_path}; run scripts/prepare_data.py first",
            file=sys.stderr,
        )
        return 1

    started = time.perf_counter()
    matrix = sparse.load_npz(matrix_path).tocsr()
    movie_map = pd.read_parquet(map_path)
    if len(movie_map) != matrix.shape[1]:
        print(
            f"error: movie map has {len(movie_map)} rows but the matrix has "
            f"{matrix.shape[1]} columns",
            file=sys.stderr,
        )
        return 1
    print(
        f"Loaded {matrix.shape[0]:,} users x {matrix.shape[1]:,} movies "
        f"({matrix.nnz:,} ratings)"
    )

    neighbors = build_item_neighbors(
        matrix,
        k=args.k,
        block_size=args.block_size,
        min_similarity=args.min_similarity,
    )
    out_path = processed / "item_neighbors.npz"
    save_neighbors(neighbors, out_path)

    with_support = int((neighbors.similarities > 0).any(axis=1).sum())
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "item_item_cosine_mean_centered",
        "k": args.k,
        "block_size": args.block_size,
        "min_similarity": args.min_similarity,
        "n_users": int(matrix.shape[0]),
        "n_items": int(matrix.shape[1]),
        "items_with_neighbors": with_support,
        "nnz": int(matrix.nnz),
        "artifact_bytes": out_path.stat().st_size,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (processed / "item_neighbors.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    stored = int((neighbors.similarities > 0).sum())
    print(
        f"Built top-{args.k} neighbors for {neighbors.counts.size:,} movies "
        f"({stored:,} stored similarities, {with_support:,} movies with support)"
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    metadata_path = processed / "item_neighbors.json"
    print(f"Done in {time.perf_counter() - started:.1f}s — metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
