#!/usr/bin/env python
"""Precompute every artifact the API serves: item neighbors, SVD factors, content features.

The API never computes a similarity, a factor or a feature row on a request. Run this after
``scripts/prepare_data.py`` and before starting the API.

Usage:
    uv run python scripts/train_and_precompute.py
    uv run python scripts/train_and_precompute.py --k 50 --rank 50
    uv run python scripts/train_and_precompute.py --no-svd --no-content
    uv run python scripts/train_and_precompute.py --no-tags
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
from scipy import sparse

# Allow `python scripts/train_and_precompute.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.content import (  # noqa: E402
    build_content_features,
    extract_tag_features,
    save_features,
)
from recsys.item_cf import build_item_neighbors, save_neighbors  # noqa: E402
from recsys.svd import save_model, train_svd  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute recommendation artifacts.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ml-32m"))
    parser.add_argument("--k", type=int, default=50, help="Item neighbors kept per movie.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--min-similarity", type=float, default=1e-6)
    parser.add_argument("--rank", type=int, default=50, help="SVD factors.")
    parser.add_argument("--bias-regularisation", type=float, default=10.0)
    parser.add_argument("--factor-regularisation", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-svd", action="store_true", help="Skip matrix factorization.")
    parser.add_argument(
        "--no-content", action="store_true", help="Skip genre/decade/tag features."
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        help="Build content features from genres and decades only.",
    )
    parser.add_argument("--tag-top", type=int, default=300, help="Most frequent tags kept.")
    parser.add_argument("--tag-min-count", type=int, default=5)
    return parser


def reconstruction_rmse(model, matrix: sparse.csr_matrix, block_size: int = 2048) -> float:
    """RMSE of the model on the ratings it was trained on, measured without densifying."""
    squared_error = 0.0
    seen = 0
    for start in range(0, matrix.shape[0], block_size):
        stop = min(start + block_size, matrix.shape[0])
        block = matrix[start:stop].tocoo()
        if block.nnz == 0:
            continue
        predicted = model.predict_for_users(np.arange(start, stop, dtype="int64"))
        values = predicted[block.row, block.col]
        squared_error += float(np.sum((values - block.data) ** 2))
        seen += int(block.nnz)
    return float(np.sqrt(squared_error / max(seen, 1)))


def write_svd(args, processed: Path, matrix: sparse.csr_matrix) -> dict:
    print(
        f"Training SVD (rank={args.rank}) on {matrix.shape[0]:,} users x "
        f"{matrix.shape[1]:,} movies ..."
    )
    step_started = time.perf_counter()
    model = train_svd(
        matrix,
        rank=args.rank,
        bias_regularisation=args.bias_regularisation,
        factor_regularisation=args.factor_regularisation,
        seed=args.seed,
    )
    train_seconds = time.perf_counter() - step_started
    fit_rmse = reconstruction_rmse(model, matrix)
    path = processed / "svd_model.npz"
    save_model(model, path)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "biased_svd",
        "rank": model.rank,
        "bias_regularisation": model.bias_regularisation,
        "factor_regularisation": model.factor_regularisation,
        "seed": model.seed,
        "n_users": model.n_users,
        "n_items": model.n_items,
        "nnz": int(matrix.nnz),
        "train_rmse": round(fit_rmse, 4),
        "artifact_bytes": path.stat().st_size,
        "runtime_seconds": round(train_seconds, 2),
    }
    (processed / "svd_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"  rank {model.rank}, training RMSE {fit_rmse:.4f}, "
        f"{path.stat().st_size / 1024 / 1024:.1f} MB in {train_seconds:.1f}s"
    )
    return metadata


def write_content(args, processed: Path, raw_dir: Path) -> dict:
    catalog_path = processed / "catalog.parquet"
    if not catalog_path.is_file():
        print(f"  warning: {catalog_path} is missing; skipping content features", file=sys.stderr)
        return {}
    catalog = pd.read_parquet(catalog_path, columns=["movie_id", "genres", "release_year"])

    tags = None
    tags_path = raw_dir / "tags.csv"
    if not args.no_tags and tags_path.is_file():
        print(f"Aggregating tags from {tags_path} ...")
        step_started = time.perf_counter()
        try:
            tags = extract_tag_features(
                tags_path,
                top_tags=args.tag_top,
                min_count=args.tag_min_count,
            )
        except ValueError as exc:
            print(f"  warning: {exc}; continuing without tags", file=sys.stderr)
        else:
            tags.to_parquet(processed / "content_tags.parquet", index=False)
            print(
                f"  {len(tags):,} movie-tag rows, {tags['tag'].nunique()} tags "
                f"in {time.perf_counter() - step_started:.1f}s"
            )

    features = build_content_features(catalog, tags=tags)
    path = processed / "content_features.npz"
    save_features(features, path)
    kinds = {
        "genres": sum(1 for name in features.feature_names if name.startswith("genre:")),
        "decades": sum(1 for name in features.feature_names if name.startswith("decade:")),
        "tags": sum(1 for name in features.feature_names if name.startswith("tag:")),
    }
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "content_genre_decade_tag",
        "movies": features.n_movies,
        "features": features.n_features,
        "feature_kinds": kinds,
        "tag_rows": int(len(tags)) if tags is not None else 0,
        "artifact_bytes": path.stat().st_size,
    }
    (processed / "content_features.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"  {features.n_movies:,} movies x {features.n_features:,} features "
        f"({kinds['genres']} genres, {kinds['decades']} decades, {kinds['tags']} tags), "
        f"{path.stat().st_size / 1024 / 1024:.1f} MB"
    )
    return metadata


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
    neighbor_metadata = {
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
        json.dumps(neighbor_metadata, indent=2), encoding="utf-8"
    )
    stored = int((neighbors.similarities > 0).sum())
    print(
        f"Built top-{args.k} neighbors for {neighbors.counts.size:,} movies "
        f"({stored:,} stored similarities, {with_support:,} movies with support)"
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    svd_metadata: dict = {}
    if not args.no_svd:
        svd_metadata = write_svd(args, processed, matrix)

    content_metadata: dict = {}
    if not args.no_content:
        print("Building content features ...")
        content_metadata = write_content(args, processed, args.raw_dir)

    run_metadata = {
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_users": int(matrix.shape[0]),
        "n_items": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "item_neighbors": neighbor_metadata,
        "svd": svd_metadata,
        "content": content_metadata,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (processed / "precompute.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Done in {time.perf_counter() - started:.1f}s — summary: "
        f"{processed / 'precompute.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
