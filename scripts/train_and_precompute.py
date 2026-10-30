#!/usr/bin/env python
"""Precompute every artifact the API serves: item neighbors, SVD factors, content features.

The API never computes a similarity, a factor or a feature row on a request. Run this after
``scripts/prepare_data.py`` and before starting the API.

Usage:
    uv run python scripts/train_and_precompute.py
    uv run python scripts/train_and_precompute.py --k 50 --rank 50
    uv run python scripts/train_and_precompute.py --no-svd --no-content
    uv run python scripts/train_and_precompute.py --no-tags

    # Retrain with the app's own ratings folded in (PLAN 5.11): a film joins the collaborative
    # model once its MovieLens count plus its app ratings reach --min-item-ratings.
    uv run python scripts/train_and_precompute.py --include-app-ratings --sync-db
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

from recsys.augment import merge_app_ratings  # noqa: E402
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
    parser.add_argument(
        "--include-app-ratings",
        action="store_true",
        help=(
            "Merge the ratings of real accounts into the training matrix before building "
            "artifacts, so films the app has collected enough ratings for join the model."
        ),
    )
    parser.add_argument(
        "--min-item-ratings",
        type=int,
        default=20,
        help="Ratings (MovieLens plus app) a film needs to join the collaborative model.",
    )
    parser.add_argument(
        "--sync-db",
        action="store_true",
        help=(
            "Mark films promoted by the merge as in_model in the database. Without it the "
            "artifacts know about the film but the catalogue still labels it as out of model."
        ),
    )
    return parser


def load_app_ratings() -> pd.DataFrame:
    """Every rating written by a real account, not by a mirrored MovieLens demo profile.

    The imports live here so the artifact builders stay importable without a database; this is
    the one step of the retrain that needs the live application's data.
    """
    from sqlalchemy import select

    from backend.app.db import get_session_factory
    from backend.app.models import Rating, User

    with get_session_factory()() as session:
        rows = session.execute(
            select(Rating.user_id, Rating.movie_id, Rating.rating)
            .join(User, User.id == Rating.user_id)
            .where(User.is_seeded.is_(False))
        ).all()
    return pd.DataFrame([tuple(row) for row in rows], columns=["user_id", "movie_id", "rating"])


def load_prior_counts(processed: Path) -> dict[int, int]:
    """MovieLens rating counts, so a rarely rated catalogue film can cross the threshold."""
    catalog_path = processed / "catalog.parquet"
    if not catalog_path.is_file():
        return {}
    catalog = pd.read_parquet(catalog_path, columns=["movie_id", "rating_count"])
    return {
        int(movie_id): int(count)
        for movie_id, count in zip(catalog["movie_id"], catalog["rating_count"], strict=True)
    }


def sync_model_flags(promoted: tuple[int, ...]) -> int:
    """Mark promoted films as part of the model universe in the catalogue table."""
    if not promoted:
        return 0
    from sqlalchemy import update

    from backend.app.db import get_session_factory
    from backend.app.models import Movie

    with get_session_factory()() as session:
        result = session.execute(
            update(Movie)
            .where(Movie.id.in_(promoted), Movie.in_model.is_(False))
            .values(in_model=True)
        )
        session.commit()
    return int(result.rowcount or 0)


def merge_into_artifacts(
    args: argparse.Namespace,
    processed: Path,
    matrix: sparse.csr_matrix,
    movie_map: pd.DataFrame,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Fold the app's ratings in, rewrite the training artifacts, and report what changed."""
    user_map_path = processed / "user_id_map.parquet"
    if not user_map_path.is_file():
        raise FileNotFoundError(
            f"{user_map_path} is missing; run scripts/prepare_data.py before merging app ratings"
        )
    user_map = pd.read_parquet(user_map_path)
    ratings = load_app_ratings()
    print(f"App ratings: {len(ratings):,} from real accounts")

    merged = merge_app_ratings(
        matrix,
        user_map,
        movie_map,
        ratings,
        min_item_ratings=args.min_item_ratings,
        prior_counts=load_prior_counts(processed),
    )
    stats = merged.stats

    sparse.save_npz(processed / "train_matrix.npz", merged.matrix)
    merged.user_map.to_parquet(user_map_path, index=False)
    merged.movie_map.to_parquet(processed / "movie_id_map.parquet", index=False)
    sidecar = {
        "merged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "min_item_ratings": args.min_item_ratings,
        **stats.as_dict(),
    }
    (processed / "app_merge.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(
        f"  merged {stats.ratings_merged:,} ratings from {stats.users_added:,} accounts; "
        f"{stats.duplicates:,} duplicate pairs dropped"
    )
    print(
        f"  {stats.movies_added:,} films promoted to the model, "
        f"{len(stats.below_threshold):,} still below {args.min_item_ratings} ratings"
    )
    if stats.previous_users_stripped or stats.previous_movies_stripped:
        print(
            f"  replaced a previous merge: {stats.previous_users_stripped:,} accounts, "
            f"{stats.previous_movies_stripped:,} films"
        )
    if args.sync_db:
        updated = sync_model_flags(stats.promoted)
        print(f"  marked {updated:,} films in_model in the database")
    elif stats.promoted:
        print(
            "  note: --sync-db was not passed; the catalogue still labels these films out of model"
        )

    return merged.matrix, merged.movie_map


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
    if args.include_app_ratings:
        try:
            matrix, movie_map = merge_into_artifacts(args, processed, matrix, movie_map)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - a missing schema or database is an operator error
            print(
                f"error: could not read app ratings ({exc}); is the database migrated and "
                "reachable? Without --include-app-ratings this step is skipped.",
                file=sys.stderr,
            )
            return 1
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
