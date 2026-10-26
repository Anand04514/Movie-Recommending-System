"""MovieLens data pipeline: chunked loading, filters, splits, ID maps, sparse matrix.

Design constraints from ``PLAN.md``:

- ``data/raw/`` is read-only; every artifact goes to ``data/processed/``.
- ``ratings.csv`` is never read whole: one streaming pass computes counts, a second writes the
  filtered sample to Parquet.
- The model universe is the filtered subset (movies with enough ratings, sampled users), while the
  catalog keeps every movie so search can find all of them.
- Splits are chronological per user; the sparse matrix is built from the training split only, so
  the test set never leaks into training.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from recsys.titles import parse_title

REQUIRED_RAW_FILES = ("movies.csv", "links.csv", "ratings.csv")
RAW_FILES = ("ratings.csv", "movies.csv", "tags.csv", "links.csv")

RATING_COLUMNS = ("userId", "movieId", "rating", "timestamp")
RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}
RATING_SCHEMA = pa.schema(
    [
        ("userId", pa.int32()),
        ("movieId", pa.int32()),
        ("rating", pa.float32()),
        ("timestamp", pa.int64()),
    ]
)

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


# --------------------------------------------------------------------------------------
# Raw input
# --------------------------------------------------------------------------------------
def verify_raw_dir(raw_dir: Path, *, require_all_four: bool = False) -> None:
    """Fail early, with instructions, when the MovieLens files are missing."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"raw data directory not found: {raw_dir}\n"
            f"Download {DOWNLOAD_URL} and extract it so that "
            f"{raw_dir / 'ratings.csv'} exists."
        )
    expected = RAW_FILES if require_all_four else REQUIRED_RAW_FILES
    missing = [name for name in expected if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing MovieLens file(s) in {raw_dir}: {', '.join(missing)}\n"
            f"Download {DOWNLOAD_URL} and extract all four CSVs into that directory."
        )


def read_links(raw_dir: Path) -> pd.DataFrame:
    """Read ``links.csv`` with nullable integer ids (many rows have no TMDB id)."""
    links = pd.read_csv(
        Path(raw_dir) / "links.csv",
        dtype={"movieId": "int32", "imdbId": "int32", "tmdbId": "Int64"},
    )
    if links["movieId"].duplicated().any():
        raise ValueError("links.csv contains duplicate movieId values")
    return links


def read_movies(raw_dir: Path) -> pd.DataFrame:
    """Read the full catalog (``movies.csv`` + ``links.csv``) with parsed titles.

    Every movie is kept, including those with too few ratings for the model universe, because
    search must find the entire catalog.
    """
    path = Path(raw_dir) / "movies.csv"
    movies = pd.read_csv(path, dtype={"movieId": "int32", "title": "string", "genres": "string"})
    if movies["movieId"].duplicated().any():
        raise ValueError("movies.csv contains duplicate movieId values")
    if movies["title"].isna().any():
        raise ValueError("movies.csv contains rows without a title")

    parsed = [parse_title(title) for title in movies["title"]]
    catalog = pd.DataFrame(
        {
            "movie_id": movies["movieId"].astype("int32"),
            "title_raw": movies["title"].astype("string"),
            "title": [p.title for p in parsed],
            "title_display": [p.display_title for p in parsed],
            "title_search": [p.search_title for p in parsed],
            "release_year": pd.array([p.release_year for p in parsed], dtype="Int16"),
            "genres": movies["genres"].fillna("").astype("string"),
            "source": "movielens",
        }
    )

    links = read_links(Path(raw_dir))
    catalog = catalog.merge(
        links.rename(
            columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}
        ),
        on="movie_id",
        how="left",
        validate="one_to_one",
    )
    return catalog


def iter_rating_chunks(path: Path, chunksize: int = 2_000_000) -> Iterator[pd.DataFrame]:
    """Yield validated rating chunks with compact dtypes."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    reader = pd.read_csv(
        Path(path),
        usecols=list(RATING_COLUMNS),
        dtype=RATING_DTYPES,
        chunksize=chunksize,
    )
    for chunk in reader:
        _validate_rating_chunk(chunk)
        yield chunk


def _validate_rating_chunk(chunk: pd.DataFrame) -> None:
    if chunk.empty:
        return
    ratings = chunk["rating"]
    if ratings.isna().any():
        raise ValueError("ratings.csv contains missing rating values")
    if ((ratings < 0.5) | (ratings > 5.0)).any():
        raise ValueError("ratings.csv contains values outside the 0.5-5.0 range")
    doubled = ratings.to_numpy(dtype="float64") * 2
    if not np.allclose(doubled, np.round(doubled)):
        raise ValueError("ratings.csv contains values that are not half-star steps")
    if (chunk["userId"] <= 0).any() or (chunk["movieId"] <= 0).any():
        raise ValueError("ratings.csv contains non-positive user or movie ids")


# --------------------------------------------------------------------------------------
# Streaming statistics
# --------------------------------------------------------------------------------------
@dataclass
class RatingScan:
    """Aggregates computed in one streaming pass over ``ratings.csv``."""

    n_ratings: int
    user_counts: pd.Series
    movie_counts: pd.Series
    movie_sums: pd.Series
    rating_histogram: pd.Series

    @property
    def n_users(self) -> int:
        return int(self.user_counts.size)

    @property
    def n_movies(self) -> int:
        return int(self.movie_counts.size)

    def summary(self) -> dict[str, object]:
        histogram = {f"{value:g}": int(count) for value, count in self.rating_histogram.items()}
        return {
            "ratings": self.n_ratings,
            "users": self.n_users,
            "movies": self.n_movies,
            "rating_histogram": dict(sorted(histogram.items(), key=lambda item: float(item[0]))),
        }


def scan_ratings(path: Path, chunksize: int = 2_000_000) -> RatingScan:
    """Count ratings per user and per movie without loading the file into memory."""
    user_counts = pd.Series(dtype="int64")
    movie_counts = pd.Series(dtype="int64")
    movie_sums = pd.Series(dtype="float64")
    histogram = pd.Series(dtype="int64")
    total = 0

    for chunk in iter_rating_chunks(path, chunksize):
        total += len(chunk)
        user_counts = user_counts.add(chunk["userId"].value_counts(), fill_value=0)
        grouped = chunk.groupby("movieId", observed=True)["rating"].agg(["count", "sum"])
        movie_counts = movie_counts.add(grouped["count"].astype("int64"), fill_value=0)
        movie_sums = movie_sums.add(grouped["sum"].astype("float64"), fill_value=0)
        histogram = histogram.add(chunk["rating"].value_counts(), fill_value=0)

    return RatingScan(
        n_ratings=total,
        user_counts=user_counts.astype("int64").sort_index(),
        movie_counts=movie_counts.astype("int64").sort_index(),
        movie_sums=movie_sums.astype("float64").sort_index(),
        rating_histogram=histogram.astype("int64").sort_index(),
    )


def model_movie_ids(scan: RatingScan, min_item_ratings: int = 20) -> np.ndarray:
    """MovieLens ids with at least ``min_item_ratings`` ratings."""
    if min_item_ratings < 1:
        raise ValueError("min_item_ratings must be at least 1")
    return scan.movie_counts[scan.movie_counts >= min_item_ratings].index.to_numpy(dtype="int64")


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def select_sample_users(
    scan: RatingScan,
    *,
    min_user_ratings: int = 20,
    max_users: int | None = 20_000,
    max_ratings: int | None = 3_000_000,
    seed: int = 42,
) -> np.ndarray:
    """Choose a reproducible set of users for the development sample.

    Users are filtered by rating count, then sampled with a seeded RNG. If ``max_ratings`` is set,
    whole users are kept until the budget is reached so no user is split.
    """
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive or None")
    if max_ratings is not None and max_ratings < 1:
        raise ValueError("max_ratings must be positive or None")

    eligible = scan.user_counts[scan.user_counts >= min_user_ratings]
    eligible_ids = np.sort(eligible.index.to_numpy(dtype="int64"))
    if eligible_ids.size == 0:
        raise ValueError(f"no users have at least {min_user_ratings} ratings")

    rng = np.random.default_rng(seed)
    if max_users is not None and eligible_ids.size > max_users:
        order = rng.choice(eligible_ids, size=max_users, replace=False)
    else:
        order = eligible_ids.copy()
        rng.shuffle(order)

    if max_ratings is not None:
        counts = eligible.to_numpy(dtype="int64")
        position = {int(user_id): index for index, user_id in enumerate(eligible.index)}
        chosen: list[int] = []
        running = 0
        for user_id in order:
            count = int(counts[position[int(user_id)]])
            if chosen and running + count > max_ratings:
                break
            chosen.append(int(user_id))
            running += count
        sample = np.asarray(chosen, dtype="int64")
    else:
        sample = order.astype("int64")

    return np.sort(sample).astype("int64")


def write_filtered_ratings(
    ratings_path: Path,
    out_path: Path,
    *,
    user_ids: Iterable[int],
    movie_ids: Iterable[int],
    chunksize: int = 2_000_000,
) -> int:
    """Second streaming pass: keep ratings of sampled users on model-universe movies."""
    users = np.sort(np.asarray(list(user_ids), dtype="int64"))
    movies = np.sort(np.asarray(list(movie_ids), dtype="int64"))
    if users.size == 0 or movies.size == 0:
        raise ValueError("cannot filter ratings with an empty user or movie set")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    writer = pq.ParquetWriter(out_path, RATING_SCHEMA, compression="zstd")
    try:
        for chunk in iter_rating_chunks(ratings_path, chunksize):
            kept = chunk[chunk["userId"].isin(users) & chunk["movieId"].isin(movies)]
            if kept.empty:
                continue
            table = pa.Table.from_pandas(kept, schema=RATING_SCHEMA, preserve_index=False)
            writer.write_table(table)
            rows += len(kept)
    finally:
        writer.close()
    return rows


# --------------------------------------------------------------------------------------
# Split and ID maps
# --------------------------------------------------------------------------------------
def chronological_split(
    ratings: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent ``test_fraction`` of ratings as the test set."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1 exclusive")
    if ratings.empty:
        raise ValueError("cannot split an empty ratings frame")

    ordered = ratings.sort_values(
        ["userId", "timestamp", "movieId"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = ordered.groupby("userId", sort=False)["movieId"].transform("size").to_numpy()
    rank_from_end = ordered.groupby("userId", sort=False).cumcount(ascending=False).to_numpy()

    n_test = np.floor(sizes * test_fraction).astype("int64")
    # Keep at least one training row per user, even for users with a single rating.
    n_test = np.where(sizes > 1, np.clip(n_test, 1, sizes - 1), 0)
    is_test = rank_from_end < n_test

    train = ordered.loc[~is_test].reset_index(drop=True)
    test = ordered.loc[is_test].reset_index(drop=True)
    return train, test


def build_id_map(ids: Iterable[int], id_column: str, index_column: str) -> pd.DataFrame:
    """Map raw ids to the compact 0..n-1 indexes used by the sparse matrix."""
    unique = np.sort(np.unique(np.asarray(list(ids), dtype="int64")))
    if unique.size == 0:
        raise ValueError("cannot build an id map from no ids")
    return pd.DataFrame({id_column: unique, index_column: np.arange(unique.size, dtype="int32")})


def apply_id_map(
    ratings: pd.DataFrame,
    user_map: pd.DataFrame,
    movie_map: pd.DataFrame,
    *,
    user_column: str = "userId",
    movie_column: str = "movieId",
) -> pd.DataFrame:
    """Add ``user_index``/``movie_index`` columns, raising if a raw id is unmapped."""
    result = ratings.merge(
        user_map.rename(columns={"userId": user_column, "user_index": "user_index"}),
        on=user_column,
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        movie_map.rename(columns={"movieId": movie_column, "movie_index": "movie_index"}),
        on=movie_column,
        how="left",
        validate="many_to_one",
    )
    missing_users = result["user_index"].isna()
    missing_movies = result["movie_index"].isna()
    if missing_users.any():
        raise KeyError(f"{int(missing_users.sum())} ratings reference unmapped users")
    if missing_movies.any():
        raise KeyError(f"{int(missing_movies.sum())} ratings reference unmapped movies")
    return result


def build_csr(
    ratings: pd.DataFrame,
    *,
    n_users: int,
    n_movies: int,
) -> sparse.csr_matrix:
    """Build a users x movies CSR matrix of raw ratings from the training split."""
    if ratings[["user_index", "movie_index"]].duplicated().any():
        raise ValueError("duplicate (user, movie) pairs: the matrix would be ambiguous")
    rows = ratings["user_index"].to_numpy(dtype="int32")
    cols = ratings["movie_index"].to_numpy(dtype="int32")
    values = ratings["rating"].to_numpy(dtype="float32")
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(n_users, n_movies), dtype="float32")
    if matrix.nnz != len(ratings):
        raise ValueError("CSR construction dropped values; check indexes and shape")
    return matrix


def matrix_stats(matrix: sparse.spmatrix) -> dict[str, object]:
    """Counts and memory figures used in the manifest and CLI output."""
    n_users, n_movies = matrix.shape
    cells = max(1, n_users * n_movies)
    memory = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return {
        "shape": [int(n_users), int(n_movies)],
        "nnz": int(matrix.nnz),
        "density": matrix.nnz / cells,
        "memory_bytes": memory,
        "memory_mb": round(memory / (1024 * 1024), 2),
    }


# --------------------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    raw_dir: Path
    processed_dir: Path
    min_item_ratings: int = 20
    min_user_ratings: int = 20
    max_users: int | None = 20_000
    max_ratings: int | None = 3_000_000
    test_fraction: float = 0.2
    seed: int = 42
    chunk_size: int = 2_000_000


def _ensure_outside_raw(raw_dir: Path, processed_dir: Path) -> None:
    raw = Path(raw_dir).resolve()
    processed = Path(processed_dir).resolve()
    if processed == raw or raw in processed.parents:
        raise ValueError("processed output must not be inside the raw data directory")


def run_pipeline(config: PipelineConfig, *, log=print) -> dict[str, object]:
    """Take raw MovieLens files to a processed sample, split, maps, CSR and manifest."""
    started = time.perf_counter()
    raw_dir = Path(config.raw_dir)
    out_dir = Path(config.processed_dir)
    verify_raw_dir(raw_dir)
    _ensure_outside_raw(raw_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading catalog from {raw_dir} ...")
    catalog = read_movies(raw_dir)

    log(f"Scanning {raw_dir / 'ratings.csv'} in {config.chunk_size:,}-row chunks ...")
    scan_started = time.perf_counter()
    scan = scan_ratings(raw_dir / "ratings.csv", config.chunk_size)
    log(
        f"  {scan.n_ratings:,} ratings from {scan.n_users:,} users on {scan.n_movies:,} movies "
        f"({time.perf_counter() - scan_started:.1f}s)"
    )

    eligible_movies = model_movie_ids(scan, config.min_item_ratings)
    users = select_sample_users(
        scan,
        min_user_ratings=config.min_user_ratings,
        max_users=config.max_users,
        max_ratings=config.max_ratings,
        seed=config.seed,
    )
    log(
        f"  {len(eligible_movies):,} movies with >= {config.min_item_ratings} ratings; "
        f"sampled {len(users):,} users"
    )

    catalog = catalog.merge(
        pd.DataFrame(
            {
                "movie_id": scan.movie_counts.index.to_numpy(dtype="int64"),
                "rating_count": scan.movie_counts.to_numpy(dtype="int64"),
                "rating_sum": scan.movie_sums.reindex(scan.movie_counts.index).to_numpy(),
            }
        ),
        on="movie_id",
        how="left",
    )
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype("int64")
    catalog["rating_sum"] = catalog["rating_sum"].fillna(0.0).astype("float64")
    catalog["rating_mean"] = (catalog["rating_sum"] / catalog["rating_count"].where(
        catalog["rating_count"] > 0
    )).astype("float64")

    log("Writing filtered ratings sample ...")
    sample_path = out_dir / "ratings_sample.parquet"
    sample_rows = write_filtered_ratings(
        raw_dir / "ratings.csv",
        sample_path,
        user_ids=users,
        movie_ids=eligible_movies,
        chunksize=config.chunk_size,
    )
    log(f"  {sample_rows:,} ratings kept")

    sample = pd.read_parquet(sample_path)
    if sample.empty:
        raise ValueError("the sample is empty; relax the filters or budgets")

    movie_universe = np.sort(sample["movieId"].unique().astype("int64"))
    catalog["in_model"] = catalog["movie_id"].isin(movie_universe)
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)

    train, test = chronological_split(sample, config.test_fraction)
    user_map = build_id_map(sample["userId"], "userId", "user_index")
    movie_map = build_id_map(movie_universe, "movieId", "movie_index")
    train = apply_id_map(train, user_map, movie_map)
    test = apply_id_map(test, user_map, movie_map)

    if train.empty:
        raise ValueError("the training split is empty; check test_fraction")

    matrix = build_csr(train, n_users=len(user_map), n_movies=len(movie_map))
    sparse.save_npz(out_dir / "train_matrix.npz", matrix)

    train.to_parquet(out_dir / "ratings_train.parquet", index=False)
    test.to_parquet(out_dir / "ratings_test.parquet", index=False)
    user_map.to_parquet(out_dir / "user_id_map.parquet", index=False)
    movie_map.to_parquet(out_dir / "movie_id_map.parquet", index=False)

    stats = matrix_stats(matrix)
    histogram = {
        f"{value:g}": int(count) for value, count in scan.rating_histogram.items()
    }
    files = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "megabytes": round(path.stat().st_size / (1024 * 1024), 2),
            }

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "name": "MovieLens ml-32m",
            "url": DOWNLOAD_URL,
            "raw_dir": str(raw_dir),
        },
        "config": {
            "min_item_ratings": config.min_item_ratings,
            "min_user_ratings": config.min_user_ratings,
            "max_users": config.max_users,
            "max_ratings": config.max_ratings,
            "test_fraction": config.test_fraction,
            "seed": config.seed,
            "chunk_size": config.chunk_size,
        },
        "raw": {
            "ratings": scan.n_ratings,
            "users": scan.n_users,
            "movies": scan.n_movies,
            "catalog_movies": len(catalog),
            "rating_histogram": dict(sorted(histogram.items(), key=lambda item: float(item[0]))),
        },
        "model_universe": {
            "eligible_movies": int(len(eligible_movies)),
            "user_filter": int((scan.user_counts >= config.min_user_ratings).sum()),
            "sampled_users": int(len(users)),
            "movies_rated_in_sample": int(len(movie_universe)),
        },
        "sample": {
            "ratings": int(len(sample)),
            "users": int(len(user_map)),
            "movies": int(len(movie_map)),
            "rating_min": float(sample["rating"].min()),
            "rating_max": float(sample["rating"].max()),
            "rating_mean": round(float(sample["rating"].mean()), 4),
        },
        "split": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_fraction": config.test_fraction,
        },
        "matrix": stats,
        "files": files,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("")
    log(f"Catalog:        {len(catalog):,} movies ({int(catalog['in_model'].sum()):,} in model)")
    log(
        f"Sample:         {len(sample):,} ratings, {len(user_map):,} users, "
        f"{len(movie_map):,} movies"
    )
    log(f"Split:          {len(train):,} train / {len(test):,} test")
    log(f"Matrix:         {stats['shape'][0]:,} x {stats['shape'][1]:,}, nnz={stats['nnz']:,}, "
        f"density={stats['density']:.5%}, memory={stats['memory_mb']} MB")
    log("Rating distribution: " + ", ".join(f"{k}*={v:,}" for k, v in sorted(
        histogram.items(), key=lambda item: float(item[0])
    )))
    log(f"Artifacts in {out_dir} — manifest: {out_dir / 'manifest.json'}")
    return manifest
