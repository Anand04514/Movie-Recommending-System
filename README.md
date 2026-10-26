# Movie Recs

A full-stack movie recommender built on the MovieLens **ml-32m** dataset: 32 million ratings,
87,585 movies, and a catalog that stays searchable for every title — including movies too new or
too obscure for the model.

FastAPI + SQLAlchemy backend, React + Vite frontend, and a pure-Python `recsys/` package that holds
all the data and ML logic (no web or database imports, so it can be tested alone).

**Current status:** Phase 0 (setup) and Phase 1.1–1.6 (data pipeline) are complete and tested.
Search, auth, ratings, and recommendations arrive in Phase 2.

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/) (installs Python 3.12 itself), Node 20+, and
optionally PostgreSQL 14+ with `pg_trgm`. SQLite works for early local development.

```bash
uv sync

# Terminal 1 — API (http://127.0.0.1:8000, docs at /docs)
uv run uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — UI (http://127.0.0.1:5173)
cd frontend
npm install
npm run dev
```

The UI calls `/api/health` through Vite's dev proxy and shows the API and database status.
Copy `.env.example` to `.env` to point `DATABASE_URL` at PostgreSQL; the backend enables `pg_trgm`
automatically at startup when the dialect is PostgreSQL.

## Dataset

Download **ml-32m** from <https://grouplens.org/datasets/movielens/> and extract it so that
`data/raw/ml-32m/ratings.csv` exists (about 228 MB zipped, 837 MB for `ratings.csv`).

| File | Rows (verified) |
| --- | --- |
| `ratings.csv` | 32,000,204 ratings from 200,948 users |
| `movies.csv` | 87,585 movies |
| `links.csv` | 87,585 rows (IMDb + TMDB ids) |
| `tags.csv` | 2,000,072 free-text tags |

Verified on this checkout: 200,948 distinct users, 84,432 movies with at least one rating, and
5,357 catalog titles released in 2021 or later (e.g. *Dune* 2021, *Everything Everywhere All at
Once* 2022).

`data/raw/` is **read-only**: nothing in the pipeline writes to it, and neither raw nor processed
data is committed. The newest release in ml-32m is from 2023, so newer films require the TMDB
ingestion planned for Phase 5.

## Build the model inputs

```bash
uv run python scripts/prepare_data.py
```

Defaults follow `PLAN.md`: movies with **at least 20 ratings**, users with at least 20 ratings, a
development sample of **20,000 users / ~3M ratings** (seed 42), and a **chronological per-user 80/20
split**. Output in `data/processed/`:

| Artifact | Contents |
| --- | --- |
| `catalog.parquet` | Every movie: normalized titles, year, genres, IMDb/TMDB ids, rating count/mean, `in_model` |
| `ratings_sample.parquet` | Filtered ratings for the sampled users |
| `ratings_train.parquet`, `ratings_test.parquet` | Chronological split, with `user_index`/`movie_index` |
| `user_id_map.parquet`, `movie_id_map.parquet` | Raw ids ↔ compact integers |
| `train_matrix.npz` | SciPy CSR users × movies matrix (training rows only) |
| `manifest.json` | Filters, sample sizes, rating histogram, sparsity, memory, timings |

Default run on the full ml-32m file (15.8 s, no model code involved):

```
23,350 movies with >= 20 ratings; sampled 18,632 users (3M-rating budget)
Sample:  2,974,026 ratings, 18,632 users, 22,973 movies
Split:   2,386,395 train / 587,631 test
Matrix:  18,632 x 22,973, nnz=2,386,395, density=0.5575%, 18.28 MB
Catalog: 87,585 movies, 22,973 marked in_model
```

Useful flags: `--max-users 0 --max-ratings 0` (no sampling; needs more RAM),
`--chunk-size 1000000`, `--min-item-ratings`, `--test-fraction`, `--seed`.

### Design decisions

- `ratings.csv` is streamed twice in bounded chunks — never loaded whole, never inserted into
  PostgreSQL. The database will hold app and demo users only.
- The **catalog is all 87,585 movies**; the **model universe** is the filtered subset. Movies outside
  it stay searchable and ratable.
- The split is **chronological per user** (most recent 20% is test); the CSR matrix is built from the
  training split only, so evaluation cannot leak.
- Missing ratings are never treated as zeros — the matrix is sparse and models mean-center per user.
- Titles are normalized to `Matrix, The (1999)` → title / display `The Matrix` / search `the matrix`,
  so both "the matrix" and "matrix" match. The real catalog contains one year-only title,
  `" (2019)"`, which is preserved instead of crashing ingestion.

## Tests and lint

```bash
uv run pytest        # 86 tests: title parsing, pipeline, API
uv run ruff check .
```

## Layout

```
recsys/          pure ML/data package (titles, chunked loading, filters, splits, ID maps)
backend/app/     FastAPI app, settings, database helpers (routers land in Phase 2)
frontend/        React + Vite shell
scripts/         prepare_data.py (seed_db.py and training scripts come later)
data/raw/        downloaded MovieLens files — never modified or committed
data/processed/  derived Parquet, CSR matrix, ID maps, manifest — never committed
tests/           tests for recsys/
```

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Setup: repo, venv, FastAPI hello-world, Vite shell, PostgreSQL/`pg_trgm`, lint, tests | Done |
| 1 | Data pipeline: titles, catalog, chunked loading, filters, CSR, chronological split, manifest | 1.1–1.6 done (seeding demo users needs the Phase 2 schema) |
| 2 | Baselines + item-CF, JWT auth, search, ratings, recommendations | Next |
| 3 | Metrics, evaluation on the chronological split, `model_runs`, UI polish | Planned |
| 4 | Filters, richer explanations, algorithm comparison, accuracy dashboard, posters | Planned |
| 5 | SVD + fold-in, content-based, hybrid, cold start, serendipity, feedback loop, TMDB | Planned |
| 6 | Deployment, README results table, demo recording, release tag | Planned |

See [PLAN.md](PLAN.md) for the full development plan and [AGENTS.md](AGENTS.md) for the rules the
build follows.
