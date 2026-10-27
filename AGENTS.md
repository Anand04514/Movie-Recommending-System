# Agent rules — movie recommender

Read `PLAN.md` before starting a task. Implement one phase (or a numbered sub-set of tasks) at a time.
Write tests for everything you add, run them, and stop when the acceptance criteria for that phase
are met.

## Stack

- Backend: FastAPI (Python 3.11+), SQLAlchemy, Alembic, PostgreSQL with `pg_trgm` (SQLite for early
  local development only)
- ML: pandas, NumPy, SciPy sparse matrices, scikit-learn; optional `implicit` and FAISS
- Auth: JWT + bcrypt
- Frontend: React + Vite
- Tests: pytest; lint: ruff

## Layout

```
recsys/     pure ML/data package — no web, DB, or FastAPI imports
backend/    FastAPI app, routers, models, tests
frontend/   React + Vite shell
scripts/    prepare_data.py, seed_db.py, train_and_precompute.py
data/raw/   downloaded MovieLens files — READ-ONLY
data/processed/  derived Parquet, sparse matrices, ID maps, manifest
tests/      tests for recsys/
```

## Hard rules

- **Never modify `data/raw/`.** All cleaning writes to `data/processed/`.
- **Never load the full `ratings.csv` with a plain `pd.read_csv`.** Read in chunks and write
  Parquet/sparse artifacts.
- **Never build a dense user-item matrix.** Use `scipy.sparse`.
- **Never treat missing ratings as zero.** Mean-center per user and compute similarity on
  co-rated items only.
- **Always use chronological per-user splits.** No random splits of ratings.
- Ratings are 0.5–5.0 in half-star steps; validate, never assume integers.
- Always exclude movies a user has already rated from their recommendations.
- Precompute similarities/factors offline; never recompute on a request.
- `recsys/` must stay importable without FastAPI, SQLAlchemy, or a database.

## Memory limits

- 16 GB RAM: full development workflow is comfortable.
- 8 GB RAM: stay on the development subset (~20k users / ≤3M ratings) and process ratings in
  chunks of at most 2M rows.

## Before finishing any task

Run `uv run pytest` and `uv run ruff check .`; both must pass. When the frontend changes, also run
`cd frontend && npm test` and `npm run build`. Commit only after the phase's acceptance criteria
pass, and never commit files under `data/raw/` or `data/processed/`.
