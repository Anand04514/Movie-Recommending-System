# Movie Recs

A full-stack movie recommender built on the MovieLens **ml-32m** dataset: 32 million ratings,
87,585 movies, and a catalog that stays searchable for every title — including movies too new or
too obscure for the model.

FastAPI + SQLAlchemy + Alembic backend, React + Vite frontend, and a pure-Python `recsys/` package
that holds all the data and ML logic (no web or database imports, so it can be tested alone).

**Current status:** Phases 0–4 are complete and tested. A new user can sign up, search the real
catalog, rate films in half-star steps, filter their picks by genre, year and predicted score, read
the ratings behind each recommendation, compare item-based and user-based lists side by side, and
see every method's offline accuracy charted from the database. Posters and synopses appear
automatically once a free TMDB key is configured. The advanced models (SVD with fold-in,
content-based, hybrid) are Phase 5.

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/) (installs Python 3.12 itself), Node 20+, and
optionally PostgreSQL 14+ with `pg_trgm`. SQLite works for local development.

```bash
uv sync

# 1. Raw MovieLens files -> Parquet, id maps, CSR matrix, chronological split (~16 s)
uv run python scripts/prepare_data.py

# 2. Train the item-item neighbors the API serves from (~15 s)
uv run python scripts/train_and_precompute.py

# 3. Create the schema and load the catalog + 500 MovieLens demo users
uv run alembic upgrade head
uv run python scripts/seed_db.py

# 4. Run both servers
uv run uvicorn backend.app.main:app --reload --port 8000   # API + docs at /docs
cd frontend && npm install && npm run dev                  # UI on http://127.0.0.1:5173
```

Copy `.env.example` to `.env` to point `DATABASE_URL` at PostgreSQL; the backend enables `pg_trgm`
automatically at startup when the dialect is PostgreSQL. `JWT_SECRET` must be replaced in any
deployment. `scripts/seed_db.py --demo-users 0` seeds the catalog only; `--refresh-demo` rebuilds
the demo users.

## What you can do

| Feature | Notes |
| --- | --- |
| Sign up / sign in | JWT access tokens, bcrypt hashing, session restored from local storage |
| Search any movie | Trigram (PostgreSQL) or word-boundary prefix ranking (SQLite); all 87,585 titles |
| Rate films | Half-star widget (0.5–5.0), upsert and delete, "My ratings" library with the average |
| For you | Item-based CF with a reason line, the runner-up ratings behind it, and a damped predicted score |
| Filters | Genre chips, a release-year range and a minimum predicted rating, applied to a wider scored pool |
| Compare | Item-CF vs user-CF for the same profile, with picks both lists agree on marked “Both” |
| Accuracy | RMSE, MAE, precision/recall/NDCG@10 and coverage per method, charted from `model_runs` |
| Posters and details | Titles open a details dialog; posters and synopses load lazily from TMDB and cache locally |
| Cold start | Users with no usable signal get the popularity baseline plus an onboarding prompt |

The demo users (`password_hash="!"`, so they can never sign in) exist to give the catalog rating
history; recommendation serving never reads them at request time, since neighbors are precomputed.

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
Once* 2022, *The Matrix Resurrections* 2021).

`data/raw/` is **read-only**: nothing in the pipeline writes to it, and neither raw nor processed
data is committed. The newest release in ml-32m is from 2023, so newer films require the TMDB
ingestion planned for Phase 5.

## The pipeline

### 1. `scripts/prepare_data.py` — raw files to model inputs

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

Default run on the full ml-32m file (15.8 s):

```
23,350 movies with >= 20 ratings; sampled 18,632 users (3M-rating budget)
Sample:  2,974,026 ratings, 18,632 users, 22,973 movies
Split:   2,386,395 train / 587,631 test
Matrix:  18,632 x 22,973, nnz=2,386,395, density=0.5575%, 18.28 MB
Catalog: 87,585 movies, 22,973 marked in_model
```

Useful flags: `--max-users 0 --max-ratings 0` (no sampling; needs more RAM),
`--chunk-size 1000000`, `--min-item-ratings`, `--test-fraction`, `--seed`.

### 2. `scripts/train_and_precompute.py` — item neighbors

Cosine similarity on the mean-centered training matrix, top-50 neighbors per movie, computed in
bounded blocks (a dense item × item matrix is never materialized) and saved to
`data/processed/item_neighbors.npz` with a JSON sidecar:

```
k=50 · 22,973 items · 22,013 with neighbors · 2,386,395 nnz · 5.5 MB · 14.2 s
```

### 3. `scripts/evaluate_models.py` — the head-to-head numbers

Builds (once) and caches `data/processed/user_neighbors.npz` for user-based CF, then scores five
methods on the same held-out pairs and writes `data/processed/evaluation.json` plus one row per
method in the `model_runs` table. See [Results](#results).

### 4. `scripts/seed_db.py` — catalog and demo users

The database stores the **catalog** (all 87,585 movies) and **demo users** only; the 32M training
ratings stay on disk as Parquet and a CSR matrix.

```
movies: 87,585 (22,973 in_model) · users: 500 demo · ratings in DB: 63,049
```

## How recommendations work

`backend/app/recommendations.py` loads the precomputed neighbors once and folds the signed-in user's
ratings in at request time, so **no similarity is ever computed on a request**:

1. Ratings for movies inside the model universe are mapped to their compact indexes.
2. `recsys.item_cf.recommend_from_ratings` scores candidates as
   `Σ similarity × (rating − user mean)` over the user's rated items — sparse, so only co-rated
   items contribute, and missing ratings are never treated as zeros.
3. Scores are turned into a predicted rating by shrinking toward the user's mean with `damping=4.0`
   and clamping to 0.5–5.0, so a couple of neighbors cannot push every pick to 5.0.
4. Movies the user already rated are always excluded; candidates need positive evidence, so rating
   something 1★ never *causes* a recommendation.
5. The reason line names the rated movie with the strongest positive contribution
   ("Because you rated The Godfather 5★"); the response also carries the next two contributors with
   their ratings and shared genres, which the card shows as "Also driven by".

User-based CF is served the same way from a precomputed user-neighbor table: the live user's vector
is folded in at request time (`recsys.user_cf.recommend_for_folded_user`), and the profile's own
rated items are excluded inside the function so a caller cannot forget.

When nothing survives mean-centering (no ratings, one rating, or ratings that all equal the user's
mean), the service falls back to the popularity baseline and reports `cold_start=true`, which is
what the UI uses to show the onboarding prompt instead of the personalization copy.

## Posters and details

The catalog already carries `tmdbId` from `links.csv`, so a poster is a join, not a second source of
truth. Without a key the app degrades cleanly: cards render text-only, the details dialog says the
synopsis is not in the catalog yet, and nothing fails.

1. Request a free key from [TMDB](https://www.themoviedb.org/settings/api) (a v4 read access token is
   preferred; a v3 API key also works).
2. Put `TMDB_READ_ACCESS_TOKEN=…` or `TMDB_API_KEY=…` in `.env`.
3. Optionally warm the cache: `uv run python scripts/fetch_posters.py --limit 500`. The script
   visits the most-rated films by default and has `--limit`, `--min-rating-count` and `--dry-run`.

Requests are server-side only, throttled to 20/s, retried once on a 429 with `Retry-After`, cached
as JSON under `data/processed/tmdb_cache/`, and written back to the movie row after the first
successful lookup. Every failure path returns `None`, so a TMDB outage cannot break search or
recommendations. TMDB's terms require the attribution in the footer:

> This product uses the TMDB API but is not endorsed or certified by TMDB.

A deployment should also show the TMDB logo per their brand guidelines; local development shows the
text notice.

## Results

Protocol (`PLAN.md` §9): the **chronological per-user split**, most recent 20% held out; a sample of
**10,000 users** (seed 42) out of the 18,632 in the model universe; **315,173 held-out ratings**;
ranking at **k=10** with a held-out rating of **4.0 or higher** counted as relevant. Every method is
scored on the same pairs; users with no relevant held-out item (234 of the sample) are reported as
skipped rather than scored as zero. Run on 2026-10-27 with `uv run python scripts/evaluate_models.py`.

| Method | RMSE | MAE | P@10 | R@10 | NDCG@10 | Catalogue coverage |
| --- | --- | --- | --- | --- | --- | --- |
| `global_mean` | 1.0625 | 0.8310 | — | — | — | — |
| `damped_item_mean` | 0.9681 | 0.7434 | 0.0183 | 0.0168 | 0.0203 | 0.17% |
| `popularity` | 0.9721 | 0.7466 | 0.0566 | 0.0532 | 0.0725 | 0.63% |
| `item_cf` | **0.9354** | **0.7120** | 0.0791 | 0.0755 | 0.1048 | **18.98%** |
| `user_cf` | 0.9395 | 0.7137 | **0.0825** | **0.0833** | **0.1082** | 4.68% |

Sample: 9,766 evaluated users per ranking row. Training and evaluation cost, same run:

| Method | Pairs on the item-mean fallback | Evaluation time | Peak traced memory |
| --- | --- | --- | --- |
| `global_mean` | — | 0.2 s | 92 MB |
| `damped_item_mean` | — | 1.4 s | 9 MB |
| `popularity` | — | 1.8 s | 8 MB |
| `item_cf` | 26.99% | 5.7 s | 304 MB |
| `user_cf` | 24.41% | 5.9 s | 294 MB |

Artifact training: item neighbors 14.2 s (5.5 MB), user neighbors 21.0 s (5.3 MB). The whole
evaluation is a 15 s job on a development subset, which is what makes iterating on models cheap.

What the numbers say:

- **The sanity checks in the plan hold.** Damped item mean beats the global mean (0.9681 vs 1.0625
  RMSE), and item-CF beats the item mean (0.9354), so the similarity path is not leaking or inverted.
- **Ratings and ranking disagree about the best method.** User-CF ranks best (NDCG@10 0.1082) while
  item-CF predicts best (RMSE 0.9354); the gap is small enough that both are worth keeping, which is
  exactly the comparison Phase 4.3 will put side by side in the UI.
- **Popularity is a strong ranking baseline** — better than the damped item mean (NDCG@10 0.0725 vs
  0.0203) — but it recommends almost the same 145 films to everyone (0.63% catalogue coverage),
  while item-CF covers 18.98%. That is the popularity bias the plan warns about, now measured.
- **Collaborative methods need support to say anything.** 26.99% of held-out pairs have no
  neighbor evidence for item-CF (24.41% for user-CF); those pairs fall back to the damped item mean
  and are reported rather than hidden, so part of item-CF's RMSE win is the baseline it borrows.

## API

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/api/health` | – | App + database status |
| `POST` | `/api/auth/signup` | – | Returns the user and an access token |
| `POST` | `/api/auth/login` | – | Same shape as signup |
| `GET` | `/api/auth/me` | ✔ | Current user profile |
| `GET` | `/api/movies/search?q=&limit=` | – | Ranked by match tier, then rating count |
| `GET` | `/api/movies/genres` | – | Every genre label in the catalog, for the filter chips |
| `GET` | `/api/movies/{id}` | – | Catalog detail, enriched from TMDB once when a key is set |
| `GET` | `/api/ratings` | ✔ | The user's ratings with movie payloads |
| `POST` | `/api/ratings` | ✔ | Upsert; rejects non-half-star values and out-of-range ratings |
| `DELETE` | `/api/ratings/{movie_id}` | ✔ | 204 on success, 404 when unrated |
| `GET` | `/api/recommend?limit=&genres=&year_min=&year_max=&min_score=` | ✔ | Picks with score, predicted rating, reason, evidence; `genres` is comma-separated and any match qualifies |
| `GET` | `/api/compare?limit=&left=&right=` | ✔ | Two algorithms for one profile, with `overlap` ids and `overlap_count` |
| `GET` | `/api/model-runs?limit=` | – | Latest evaluation per method, newest first |

## Tests and lint

```bash
uv run pytest        # 205 tests: titles, pipeline, metrics, recommenders, fold-in, evaluation, API, TMDB
uv run ruff check .  # clean
cd frontend
npm test             # 74 component tests: cards, dialog, stars, filters, picks, compare, dashboard, session
npm run build        # production build
```

The backend suite runs against SQLite in a temporary directory and never touches a developer's
database, and the evaluation tests use small synthetic matrices with hand-computed expectations, so
`uv run pytest` needs no generated artifacts. Frontend tests use Vitest + Testing Library in jsdom
and mock the API module.

## Design decisions

- `ratings.csv` is streamed twice in bounded chunks — never loaded whole, never inserted into
  PostgreSQL. The database holds app and demo users only.
- The **catalog is all 87,585 movies**; the **model universe** is the filtered subset. Movies outside
  it stay searchable and ratable.
- The split is **chronological per user** (most recent 20% is test); the CSR matrix is built from the
  training split only, so evaluation cannot leak.
- Offline ranking uses **exactly the serving rule** for candidates (positive evidence, rated movies
  removed), so the reported NDCG describes the lists users actually receive.
- A collaborative pair with no neighbor support falls back to the damped item mean; the share is
  published as `fallback_rate` instead of quietly flattering RMSE.
- Titles are normalized to `Matrix, The (1999)` → title / display `The Matrix` / search `the matrix`,
  so both "the matrix" and "matrix" match. The real catalog contains one year-only title,
  `" (2019)"`, which is preserved instead of crashing ingestion.
- Search ranks a word-start match above a mid-word one, so `godfather` returns
  *The Godfather (1972)* first instead of burying it behind `GodFather (1991)`.
- Ratings are validated as half-star steps in 0.5–5.0; `4.2` and `6.0` are both rejected with a
  specific message.
- Cold start is reported honestly: a popularity list is never described as personalized.
- A stored session survives a transient API failure — only a token the server actually rejects is
  discarded, so a network blip does not sign the user out for good.
- Filters are applied after scoring a wider pool (8× the limit, capped at 200), because the artifacts
  store neighbours, not metadata. Genre matching is case-insensitive "any of"; year and score are
  exact. A filtered list that runs dry says so and offers to clear the filters.
- Only **positive** contributions explain a pick. Rating a film 1★ can legitimately *demote* its
  neighbours, but the card should never say "because you hated it" as the reason to watch something.
- The dashboard reads the latest run per algorithm and computes the best value per metric; when the
  table is empty it shows the command to produce the numbers instead of inventing any.
- Posters are cosmetic: a failed image hides itself, and a missing key leaves the layout unchanged.

## Layout

```
recsys/          pure ML/data package: titles, chunked loading, filters, splits, id maps,
                 baselines, item-item CF, user CF, metrics, evaluation harness
backend/app/     FastAPI app: settings, database, models, schemas, security, routers, TMDB client
backend/tests/   API tests (auth, search, ratings, recommendations, TMDB, poster prefetch)
migrations/      Alembic environment, initial schema, model_runs table
frontend/src/    React app: pages (discover, for you, compare, ratings, accuracy), components, api client, tests
scripts/         prepare_data.py, train_and_precompute.py, evaluate_models.py, seed_db.py, fetch_posters.py
data/raw/        downloaded MovieLens files — never modified or committed
data/processed/  derived Parquet, CSR matrix, id maps, neighbors, evaluation.json — never committed
tests/           tests for recsys/
```

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Setup: repo, venv, FastAPI hello-world, Vite shell, PostgreSQL/`pg_trgm`, lint, tests | Done |
| 1 | Data pipeline: titles, catalog, chunked loading, filters, CSR, chronological split, demo users | Done |
| 2 | Baselines + item-CF, JWT auth, search, ratings, recommendations with reasons | Done |
| 3 | Metrics, evaluation on the chronological split, `model_runs`, UI polish, API tests | Done |
| 4 | Filters, "why this recommendation", algorithm comparison, accuracy dashboard, posters and details | Done |
| 5 | SVD + fold-in, content-based, hybrid, serendipity, feedback loop, TMDB ingestion | Next |
| 6 | Deployment, demo recording, release tag | Planned |

## Known gaps

- Local development runs on SQLite; the PostgreSQL path (including `pg_trgm` similarity ranking) is
  implemented and enabled at startup but has not been exercised on this machine.
- One seed, one sample: no confidence intervals, and hyperparameters (`damping`, `k`) are untuned —
  Phase 3.2's "tune on a validation slice" is not done yet, so the current numbers are honest
  first-pass measurements, not the best achievable.
- The relevance cut-off is a plain 4.0+ threshold; graded relevance would reward ranking 5★ above
  4★ films.
- With a two-film profile, reasons can all cite the same source film and predicted scores sit close
  to the user's mean; both improve as the profile grows. Diversity and serendipity are Phase 5.
- No engagement analytics yet: the dashboard reports offline metrics from `model_runs` but not
  "how often a shown recommendation gets rated or liked", because impressions are not logged. That
  logging belongs with the Phase 5 feedback loop, where a like/reject action actually exists.
- Posters and synopses need a TMDB key; this checkout verified the client, cache and fallbacks with
  mocked responses, and no live TMDB call was made here.
- The comparison view has the same evidence limit on both sides and no per-user caching; it runs two
  fold-ins per request and is comfortably fast at development scale, but would need caching before a
  large public deployment.

See [PLAN.md](PLAN.md) for the full development plan and [AGENTS.md](AGENTS.md) for the rules the
build follows.
