# Movie Recommendation System — Development Plan (MovieLens 32M)

Built with Antigravity · Dataset: MovieLens ml-32m (GroupLens) · B.Tech CS project

## 1. Goal

Build a full-stack movie recommender: users sign up, search for real movies (including recent ones), rate them, and get personalized recommendations with explanations. Start with a working collaborative-filtering MVP, then add nice-to-have and advanced features. Every phase ends with something runnable and testable.

The dataset is large, so this plan is designed around scale from day one: a full movie catalog for search, a filtered subset for model training, and precomputed results for fast serving.

## 2. Tech Stack

- **Backend:** FastAPI (Python 3.11+), SQLAlchemy + Alembic
- **ML:** pandas or Polars, NumPy, SciPy (sparse matrices, `svds`), scikit-learn. Optional: the `implicit` library for ALS, FAISS for fast nearest-neighbor search.
- **Database:** PostgreSQL with `pg_trgm` for title search (SQLite for early local dev only)
- **Auth:** JWT + bcrypt
- **Frontend:** React + Vite
- **Optional:** Redis for cached recommendations. Also needed: a free TMDB API key for posters and for adding movies newer than MovieLens (Phases 4 and 5)
- **Testing:** pytest
- **Deploy:** Render or Railway (free tiers have limits, see Phase 6)

**Machine:** 16 GB RAM is comfortable. With 8 GB, stay on the development subset and process ratings in chunks.

## 3. Dataset: MovieLens ml-32m

**Download:** from https://grouplens.org/datasets/movielens/ (the ml-32m zip, roughly a few hundred MB compressed).

**What's in it:** about 32 million ratings from about 200,000 users on about 87,000 movies, released in October 2023. Every user has at least 20 ratings. Ratings run from 0.5 to 5.0 in half-star steps. The newest movies are from around 2023; nothing later exists in any MovieLens release.

**Files:**

- `ratings.csv` — userId, movieId, rating, timestamp (the big one, close to 1 GB unzipped)
- `movies.csv` — movieId, title, genres
- `tags.csv` — userId, movieId, tag, timestamp
- `links.csv` — movieId, imdbId, tmdbId (use the tmdbId for posters)

**Things that will surprise you:**

- Titles look like `Matrix, The (1999)`: the year is inside the title and articles are moved to the end. Normalize titles into a clean title, a release year, and a display title ("The Matrix"), and keep a search-normalized version.
- Genres are one pipe-separated string such as `Action|Sci-Fi`. There are about 20 labels, including IMAX and `(no genres listed)`.
- Many movies have only a handful of ratings. They belong in the catalog but not in the trained model.
- There are no user demographics and no ready-made train/test splits.

**Core data strategy (the key design decision):**

1. **Catalog = all movies.** Load all of `movies.csv` and `links.csv` into the database so search finds everything.
2. **Model universe = filtered subset.** Train only on movies with at least 20 ratings and users with at least 20 ratings (already true for users). For development, further sample to roughly 20,000 users or 2–3 million ratings so experiments take minutes, not hours.
3. **Do not put 32M ratings in PostgreSQL.** Keep the training ratings as Parquet or a saved sparse matrix on disk. The database stores only app users' ratings plus a small set of seeded demo users.
4. **Movies outside the model universe** (new or rarely rated) still appear in search and can be rated. They get recommended through the content-based model and the popularity fallback.
5. Keep `data/raw/` read-only. All cleaning goes to `data/processed/`.

**Mean-centering note:** missing ratings are not zeros. Mean-center each user's ratings, and compute similarity on co-rated items with sparse matrices.

**Evaluation split:** chronological per user. For each user, the most recent 20% of ratings are test, the rest are train. Evaluate on a sample of about 10,000 users to keep it fast.

## 4. Working with Antigravity

- Save this plan in the repo as `PLAN.md`. Add a rules file stating the stack, folder layout, the memory limits above, and "write tests and run them before finishing".
- **One phase = one agent task.** Give it a bounded scope, review the plan it proposes before approving, and read the diffs.
- Keep ML code in a pure Python package (`recsys/`) with no web imports so it can be tested alone.
- Tell the agent explicitly: never load the full ratings file with a plain `pd.read_csv` into the database, and never modify `data/raw/`.
- If your Antigravity version has a browser agent, use it to test signup, search, rating, and recommendation flows once the frontend exists.
- Commit after each phase once its acceptance criteria pass.

**Prompt template per task:**

> Read PLAN.md. Implement Phase N, tasks N.x–N.y only. Write tests for everything you add. Run the tests. Stop and summarize what changed when the acceptance criteria are met.

## 5. Repo Layout

```
movie-recsys/
├── PLAN.md
├── data/
│   ├── raw/ml-32m/               # read-only
│   └── processed/                # Parquet, sparse matrices, ID maps
├── recsys/                       # pure ML package
│   ├── data.py                   # chunked loading, filters, splits, ID mapping
│   ├── titles.py                 # title/year/article normalization
│   ├── baselines.py              # global mean, damped item mean, popularity
│   ├── item_cf.py                # item-item top-K neighbors
│   ├── user_cf.py                # sampled-user CF (dev scale)
│   ├── svd.py                    # matrix factorization + fold-in for new users
│   ├── content.py                # genre/tag/year features
│   ├── hybrid.py
│   ├── metrics.py                # RMSE, MAE, precision@K, recall@K, NDCG@K
│   └── explain.py
├── backend/
│   ├── app/main.py
│   ├── app/routers/              # auth, movies, search, ratings, recommendations, admin
│   ├── app/models.py
│   └── tests/
├── frontend/                     # React + Vite
├── scripts/
│   ├── prepare_data.py           # raw -> processed
│   ├── seed_db.py                # catalog + demo users
│   └── train_and_precompute.py   # train models, store results
└── tests/
```

## 6. Data Model

- **users:** id, email, password\_hash, display\_name, is\_seeded, created\_at
- **movies:** id (MovieLens movieId), title\_display, title\_search, release\_year, genres, imdb\_id, tmdb\_id, poster\_path, rating\_count, in\_model (true if part of the trained universe), source (movielens or tmdb), overview, tmdb\_keywords, added\_at
- **ratings:** user\_id, movie\_id, rating (0.5–5.0), timestamp. Unique on (user\_id, movie\_id). App users and seeded demo users only.
- **feedback:** user\_id, movie\_id, action (liked, rejected, clicked), source\_algorithm, created\_at
- **recommendation\_logs:** user\_id, movie\_id, algorithm, score, shown\_at
- **precomputed\_recs:** user\_id, algorithm, movie\_id, score, rank, computed\_at (for seeded users)
- **model\_runs:** id, algorithm, params, rmse, mae, precision\_at\_k, recall\_at\_k, ndcg\_at\_k, trained\_at

Add a trigram index on `title_search` for fast partial-title search.

## 7. Phases

### Phase 0 — Setup (1–2 days)

- 0.1 Create the repo, virtual environment, folder layout, and Antigravity rules file.
- 0.2 Download ml-32m into `data/raw/` and verify the four CSV files.
- 0.3 FastAPI hello-world, React + Vite shell, PostgreSQL connection with `pg_trgm` enabled.
- 0.4 Linting, pytest, and a README stub.

**Done when:** backend and frontend both start, and tests run.

### Phase 1 — Data Pipeline (1–2 weeks)

- 1.1 `titles.py`: parse `Matrix, The (1999)` into clean title, year, and display title. Test with at least 20 tricky titles (articles in several forms, titles with parentheses, missing years).
- 1.2 Load `movies.csv` and `links.csv` into the catalog. Mark `in_model` after filtering.
- 1.3 Read `ratings.csv` in chunks. Map user and movie ids to compact integer indexes and save the maps.
- 1.4 Apply filters (movies with at least 20 ratings) and create the development sample.
- 1.5 Build and save a sparse user-item matrix (CSR). Print counts, sparsity, rating distribution, and memory use.
- 1.6 Chronological per-user train/test split saved to disk.
- 1.7 Seed about 500–1,000 demo users with their ratings into the database.

**Done when:** one command takes raw files to processed files and a seeded database, and search for a recent movie (for example one released in 2021) returns results.

### Phase 2 — Core MVP (2–3 weeks)

**Models**

- 2.1 Baselines: global mean, damped item mean, popularity ranking.
- 2.2 Item-based CF: sparse cosine similarity on mean-centered ratings, keeping the top 50–100 neighbors per movie. Build it in chunks so memory stays bounded.
- 2.3 Recommend for a user from their ratings using item neighbors, excluding movies already rated.

**Backend**

- 2.4 Signup and login with JWT.
- 2.5 `GET /movies/search?q=` (trigram search, ranked by match then popularity), `GET /movies/{id}`.
- 2.6 `POST /ratings` (0.5–5.0 in half steps), update, delete.
- 2.7 `GET /recommend` returns top-N with predicted score and a reasoning line such as "Because you rated The Matrix 5 stars".

**Frontend**

- 2.8 Signup and login pages.
- 2.9 Search bar, movie cards with half-star rating widget.
- 2.10 "My ratings" page.
- 2.11 Recommendations page with predicted score and reasoning line.

**Done when:** a new user can sign up, search for specific movies, rate them, and see personalized recommendations with reasons. Tests pass.

### Phase 3 — Evaluation and Polish (1–2 weeks)

- 3.1 Metrics: RMSE, MAE, precision@10, recall@10, NDCG@10 (ratings of 4.0 or higher are relevant).
- 3.2 Evaluate baselines and item-CF on the chronological split, using a sample of about 10,000 users.
- 3.3 Add sampled-user CF (development scale) and compare.
- 3.4 Store results in `model_runs` and put a results table in the README.
- 3.5 UI polish: loading states, empty states, error messages, responsive layout.
- 3.6 API tests for auth, search, ratings, and recommendations.

**Done when:** a results table compares at least four methods, and the app looks presentable.

**Sanity checks:** damped item mean should beat the global mean. If CF does not beat item mean, look for a leak or a similarity bug before tuning. Remember that popularity is a strong baseline on MovieLens.

### Phase 4 — Nice-to-Have Features (1–2 weeks)

- 4.1 **Search and filter:** filter recommendations by genre, release year range, and minimum predicted rating.
- 4.2 **Richer explainability** ("Why this recommendation?"): show the specific movies you rated that drove it, plus shared genres.
- 4.3 **Algorithm comparison view:** pick two algorithms and see their top-N for the same user side by side, with overlap highlighted.
- 4.4 **Accuracy dashboard:** charts of RMSE and NDCG@10 per algorithm from `model_runs`, plus how often shown recommendations get rated or liked from `recommendation_logs`.
- 4.5 **Posters and details:** use `tmdbId` from `links.csv` with the TMDB API (free key). Fetch lazily, cache locally, and show TMDB attribution as their terms require.

**Done when:** every nice-to-have feature is reachable from the UI and has at least one test.

### Phase 5 — Advanced Features (2–3 weeks)

- 5.1 **Matrix factorization (SVD or ALS):** train on the sparse matrix with a moderate number of factors (start at 50). Compare against item-CF.
- 5.2 **Fold-in for new users:** compute a new user's vector from their ratings against fixed item factors, so recommendations update instantly without retraining.
- 5.3 **Content-based model:** genre, release-year, and tag features. Use it for movies outside the model universe.
- 5.4 **Hybrid recommender:** weighted blend of SVD/CF and content-based scores. Tune weights on a validation slice of the training data, never on test.
- 5.5 **Cold-start handling:** onboarding screen asking new users to rate 5–10 well-known, diverse movies. Fall back to popularity plus genre preferences until they have enough ratings.
- 5.6 **Serendipity:** reserve 10–20% of the list for lower-confidence or out-of-profile picks, labeled as discovery.
- 5.7 **Real-time feedback loop:** like and reject buttons on each recommendation. Rejected movies are excluded immediately and logged in `feedback`. Retrain on a schedule or with an admin button, not on every click.
- 5.8 **A/B testing:** randomly assign users to algorithm variants, log impressions and outcomes in `recommendation_logs`, and report which variant gets more positive actions.
- 5.9 **Precompute and cache:** nightly job fills `precomputed_recs` for seeded users; optionally cache hot results in Redis.

* 5.10 **On-demand TMDB ingestion (movies newer than MovieLens):** when a search finds only a few local results, query the TMDB search endpoint from the backend and show those results labeled as from TMDB. Add a movie to the catalog only when the user rates it or opens its page, not on every search. First match on `tmdbId` from `links.csv` so you never create a duplicate of a movie MovieLens already has. Give new movies ids from a separate range above all MovieLens ids, and store `source = tmdb`, genres mapped to MovieLens genre names, overview, keywords, release date, and poster path.
* 5.11 **Cold-start path for TMDB movies:** these movies start with no ratings, so recommend them through the content-based model (genres, keywords, TF-IDF on the overview) and the hybrid blend. The retrain job merges app users' ratings into the training data, and a movie joins the trained model once it has at least 20 ratings.
* 5.12 **New-releases boost:** add a small, tunable freshness bonus to the final score for movies released in the last 2 years, for example `final = score + w * exp(-age_in_days / 365)`, applied only when re-ranking the candidate list. Show a New badge and a New releases for you row. The offline test set cannot measure new movies, so check the boost by watching the share of new movies in lists and their like and click rates in `recommendation_logs`.
* 5.13 **TMDB safety:** keep the API key on the server, cache responses, handle rate limits and failures without breaking search, and mock TMDB in tests (no live calls). Keep the TMDB attribution visible.

**Done when:** hybrid matches or beats the best single model, new users get sensible recommendations right after onboarding, and the feedback loop visibly changes results.

### Phase 6 — Ship (about 1 week)

- 6.1 **Decide what to deploy.** Free tiers cannot hold or process 32M ratings. Deploy the app with the catalog, the precomputed item neighbors or factors, and the processed subset. Document that full training runs offline.
- 6.2 Deploy backend, database, and frontend on Render or Railway. Move secrets to environment variables.
- 6.3 README with screenshots, architecture diagram, results table, dataset notes (filters, sample sizes), and setup steps.
- 6.4 Record a short demo video or GIF.
- 6.5 Clean up code, remove dead files, and tag a release.

## 8. Feature Checklist

**Core**

- [ ] User signup and login
- [ ] Search for any movie in the catalog
- [ ] Rate movies (half-star steps)
- [ ] Personalized recommendations
- [ ] Basic recommendation reasoning

**Nice-to-have**

- [ ] Filter recommendations by genre, year, score
- [ ] Detailed "Why this recommendation?" explanations
- [ ] Side-by-side algorithm comparison
- [ ] Accuracy dashboard
- [ ] Posters and movie details

**Advanced**

- [ ] Matrix factorization with fold-in for new users
- [ ] Content-based and hybrid models
- [ ] Cold-start onboarding
- [ ] Serendipity picks
- [ ] Real-time feedback loop
- [ ] A/B testing of algorithms
- [ ] Precomputed recommendations and caching
- [ ] Deployment and accuracy tracking over time

* [ ] On-demand TMDB ingestion for movies newer than MovieLens
* [ ] Content-based cold start for TMDB movies
* [ ] New-releases boost and New releases row

## 9. Evaluation Plan

- **Rating prediction:** RMSE and MAE on held-out ratings.
- **Ranking quality:** precision@10, recall@10, NDCG@10 with ratings of 4.0 or higher as relevant.
- **Baselines to beat:** global mean, damped item mean, popularity.
- **Protocol:** chronological per-user split, evaluation on a sample of about 10,000 users, tuning on a validation slice of the training data only.
- **Report:** one README table with every method, every metric, sample sizes, and the run date.
- **Scale note:** also record training time and memory per method. That is a useful result on a dataset this size.

## 10. Common Pitfalls

- Loading the whole `ratings.csv` into pandas or PostgreSQL at once. Read in chunks and store sparse or Parquet files.
- Building a dense user-item matrix. It will not fit in memory.
- Random train/test splits that leak the future. Use chronological splits.
- Treating missing ratings as zeros.
- Forgetting that ratings are half-stars. Validate input from 0.5 to 5.0, and don't assume integers.
- Poor title handling. `Matrix, The (1999)` must be searchable as "the matrix" and "matrix".
- Recommending movies the user already rated. Always filter them out.
- Recomputing similarities on every request. Precompute neighbors or factors offline.
- Popularity bias: unfiltered, the top results are the same blockbusters for everyone. This is why serendipity and evaluation of coverage matter.
- Letting the agent edit raw data, or giving it huge tasks. Keep tasks small and tested.

* Creating duplicate movies when ingesting from TMDB. Always match on `tmdbId` first, and cache TMDB responses to stay inside rate limits.

## 11. Suggested Timeline

- **Week 0:** Phase 0
- **Weeks 1–2:** Phase 1 (data pipeline)
- **Weeks 3–5:** Phase 2 (core MVP)
- **Weeks 6–7:** Phase 3 (evaluation and polish)
- **Weeks 8–9:** Phase 4 (nice-to-have)
- **Weeks 10–12:** Phase 5 (advanced)
- **Week 13:** Phase 6 (ship)

If time is tight, stop after Phase 4 and add two items from Phase 5 (SVD with fold-in, plus cold-start onboarding, is a strong pair).

## 12. Definition of Done

- All core features work end to end and are tested, including search for recent movies.
- At least four algorithms are compared with documented metrics on a chronological split.
- New users get reasonable recommendations through onboarding.
- The app is deployed with a public link and a README that documents dataset filters and sample sizes.
