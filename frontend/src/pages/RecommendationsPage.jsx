import { useCallback, useEffect, useMemo, useState } from "react";

import * as api from "../api.js";
import MovieCard from "../components/MovieCard.jsx";
import RecommendationFilters, {
  EMPTY_FILTERS,
  isFiltered,
  toRequestFilters,
} from "../components/RecommendationFilters.jsx";
import SignInPrompt from "../components/SignInPrompt.jsx";

const LIMIT = 12;
// The server logs this row's impressions under this label, so clicks have to match it.
const NEW_RELEASE_ALGORITHM = "new_releases";
const FILTER_DEBOUNCE_MS = 200;
const MODELS = [
  {
    id: "hybrid",
    label: "Hybrid · CF + content",
    note: "Similar viewers plus genre, tag and decade overlap, blended at the weight picked by validation.",
  },
  {
    id: "svd",
    label: "SVD with fold-in",
    note: "Matrix factorization: latent taste factors learned from the ratings, then folded in from yours.",
  },
  {
    id: "item_cf",
    label: "Item-based CF",
    note: "Films that people who rated your favourites highly also rated highly.",
  },
  {
    id: "user_cf",
    label: "User-based CF",
    note: "Viewers whose ratings line up with yours, weighted by how closely they agree.",
  },
  {
    id: "content",
    label: "Content-based",
    note: "The genres, decades and tags you keep choosing, scored over the whole catalog.",
  },
];
const MODEL_NOTES = Object.fromEntries(MODELS.map((model) => [model.id, model.note]));
MODEL_NOTES.user_cf = "Viewers whose ratings line up with yours, weighted by how closely they agree.";
MODEL_NOTES.popularity = "The crowd's list, until your ratings are enough to personalise.";
const ALGORITHM_LABELS = {
  hybrid: "Hybrid · collaborative + content",
  svd: "SVD with fold-in",
  item_cf: "Item-based collaborative filtering",
  user_cf: "User-based collaborative filtering",
  content: "Content-based model",
  popularity: "Popularity baseline",
};

export default function RecommendationsPage({
  user,
  refreshKey,
  ratingsByMovie,
  onRate,
  onRequireAuth,
  onExplore,
  onOnboard,
}) {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "idle", data: null, error: null });
  const [savingId, setSavingId] = useState(null);
  const [feedbackBusyId, setFeedbackBusyId] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [notice, setNotice] = useState(null);
  // Null means "let the A/B experiment assign the default"; a choice pins it explicitly.
  const [algorithm, setAlgorithm] = useState(null);
  const [filters, setFilters] = useState({ ...EMPTY_FILTERS, genres: [] });
  const [genreOptions, setGenreOptions] = useState([]);
  const [fresh, setFresh] = useState([]);

  const request = useMemo(() => toRequestFilters(filters), [filters]);
  const filtered = isFiltered(filters);

  useEffect(() => {
    const controller = new AbortController();
    api
      .listGenres({ signal: controller.signal })
      .then((genres) => setGenreOptions(genres))
      .catch((error) => {
        if (error.name === "AbortError") return;
        // Genre chips are a convenience; year and score filters still work without them.
        setGenreOptions([]);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!user) {
      setFresh([]);
      return undefined;
    }
    // Films newer than the dataset have no collaborative signal, so they get their own row built
    // by the content model. A failure here must never take the main list down with it.
    const controller = new AbortController();
    api
      .getNewReleases({ limit: 6, signal: controller.signal })
      .then((rows) => setFresh(Array.isArray(rows) ? rows : []))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setFresh([]);
      });
    return () => controller.abort();
  }, [user, refreshKey]);

  useEffect(() => {
    if (!user) {
      setState({ status: "idle", data: null, error: null });
      return undefined;
    }
    // An invalid year range is reported beside the inputs; keep the last good list until it is.
    if (request.error) {
      setState((previous) =>
        previous.data || previous.status === "idle"
          ? previous
          : { status: "idle", data: null, error: null },
      );
      return undefined;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setState((previous) => ({
        status: previous.data ? "refreshing" : "loading",
        data: previous.data,
        error: null,
      }));

      api
        .getRecommendations({
          limit: LIMIT,
          algorithm,
          genres: request.genres,
          yearMin: request.yearMin,
          yearMax: request.yearMax,
          minScore: request.minScore,
          signal: controller.signal,
        })
        .then((data) => setState({ status: "ready", data, error: null }))
        .catch((error) => {
          if (error.name === "AbortError") return;
          setState({ status: "error", data: null, error: error.message });
        });
    }, FILTER_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [user, refreshKey, reloadKey, request, algorithm]);

  async function rate(movie, value) {
    if (!user) {
      onRequireAuth("Sign in to save your ratings.");
      return;
    }
    setSavingId(movie.id);
    setActionError(null);
    try {
      await onRate(movie, value);
    } catch (error) {
      setActionError(error.message);
    } finally {
      setSavingId(null);
    }
  }

  // A click is what separates "shown" from "chosen" in the A/B report. It is fire-and-forget: a
  // failed beacon must never delay the dialog or put an error in front of the reader.
  const logClick = useCallback((movie, source) => {
    api.sendFeedback(movie.id, "clicked", source).catch(() => {});
  }, []);

  async function react(movie, action) {
    setFeedbackBusyId(movie.id);
    setActionError(null);
    try {
      await api.sendFeedback(movie.id, action, state.data?.algorithm ?? algorithm);
      setState((previous) =>
        previous.data
          ? {
              ...previous,
              data: {
                ...previous.data,
                recommendations: previous.data.recommendations.filter(
                  (item) => item.movie.id !== movie.id,
                ),
              },
            }
          : previous,
      );
      setNotice(
        action === "liked"
          ? `Liked ${movie.title} — we will look for more like it.`
          : `Noted. ${movie.title} will not be shown again.`,
      );
    } catch (error) {
      setActionError(error.message);
    } finally {
      setFeedbackBusyId(null);
    }
  }

  function resetFilters() {
    setFilters({ ...EMPTY_FILTERS, genres: [] });
  }

  if (!user) {
    return (
      <SignInPrompt
        title="Your picks live behind the door."
        body="Sign in to see recommendations computed from your own ratings, with a reason for every film."
        onSignIn={() => onRequireAuth("Sign in to see picks made for you.")}
      />
    );
  }

  const data = state.data;
  const picks = data?.recommendations ?? [];
  const algorithmLabel = ALGORITHM_LABELS[data?.algorithm] ?? "Recommendations";
  const algorithmNote = MODEL_NOTES[data?.algorithm] ?? "Ranked by the selected model.";
  const servedModel = MODELS.some((model) => model.id === data?.algorithm)
    ? data.algorithm
    : null;
  const selectValue = algorithm ?? servedModel ?? "";
  const experimentLabel =
    data?.experiment && data?.variant ? `A/B arm · ${data.variant}` : null;

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">For you</p>
        <h1>Picked from your taste.</h1>
        <p className="lede">
          Candidates come from movies you rated. Every pick names the rating that pushed it up —
          narrow the list by genre, release year, or how highly the model rates it.
        </p>
      </section>

      <RecommendationFilters
        genres={genreOptions}
        value={filters}
        onChange={setFilters}
        onReset={resetFilters}
        error={request.error}
      />

      {actionError ? (
        <div className="alert" role="alert">
          {actionError}
          <button type="button" onClick={() => setActionError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ) : null}

      {notice ? (
        <div className="notice" role="status">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss notice">
            ×
          </button>
        </div>
      ) : null}

      {state.status === "error" ? (
        <div className="panel empty">
          <h3>Recommendations are unavailable.</h3>
          <p>{state.error}</p>
          <button type="button" className="button" onClick={() => setReloadKey((key) => key + 1)}>
            Try again
          </button>
        </div>
      ) : null}

      {state.status === "loading" ? (
        <div className="movie-grid">
          {Array.from({ length: 6 }, (_, index) => (
            <div className="movie-card skeleton" key={index} aria-hidden="true">
              <div className="skeleton-line wide" />
              <div className="skeleton-line" />
              <div className="skeleton-line short" />
            </div>
          ))}
        </div>
      ) : null}

      {data && picks.length > 0 ? (
        <div className="algo-bar">
          <label className="inline-select model-select">
            <span>Model</span>
            <select
              aria-label="Recommendation model"
              value={selectValue}
              onChange={(event) => setAlgorithm(event.target.value)}
            >
              {servedModel === null ? (
                <option value="" disabled>
                  {data.cold_start ? "Popularity fallback" : "Assigned model"}
                </option>
              ) : null}
              {MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          <span className="algo-dot" aria-hidden="true" />
          <span>{algorithmLabel}</span>
          {data.cold_start ? (
            <span className="muted">Rate a few films and this row starts reflecting you.</span>
          ) : (
            <span className="muted">{algorithmNote}</span>
          )}
          {experimentLabel ? <span className="algo-arm">{experimentLabel}</span> : null}
          <span className="algo-count">
            {picks.length} {picks.length === 1 ? "pick" : "picks"}
            {data.candidates_considered
              ? ` · ${data.candidates_considered} candidates scored`
              : null}
          </span>
        </div>
      ) : null}

      {data?.needs_onboarding && picks.length > 0 ? (
        <div className="panel onboarding">
          <div>
            <h3>Five ratings in, this becomes yours.</h3>
            <p>
              Until then the row leans on popularity and the genres of whatever you have rated.
              A short onboarding set — a few well-known films across genres — gives the models a
              real starting point.
            </p>
          </div>
          <button type="button" className="button" onClick={onOnboard}>
            Start onboarding
          </button>
        </div>
      ) : null}

      {data?.cold_start && !data?.needs_onboarding && picks.length > 0 ? (
        <div className="panel onboarding">
          <div>
            <h3>This is the crowd&apos;s list, not yours — yet.</h3>
            <p>
              Your ratings do not agree on anything yet, so every model falls back to the
              popularity baseline. Mixing a few high and low ratings will break the tie.
            </p>
          </div>
          <button type="button" className="button" onClick={onExplore}>
            Rate some films
          </button>
        </div>
      ) : null}

      {data && picks.length === 0 && filtered ? (
        <div className="panel empty">
          <h3>No picks match those filters.</h3>
          <p>
            Loosen the year range, drop a genre, or lower the predicted score — the rows behind
            this filter are still there.
          </p>
          <button type="button" className="button" onClick={resetFilters}>
            Clear filters
          </button>
        </div>
      ) : null}

      {data && picks.length === 0 && !filtered ? (
        <div className="panel empty">
          <h3>Nothing left to suggest.</h3>
          <p>
            You have rated everything in the model universe. Impressive. Try searching newer films.
          </p>
          <button type="button" className="button" onClick={onExplore}>
            Search the catalog
          </button>
        </div>
      ) : null}

      {picks.length > 0 ? (
        <div className={`movie-grid ${state.status === "refreshing" ? "is-refreshing" : ""}`}>
          {picks.map((recommendation) => (
            <MovieCard
              key={recommendation.movie.id}
              movie={recommendation.movie}
              rating={ratingsByMovie.get(recommendation.movie.id) ?? 0}
              onRate={rate}
              onFeedback={react}
              onOpen={(movie) => logClick(movie, state.data?.algorithm ?? algorithm)}
              busy={savingId === recommendation.movie.id}
              feedbackBusy={feedbackBusyId === recommendation.movie.id}
              reason={recommendation.reason}
              predictedRating={recommendation.predicted_rating}
              evidence={recommendation.evidence}
              badge={recommendation.discovery ? "Discovery" : undefined}
            />
          ))}
        </div>
      ) : null}

      {!filtered && fresh.length > 0 ? (
        <section className="new-releases" aria-labelledby="new-releases-heading">
          <header className="section-head">
            <h2 id="new-releases-heading">New releases for you</h2>
            <span className="muted">Content model · via TMDB</span>
          </header>
          <p className="muted external-note">
            Released after the dataset, so no collaborative model has seen them. These are scored
            from genre, keywords and decade against your taste; rating one adds it to your history
            and filters it out of this row.
          </p>
          <div className="movie-grid">
            {fresh.map((recommendation) => (
              <MovieCard
                key={recommendation.movie.id}
                movie={recommendation.movie}
                rating={ratingsByMovie.get(recommendation.movie.id) ?? 0}
                onRate={rate}
                onOpen={(movie) => logClick(movie, NEW_RELEASE_ALGORITHM)}
                busy={savingId === recommendation.movie.id}
                reason={recommendation.reason}
                predictedRating={recommendation.predicted_rating}
                badge={recommendation.new_release ? "New" : undefined}
              />
            ))}
          </div>
        </section>
      ) : null}
    </>
  );
}
