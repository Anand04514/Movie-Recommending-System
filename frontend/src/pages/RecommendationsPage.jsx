import { useEffect, useState } from "react";

import * as api from "../api.js";
import MovieCard from "../components/MovieCard.jsx";
import SignInPrompt from "../components/SignInPrompt.jsx";

export default function RecommendationsPage({
  user,
  refreshKey,
  ratingsByMovie,
  onRate,
  onRequireAuth,
  onExplore,
}) {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "idle", data: null, error: null });
  const [savingId, setSavingId] = useState(null);
  const [actionError, setActionError] = useState(null);

  useEffect(() => {
    if (!user) {
      setState({ status: "idle", data: null, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState((previous) => ({
      status: previous.data ? "refreshing" : "loading",
      data: previous.data,
      error: null,
    }));

    api
      .getRecommendations({ limit: 12, signal: controller.signal })
      .then((data) => setState({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ status: "error", data: null, error: error.message });
      });

    return () => controller.abort();
  }, [user, refreshKey, reloadKey]);

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
  const algorithmLabel =
    data?.algorithm === "item_cf" ? "Item-based collaborative filtering" : "Popularity baseline";

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">For you</p>
        <h1>Picked from your taste.</h1>
        <p className="lede">
          Candidates come from movies you rated. Every pick names the rating that pushed it up.
        </p>
      </section>

      {actionError ? (
        <div className="alert" role="alert">
          {actionError}
          <button type="button" onClick={() => setActionError(null)} aria-label="Dismiss">
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

      {data ? (
        <div className="algo-bar">
          <span className="algo-dot" aria-hidden="true" />
          <span>{algorithmLabel}</span>
          {data.cold_start ? (
            <span className="muted">Rate a few films and this row starts reflecting you.</span>
          ) : (
            <span className="muted">Similar viewers, weighted by your ratings.</span>
          )}
        </div>
      ) : null}

      {data?.cold_start ? (
        <div className="panel onboarding">
          <div>
            <h3>This is the crowd&apos;s list, not yours — yet.</h3>
            <p>
              Rate five or six films you have opinions about. The moment you do, recommendations
              switch from popularity to item-based collaborative filtering.
            </p>
          </div>
          <button type="button" className="button" onClick={onExplore}>
            Rate some films
          </button>
        </div>
      ) : null}

      {data && data.recommendations.length === 0 ? (
        <div className="panel empty">
          <h3>Nothing left to suggest.</h3>
          <p>You have rated everything in the model universe. Impressive. Try searching newer films.</p>
          <button type="button" className="button" onClick={onExplore}>
            Search the catalog
          </button>
        </div>
      ) : null}

      {data && data.recommendations.length > 0 ? (
        <div className={`movie-grid ${state.status === "refreshing" ? "is-refreshing" : ""}`}>
          {data.recommendations.map((recommendation) => (
            <MovieCard
              key={recommendation.movie.id}
              movie={recommendation.movie}
              rating={ratingsByMovie.get(recommendation.movie.id) ?? 0}
              onRate={rate}
              busy={savingId === recommendation.movie.id}
              reason={recommendation.reason}
              predictedRating={recommendation.predicted_rating}
            />
          ))}
        </div>
      ) : null}
    </>
  );
}
