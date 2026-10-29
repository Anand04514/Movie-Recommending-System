import { useEffect, useState } from "react";

import * as api from "../api.js";
import MovieCard from "../components/MovieCard.jsx";
import SignInPrompt from "../components/SignInPrompt.jsx";

const LIMIT = 12;

export default function OnboardingPage({ user, ratingsByMovie, onRate, onDone, onSkip, onRequireAuth }) {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  const [savingId, setSavingId] = useState(null);
  const [actionError, setActionError] = useState(null);

  useEffect(() => {
    if (!user) {
      setState({ status: "idle", data: null, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState({ status: "loading", data: null, error: null });
    api
      .getOnboarding({ limit: LIMIT, signal: controller.signal })
      .then((data) => setState({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ status: "error", data: null, error: error.message });
      });
    return () => controller.abort();
  }, [user, reloadKey]);

  if (!user) {
    return (
      <SignInPrompt
        title="Start with five films."
        body="Sign in and rate a handful of well-known movies so the models have something to learn from."
        onSignIn={() => onRequireAuth?.("Sign in to start onboarding.")}
      />
    );
  }

  const data = state.data;
  const movies = data?.movies ?? [];
  const target = data?.target ?? 5;
  const count = movies.filter((movie) => ratingsByMovie.has(movie.id)).length;
  const done = count >= target;

  async function rate(movie, value) {
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

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Cold start</p>
        <h1>Teach it your taste.</h1>
        <p className="lede">
          A recommender cannot know anything about you until you tell it something. Rate five of
          these well-known films — the set deliberately spans genres and decades — and every model
          has a starting point. You can keep rating afterwards.
        </p>
      </section>

      {state.status === "error" ? (
        <div className="panel empty">
          <h3>Onboarding is unavailable.</h3>
          <p>{state.error}</p>
          <button type="button" className="button" onClick={() => setReloadKey((key) => key + 1)}>
            Try again
          </button>
        </div>
      ) : null}

      {actionError ? (
        <div className="alert" role="alert">
          {actionError}
          <button type="button" onClick={() => setActionError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ) : null}

      {data ? (
        <div className="onboarding-progress">
          <div className="progress-copy">
            <strong>{count}</strong> of {target} rated
          </div>
          <div
            className="progress-track"
            role="progressbar"
            aria-label="Onboarding progress"
            aria-valuemin={0}
            aria-valuemax={target}
            aria-valuenow={count}
          >
            <span style={{ width: `${Math.min(100, (count / target) * 100)}%` }} />
          </div>
          <button type="button" className="link-button" onClick={onSkip}>
            Skip for now
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

      {data && movies.length === 0 ? (
        <div className="panel empty">
          <h3>The catalog is empty.</h3>
          <p>Seed the movie catalog, then come back to pick a starting set.</p>
        </div>
      ) : null}

      {movies.length > 0 ? (
        <div className="movie-grid">
          {movies.map((movie) => (
            <MovieCard
              key={movie.id}
              movie={movie}
              rating={ratingsByMovie.get(movie.id) ?? 0}
              onRate={rate}
              busy={savingId === movie.id}
            />
          ))}
        </div>
      ) : null}

      {data && movies.length > 0 ? (
        <div className={`onboarding-finish ${done ? "is-ready" : ""}`}>
          <div>
            <h3>{done ? "That is enough to start." : "A few more films to go."}</h3>
            <p>
              {done
                ? "Your recommendations already reflect these ratings. Add more whenever you like; the list updates immediately."
                : `Rate ${target - count} more to unlock the personalized models.`}
            </p>
          </div>
          <button type="button" className="button" onClick={onDone}>
            {done ? "See my recommendations" : "Skip to my recommendations"}
          </button>
        </div>
      ) : null}
    </>
  );
}
