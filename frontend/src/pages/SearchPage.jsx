import { useEffect, useState } from "react";

import * as api from "../api.js";
import MovieCard from "../components/MovieCard.jsx";

const SUGGESTIONS = ["Dune", "The Matrix", "Amélie"];

function SkeletonCard() {
  return (
    <div className="movie-card skeleton" aria-hidden="true">
      <div className="skeleton-line wide" />
      <div className="skeleton-line" />
      <div className="skeleton-line short" />
    </div>
  );
}

export default function SearchPage({ user, ratingsByMovie, onRate, onRequireAuth }) {
  const [query, setQuery] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  const [savingId, setSavingId] = useState(null);
  const [actionError, setActionError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    const delay = query.trim() ? 250 : 0;
    const timer = window.setTimeout(async () => {
      setState((previous) => ({
        status: previous.data ? "refreshing" : "loading",
        data: previous.data,
        error: null,
      }));
      try {
        const data = await api.searchMovies(query, { signal: controller.signal, limit: 24 });
        setState({ status: "ready", data, error: null });
      } catch (error) {
        if (error.name === "AbortError") return;
        setState({ status: "error", data: null, error: error.message });
      }
    }, delay);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query, reloadKey]);

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

  const movies = state.data?.results ?? [];
  const trimmed = query.trim();

  return (
    <>
      <section className="hero">
        <p className="eyebrow">MovieLens ml-32m · 87,585 films</p>
        <h1>
          Find the film
          <br />
          you haven&apos;t met yet.
        </h1>
        <p className="lede">
          Every title in the catalog is searchable, including releases the model has never seen.
          Rate films to teach it your taste.
        </p>

        <form className="search-form" onSubmit={(event) => event.preventDefault()}>
          <svg className="search-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="1.8" />
            <path d="m16.5 16.5 4 4" fill="none" stroke="currentColor" strokeWidth="1.8" />
          </svg>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by title — “Dune”, “The Matrix”…"
            aria-label="Search the movie catalog"
            autoComplete="off"
          />
          {query ? (
            <button
              type="button"
              className="clear-search"
              onClick={() => setQuery("")}
              aria-label="Clear search"
            >
              ×
            </button>
          ) : null}
        </form>

        {!trimmed ? (
          <div className="quick-searches">
            <span>Try</span>
            {SUGGESTIONS.map((suggestion) => (
              <button type="button" key={suggestion} onClick={() => setQuery(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}
      </section>

      <section className="results">
        <header className="section-head">
          <h2>{trimmed ? `Results for “${trimmed}”` : "Popular right now"}</h2>
          <span className="muted" role="status">
            {state.status === "refreshing"
              ? "Searching…"
              : state.data
                ? `${state.data.count} ${state.data.count === 1 ? "title" : "titles"}`
                : "\u00a0"}
          </span>
        </header>

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
            <h3>We couldn&apos;t reach the catalog.</h3>
            <p>{state.error}</p>
            <button type="button" className="button" onClick={() => setReloadKey((key) => key + 1)}>
              Try again
            </button>
          </div>
        ) : null}

        {state.status === "loading" ? (
          <div className="movie-grid">
            {Array.from({ length: 6 }, (_, index) => (
              <SkeletonCard key={index} />
            ))}
          </div>
        ) : null}

        {state.status !== "loading" && state.status !== "error" && movies.length === 0 ? (
          <div className="panel empty">
            <h3>No films match “{trimmed}”.</h3>
            <p>Try a shorter title, or search by a distinctive word like “matrix” or “dune”.</p>
            <button type="button" className="button" onClick={() => setQuery("")}>
              Browse popular films
            </button>
          </div>
        ) : null}

        {movies.length > 0 ? (
          <div className={`movie-grid ${state.status === "refreshing" ? "is-refreshing" : ""}`}>
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
      </section>
    </>
  );
}
