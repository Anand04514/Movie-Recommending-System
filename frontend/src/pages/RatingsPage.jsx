import { useState } from "react";

import MovieCard from "../components/MovieCard.jsx";
import SignInPrompt from "../components/SignInPrompt.jsx";

function SkeletonCard() {
  return (
    <div className="movie-card skeleton" aria-hidden="true">
      <div className="skeleton-line wide" />
      <div className="skeleton-line" />
      <div className="skeleton-line short" />
    </div>
  );
}

export default function RatingsPage({
  user,
  ratings,
  loading = false,
  error = null,
  onRate,
  onRemove,
  onRequireAuth,
  onExplore,
  onRetry,
}) {
  const [savingId, setSavingId] = useState(null);
  const [actionError, setActionError] = useState(null);

  if (!user) {
    return (
      <SignInPrompt
        title="Your ratings, kept in one place."
        body="Sign in to see every film you have rated, change a score, or clear one out."
        onSignIn={() => onRequireAuth("Sign in to see your ratings.")}
      />
    );
  }

  const average =
    ratings.length > 0
      ? (ratings.reduce((total, row) => total + row.rating, 0) / ratings.length).toFixed(2)
      : null;

  async function rate(movie, value) {
    setSavingId(movie.id);
    setActionError(null);
    try {
      await onRate(movie, value);
    } catch (err) {
      setActionError(err.message);
    } finally {
      setSavingId(null);
    }
  }

  async function remove(movie) {
    setSavingId(movie.id);
    setActionError(null);
    try {
      await onRemove(movie);
    } catch (err) {
      setActionError(err.message);
    } finally {
      setSavingId(null);
    }
  }

  const heading =
    loading || error
      ? null
      : ratings.length > 0
        ? `${ratings.length} ${ratings.length === 1 ? "film" : "films"} rated · average ${average}★`
        : "Nothing rated yet — your taste graph starts with the first film.";

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Your library</p>
        <h1>My ratings</h1>
        <p className="lede" role="status">
          {loading ? "Loading your ratings…" : (heading ?? "\u00a0")}
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

      {error ? (
        <div className="panel empty" role="alert">
          <h3>We couldn&apos;t load your ratings.</h3>
          <p>{error}</p>
          <button type="button" className="button" onClick={onRetry}>
            Try again
          </button>
        </div>
      ) : null}

      {!error && loading ? (
        <div className="movie-grid" aria-busy="true">
          {Array.from({ length: 3 }, (_, index) => (
            <SkeletonCard key={index} />
          ))}
        </div>
      ) : null}

      {!error && !loading && ratings.length === 0 ? (
        <div className="panel empty">
          <h3>No ratings yet.</h3>
          <p>Search the catalog and give a few films a score. Half stars are allowed.</p>
          <button type="button" className="button" onClick={onExplore}>
            Find films to rate
          </button>
        </div>
      ) : null}

      {!error && !loading && ratings.length > 0 ? (
        <div className="movie-grid">
          {ratings.map((row) => (
            <MovieCard
              key={row.movie.id}
              movie={row.movie}
              rating={row.rating}
              onRate={rate}
              onRemove={remove}
              busy={savingId === row.movie.id}
            />
          ))}
        </div>
      ) : null}
    </>
  );
}
