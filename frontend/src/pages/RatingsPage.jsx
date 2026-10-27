import { useState } from "react";

import MovieCard from "../components/MovieCard.jsx";
import SignInPrompt from "../components/SignInPrompt.jsx";

export default function RatingsPage({ user, ratings, onRate, onRemove, onRequireAuth, onExplore }) {
  const [savingId, setSavingId] = useState(null);
  const [error, setError] = useState(null);

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
    setError(null);
    try {
      await onRate(movie, value);
    } catch (err) {
      setError(err.message);
    } finally {
      setSavingId(null);
    }
  }

  async function remove(movie) {
    setSavingId(movie.id);
    setError(null);
    try {
      await onRemove(movie);
    } catch (err) {
      setError(err.message);
    } finally {
      setSavingId(null);
    }
  }

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Your library</p>
        <h1>My ratings</h1>
        <p className="lede">
          {ratings.length > 0
            ? `${ratings.length} ${ratings.length === 1 ? "film" : "films"} rated · average ${average}★`
            : "Nothing rated yet — your taste graph starts with the first film."}
        </p>
      </section>

      {error ? (
        <div className="alert" role="alert">
          {error}
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ) : null}

      {ratings.length === 0 ? (
        <div className="panel empty">
          <h3>No ratings yet.</h3>
          <p>Search the catalog and give a few films a score. Half stars are allowed.</p>
          <button type="button" className="button" onClick={onExplore}>
            Find films to rate
          </button>
        </div>
      ) : (
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
      )}
    </>
  );
}
