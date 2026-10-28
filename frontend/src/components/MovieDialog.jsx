import { useEffect, useRef, useState } from "react";

import * as api from "../api.js";
import StarRating from "./StarRating.jsx";

export default function MovieDialog({ movie, rating = 0, busy = false, onRate, onClose }) {
  const dialogRef = useRef(null);
  const [state, setState] = useState({ status: "loading", data: null });

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return undefined;
    // jsdom and older browsers may not implement the modal API; the open attribute still shows it.
    try {
      if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
      else dialog.setAttribute("open", "");
    } catch {
      dialog.setAttribute("open", "");
    }
    return () => {
      if (dialog.open && typeof dialog.close === "function") dialog.close();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading", data: null });
    api
      .getMovie(movie.id, { signal: controller.signal })
      .then((data) => setState({ status: "ready", data }))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ status: "error", data: null });
      });
    return () => controller.abort();
  }, [movie.id]);

  const detail = state.data ?? movie;
  const genres = detail.genres ?? movie.genres ?? [];

  return (
    <dialog
      className="movie-dialog"
      ref={dialogRef}
      onCancel={(event) => {
        // Escape closes through React state; the native close event also fires on unmount and must
        // not be mistaken for the user dismissing the dialog.
        event.preventDefault();
        onClose?.();
      }}
      aria-labelledby="movie-dialog-title"
      onClick={(event) => {
        if (event.target === dialogRef.current) onClose?.();
      }}
    >
      <div className="dialog-body">
        <button type="button" className="dialog-close" onClick={onClose} aria-label="Close details">
          ×
        </button>

        <div className="dialog-head">
          {detail.poster_url ? (
            <img className="dialog-poster" src={detail.poster_url} alt="" width="120" height="180" />
          ) : null}
          <div className="dialog-heading">
            <h2 id="movie-dialog-title">{detail.title}</h2>
            <p className="movie-meta">
              {detail.release_year ?? "Year unknown"}
              <span aria-hidden="true"> · </span>
              {(detail.rating_count ?? 0).toLocaleString()} ratings
            </p>
            {genres.length > 0 ? (
              <ul className="genre-list" aria-label="Genres">
                {genres.map((genre) => (
                  <li key={genre}>{genre}</li>
                ))}
              </ul>
            ) : null}
          </div>
        </div>

        {state.status === "loading" ? (
          <p className="dialog-text muted">Loading synopsis…</p>
        ) : null}

        {state.status === "error" ? (
          <p className="dialog-text muted">
            Full details are unavailable right now. Your rating still counts.
          </p>
        ) : null}

        {state.status === "ready" && detail.overview ? (
          <p className="dialog-text">{detail.overview}</p>
        ) : null}

        {state.status === "ready" && !detail.overview ? (
          <p className="dialog-text muted">
            No synopsis in the catalog yet — it arrives once the TMDB key is configured.
          </p>
        ) : null}

        {onRate ? (
          <div className="dialog-rating">
            <span>Your rating</span>
            <StarRating
              value={rating}
              onChange={(value) => onRate(detail, value)}
              disabled={busy}
              label={`Rate ${detail.title}`}
            />
          </div>
        ) : null}

        {detail.poster_url || detail.overview ? (
          <p className="dialog-attribution">Poster and synopsis via TMDB.</p>
        ) : null}
      </div>
    </dialog>
  );
}
