import { useState } from "react";

import MovieDialog from "./MovieDialog.jsx";
import StarRating from "./StarRating.jsx";

function formatStars(value) {
  return Number.isInteger(value) ? `${value}★` : `${value.toFixed(1)}★`;
}

function Poster({ url }) {
  const [failed, setFailed] = useState(false);
  if (!url || failed) return null;
  return (
    <img
      className="poster"
      src={url}
      alt=""
      width="54"
      height="81"
      loading="lazy"
      decoding="async"
      onError={() => setFailed(true)}
    />
  );
}

function Evidence({ items }) {
  if (items.length === 0) return null;
  return (
    <div className="evidence">
      <span className="evidence-label">Also driven by</span>
      <ul aria-label="Other ratings behind this pick">
        {items.map((item) => (
          <li key={item.movie.id}>
            <span className="evidence-title">{item.movie.title}</span>
            <span className="evidence-rating">{formatStars(item.rating)}</span>
            {item.shared_genres?.length > 0 ? (
              <span className="evidence-genres">{item.shared_genres.slice(0, 2).join(" · ")}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function MovieCard({
  movie,
  rating = 0,
  onRate,
  onRemove,
  onFeedback,
  onOpen,
  busy = false,
  feedbackBusy = false,
  reason,
  predictedRating,
  badge,
  evidence = [],
}) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  // The sentence already names the strongest rating; the chips add the runners-up.
  const secondary = evidence.slice(1, 3);

  return (
    <article className={`movie-card ${busy ? "is-busy" : ""}`} aria-busy={busy}>
      <div className="movie-head">
        <Poster key={movie.poster_url} url={movie.poster_url} />
        <div className="movie-title">
          <h3>
            <button
              type="button"
              className="title-button"
              onClick={() => {
                setDetailsOpen(true);
                // Telemetry, not navigation: the dialog opens whatever the caller does here.
                onOpen?.(movie);
              }}
              title={`Details for ${movie.title}`}
            >
              {movie.title}
            </button>
          </h3>
          <p className="movie-meta">
            {movie.release_year ?? "Year unknown"}
            <span aria-hidden="true"> · </span>
            {movie.rating_count.toLocaleString()} ratings
          </p>
        </div>
        {badge ? <span className="badge">{badge}</span> : null}
      </div>

      {movie.genres.length > 0 && (
        <ul className="genre-list" aria-label="Genres">
          {movie.genres.slice(0, 3).map((genre) => (
            <li key={genre}>{genre}</li>
          ))}
        </ul>
      )}

      {reason ? (
        <div className="reason-block">
          <p className="reason">
            <span className="reason-mark" aria-hidden="true">
              ↳
            </span>{" "}
            {reason}
            {predictedRating ? (
              <span className="predicted">Predicted {predictedRating.toFixed(1)}★</span>
            ) : null}
          </p>
          <Evidence items={secondary} />
        </div>
      ) : null}

      <div className="movie-foot">
        <StarRating
          value={rating}
          onChange={(value) => onRate?.(movie, value)}
          disabled={busy || !onRate}
          label={`Rate ${movie.title}`}
        />
        {onRemove && rating > 0 ? (
          <button type="button" className="link-button" onClick={() => onRemove(movie)}>
            Remove
          </button>
        ) : null}
      </div>

      {onFeedback ? (
        <div className="feedback-row">
          <span className="feedback-prompt">More like this?</span>
          <button
            type="button"
            className="reaction is-negative"
            onClick={() => onFeedback(movie, "rejected")}
            disabled={busy || feedbackBusy}
            aria-label={`Not interested in ${movie.title}`}
          >
            Not for me
          </button>
          <button
            type="button"
            className="reaction is-positive"
            onClick={() => onFeedback(movie, "liked")}
            disabled={busy || feedbackBusy}
            aria-label={`Like ${movie.title}`}
          >
            ♥ Like
          </button>
        </div>
      ) : null}

      {detailsOpen ? (
        <MovieDialog
          movie={movie}
          rating={rating}
          busy={busy}
          onRate={onRate}
          onClose={() => setDetailsOpen(false)}
        />
      ) : null}
    </article>
  );
}
