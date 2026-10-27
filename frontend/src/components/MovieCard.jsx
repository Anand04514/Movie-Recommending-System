import StarRating from "./StarRating.jsx";

export default function MovieCard({
  movie,
  rating = 0,
  onRate,
  onRemove,
  busy = false,
  reason,
  predictedRating,
  badge,
}) {
  return (
    <article className={`movie-card ${busy ? "is-busy" : ""}`} aria-busy={busy}>
      <div className="movie-head">
        <div>
          <h3>{movie.title}</h3>
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
        <p className="reason">
          <span className="reason-mark" aria-hidden="true">
            ↳
          </span>{" "}
          {reason}
          {predictedRating ? (
            <span className="predicted">Predicted {predictedRating.toFixed(1)}★</span>
          ) : null}
        </p>
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
    </article>
  );
}
