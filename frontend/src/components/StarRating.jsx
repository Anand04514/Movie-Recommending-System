import { useId, useState } from "react";

const STARS = [1, 2, 3, 4, 5];

export default function StarRating({ value = 0, onChange, disabled = false, label = "Rate this movie" }) {
  const name = useId();
  const [hover, setHover] = useState(0);
  const shown = hover || value || 0;

  return (
    <div className="rating" onMouseLeave={() => setHover(0)}>
      <div className="stars" role="radiogroup" aria-label={label}>
        {STARS.map((star) => {
          const fill = Math.max(0, Math.min(1, shown - (star - 1))) * 100;
          return (
            <span className="star" key={star}>
              <span className="star-icon star-bg" aria-hidden="true">
                ★
              </span>
              <span className="star-icon star-fill" style={{ width: `${fill}%` }} aria-hidden="true">
                ★
              </span>
              {[star - 0.5, star].map((option) => (
                <label
                  className="star-half"
                  key={option}
                  onMouseEnter={() => {
                    if (!disabled) setHover(option);
                  }}
                >
                  <input
                    type="radio"
                    name={name}
                    value={option}
                    checked={value === option}
                    disabled={disabled}
                    onChange={() => onChange?.(option)}
                    aria-label={`${option} out of 5 stars`}
                  />
                </label>
              ))}
            </span>
          );
        })}
      </div>
      <span className={`rating-value ${value ? "is-set" : ""}`} aria-live="polite">
        {value ? value.toFixed(1) : "Rate"}
      </span>
    </div>
  );
}
