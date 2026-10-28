import { useState } from "react";

export const EMPTY_FILTERS = Object.freeze({
  genres: [],
  yearMin: "",
  yearMax: "",
  minScore: 0,
});

export function isFiltered(filters) {
  return (
    filters.genres.length > 0 ||
    filters.yearMin.trim() !== "" ||
    filters.yearMax.trim() !== "" ||
    filters.minScore > 0
  );
}

function parseYear(raw) {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isInteger(value) ? value : Number.NaN;
}

/** ``{ genres, yearMin, yearMax, minScore, error }`` for the API, or a message instead. */
export function toRequestFilters(filters) {
  const yearMin = parseYear(filters.yearMin);
  const yearMax = parseYear(filters.yearMax);
  if (Number.isNaN(yearMin) || Number.isNaN(yearMax)) {
    return { error: "Years must be whole numbers." };
  }
  if (yearMin != null && yearMax != null && yearMin > yearMax) {
    return { error: "The first year must come before the second." };
  }
  return {
    genres: filters.genres,
    yearMin,
    yearMax,
    minScore: filters.minScore > 0 ? filters.minScore : null,
    error: null,
  };
}

function summarize(filters) {
  const parts = [];
  if (filters.genres.length > 0) parts.push(filters.genres.join(", "));
  const from = filters.yearMin.trim();
  const to = filters.yearMax.trim();
  if (from && to) parts.push(`${from}–${to}`);
  else if (from) parts.push(`from ${from}`);
  else if (to) parts.push(`until ${to}`);
  if (filters.minScore > 0) parts.push(`${filters.minScore.toFixed(1)}★+`);
  return parts.join(" · ");
}

export default function RecommendationFilters({ genres, value, onChange, onReset, error }) {
  const dirty = isFiltered(value);
  const summary = summarize(value);
  // Phones start collapsed so the picks stay near the top; a wider screen has room for the panel.
  const [open, setOpen] = useState(
    () => !(typeof window !== "undefined" && window.matchMedia?.("(max-width: 760px)").matches),
  );

  function toggleGenre(genre) {
    const next = value.genres.includes(genre)
      ? value.genres.filter((item) => item !== genre)
      : [...value.genres, genre];
    onChange({ ...value, genres: next });
  }

  return (
    <section className="filter-bar" aria-label="Filter recommendations">
      <div className="filter-head">
        <button
          type="button"
          className="filter-toggle"
          aria-expanded={open}
          aria-controls="filter-controls"
          onClick={() => setOpen((current) => !current)}
        >
          <span className="filter-toggle-label">Filters</span>
          <span className="filter-summary">{summary || "Genre, year, predicted score"}</span>
          <span className="filter-caret" aria-hidden="true">
            {open ? "−" : "+"}
          </span>
        </button>
        {dirty ? (
          <button type="button" className="link-button filter-reset" onClick={onReset}>
            Reset
          </button>
        ) : null}
      </div>

      <div className="filter-controls" id="filter-controls" hidden={!open}>
        <div className="filter-row">
          <span className="filter-label" id="filter-genres-label">
            Genres
          </span>
          {genres.length > 0 ? (
            <div className="chip-row" role="group" aria-labelledby="filter-genres-label">
              {genres.map((genre) => (
                <button
                  type="button"
                  key={genre}
                  className="chip"
                  aria-pressed={value.genres.includes(genre)}
                  onClick={() => toggleGenre(genre)}
                >
                  {genre}
                </button>
              ))}
            </div>
          ) : (
            <span className="filter-placeholder">
              Genre list unavailable — year and score still work.
            </span>
          )}
        </div>

        <div className="filter-row">
          <span className="filter-label" id="filter-years-label">
            Released
          </span>
          <div className="year-range">
            <input
              type="number"
              inputMode="numeric"
              placeholder="From"
              aria-label="Earliest release year"
              min="1870"
              max="2100"
              aria-invalid={Boolean(error)}
              value={value.yearMin}
              onChange={(event) => onChange({ ...value, yearMin: event.target.value })}
            />
            <span className="year-dash" aria-hidden="true">
              –
            </span>
            <input
              type="number"
              inputMode="numeric"
              placeholder="To"
              aria-label="Latest release year"
              min="1870"
              max="2100"
              aria-invalid={Boolean(error)}
              value={value.yearMax}
              onChange={(event) => onChange({ ...value, yearMax: event.target.value })}
            />
          </div>

          <span className="filter-label filter-label-score" id="filter-score-label">
            Predicted
          </span>
          <div className="score-range">
            <input
              type="range"
              min="0"
              max="5"
              step="0.5"
              aria-label="Minimum predicted rating"
              value={value.minScore}
              onChange={(event) => onChange({ ...value, minScore: Number(event.target.value) })}
            />
            <output className={`score-output ${value.minScore > 0 ? "is-set" : ""}`}>
              {value.minScore > 0 ? `${value.minScore.toFixed(1)}★ and up` : "Any score"}
            </output>
          </div>
        </div>
      </div>

      {error ? (
        <p className="filter-error" role="status">
          {error}
        </p>
      ) : null}
    </section>
  );
}
