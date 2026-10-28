import { useEffect, useState } from "react";

import * as api from "../api.js";
import SignInPrompt from "../components/SignInPrompt.jsx";

const METHOD = {
  item_cf: {
    name: "Item-based CF",
    blurb: "Scores a film by how similar it is to the ones you rated highly.",
  },
  user_cf: {
    name: "User-based CF",
    blurb: "Finds viewers whose ratings look like yours, then borrows what they loved.",
  },
  popularity: {
    name: "Popularity baseline",
    blurb: "What the crowd watches most — the floor any personal model has to beat.",
  },
};

const LIMITS = [5, 10, 15];

function methodFor(algorithm) {
  return (
    METHOD[algorithm] ?? {
      name: algorithm.replaceAll("_", " "),
      blurb: "A precomputed method from the evaluation run.",
    }
  );
}

function Column({ side, sideKey, overlap }) {
  const meta = methodFor(side.algorithm);
  return (
    <section className="compare-column" aria-labelledby={`compare-head-${sideKey}`}>
      <header className="compare-head">
        <h2 id={`compare-head-${sideKey}`}>{meta.name}</h2>
        <p className="compare-note">{meta.blurb}</p>
      </header>

      {side.recommendations.length === 0 ? (
        <p className="muted compare-empty">No picks for this method yet.</p>
      ) : (
        <ol className="compare-list">
          {side.recommendations.map((rec, index) => {
            const both = overlap.has(rec.movie.id);
            return (
              <li key={rec.movie.id} className={`compare-item ${both ? "is-overlap" : ""}`}>
                <span className="compare-rank" aria-hidden="true">
                  {index + 1}
                </span>
                <span className="compare-body">
                  <span className="compare-title">
                    {rec.movie.title}
                    {both ? (
                      <span className="overlap-badge" title="Also in the other list">
                        Both
                      </span>
                    ) : null}
                  </span>
                  <span className="compare-reason">{rec.reason}</span>
                </span>
                <span className="compare-score">{rec.predicted_rating.toFixed(1)}★</span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

export default function ComparePage({ user, onRequireAuth, onExplore }) {
  const [limit, setLimit] = useState(10);
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "idle", data: null, error: null });

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
      .compareAlgorithms({ limit, signal: controller.signal })
      .then((data) => setState({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ status: "error", data: null, error: error.message });
      });

    return () => controller.abort();
  }, [user, limit, reloadKey]);

  if (!user) {
    return (
      <SignInPrompt
        title="Two methods, your taste."
        body="Sign in to see item-based and user-based filtering side by side, ranked on your own ratings."
        onSignIn={() => onRequireAuth("Sign in to compare algorithms made for you.")}
      />
    );
  }

  const data = state.data;
  const overlap = new Set(data?.overlap ?? []);
  const leftCount = data?.left.recommendations.length ?? 0;
  const coldStart = Boolean(data && (data.left.cold_start || data.right.cold_start));
  const loading = state.status === "loading";

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Algorithms</p>
        <h1>Two lenses, one taste.</h1>
        <p className="lede">
          Item-based filtering asks “what resembles what you loved?”. User-based filtering asks
          “who resembles you, and what did they love?”. Picks both lists agree on are marked
          <span className="inline-badge">Both</span>.
        </p>
      </section>

      <div className="compare-controls">
        <label className="inline-select">
          <span>List length</span>
          <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
            {LIMITS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        {data ? (
          <span className="compare-summary" role="status">
            {data.overlap_count === 0 ? (
              <>The two methods picked completely different films this time.</>
            ) : (
              <>
                <strong>{data.overlap_count}</strong> of {leftCount} picks appear in both lists
              </>
            )}
          </span>
        ) : null}
      </div>

      {state.status === "error" ? (
        <div className="panel empty">
          <h3>The comparison is unavailable.</h3>
          <p>{state.error}</p>
          <button type="button" className="button" onClick={() => setReloadKey((key) => key + 1)}>
            Try again
          </button>
        </div>
      ) : null}

      {coldStart ? (
        <div className="panel onboarding">
          <div>
            <h3>Rate a few films to make this a real comparison.</h3>
            <p>
              With nothing rated, both methods fall back to the popularity list, so the columns
              agree by construction. A handful of ratings gives each model something to disagree
              about — which is the interesting part.
            </p>
          </div>
          <button type="button" className="button" onClick={onExplore}>
            Rate some films
          </button>
        </div>
      ) : null}

      {loading ? (
        <div className="compare-grid" aria-busy="true">
          {["left", "right"].map((side) => (
            <div className="compare-column skeleton" key={side} aria-hidden="true">
              <div className="skeleton-line wide" />
              <div className="skeleton-line" />
              <div className="skeleton-line short" />
              <div className="skeleton-line" />
            </div>
          ))}
        </div>
      ) : null}

      {data ? (
        <div className={`compare-grid ${state.status === "refreshing" ? "is-refreshing" : ""}`}>
          <Column side={data.left} sideKey="left" overlap={overlap} />
          <Column side={data.right} sideKey="right" overlap={overlap} />
        </div>
      ) : null}

      {data ? (
        <p className="compare-footnote">
          Both lists come from artifacts precomputed offline. Predictions are shrunk toward your
          own average, so a film needs several agreeing neighbours to reach the top.
        </p>
      ) : null}
    </>
  );
}
