import { useCallback, useEffect, useState } from "react";

const NEXT_UP = [
  {
    title: "Search everything",
    body: "Every title in the MovieLens catalog, including recent releases — not just the movies the model was trained on.",
  },
  {
    title: "Rate in half stars",
    body: "A five-star widget with half-step precision, because MovieLens ratings are not integers.",
  },
  {
    title: "Picks with reasons",
    body: "Personalized recommendations that say which of your ratings drove them.",
  },
];

function useApiHealth() {
  const [health, setHealth] = useState({ state: "loading" });

  const check = useCallback(async (signal) => {
    setHealth({ state: "loading" });
    try {
      const response = await fetch("/api/health", { signal });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.database?.status ?? `HTTP ${response.status}`);
      }
      setHealth({ state: "online", payload });
    } catch (error) {
      if (error.name !== "AbortError") {
        setHealth({ state: "offline", message: error.message });
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    check(controller.signal);
    return () => controller.abort();
  }, [check]);

  return [health, () => check()];
}

function StatusCard({ health, retry }) {
  const online = health.state === "online";
  const loading = health.state === "loading";
  const label = loading ? "Checking the API" : online ? "API connected" : "API offline";

  return (
    <section className={`status-card ${online ? "is-online" : ""}`}>
      <div className="status-head">
        <span className={`dot ${loading ? "is-loading" : ""} ${online ? "is-ok" : ""}`} />
        <span className="status-label">{label}</span>
        {health.payload?.version && online && (
          <span className="status-meta">v{health.payload.version}</span>
        )}
      </div>

      {online && (
        <dl className="status-grid">
          <div>
            <dt>Database</dt>
            <dd>{health.payload.database.dialect}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>{health.payload.database.status}</dd>
          </div>
          <div>
            <dt>Endpoint</dt>
            <dd>/api/health</dd>
          </div>
        </dl>
      )}

      {health.state === "offline" && (
        <div className="status-offline">
          <p>Start the backend, then try again:</p>
          <code>uv run uvicorn backend.app.main:app --reload --port 8000</code>
          <button type="button" onClick={retry}>
            Retry connection
          </button>
        </div>
      )}

      {loading && <p className="status-note">Waiting for the FastAPI service…</p>}
    </section>
  );
}

export default function App() {
  const [health, retry] = useApiHealth();

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Movie Recs home">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          movie recs
        </a>
        <span className="phase">Phase 0 · shell</span>
      </header>

      <main>
        <section className="hero">
          <p className="eyebrow">MovieLens ml-32m · 32 million ratings</p>
          <h1>
            Find the film
            <br />
            you haven&apos;t met yet.
          </h1>
          <p className="lede">
            A full-stack recommender built from 32 million real ratings — searchable catalog,
            half-star ratings, and picks that explain themselves.
          </p>
        </section>

        <StatusCard health={health} retry={retry} />

        <section className="next">
          <h2>Next on the reel</h2>
          <div className="next-grid">
            {NEXT_UP.map((item, index) => (
              <article className="next-card" key={item.title}>
                <span className="next-index">0{index + 1}</span>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer>
        Ratings and titles from the{" "}
        <a href="https://grouplens.org/datasets/movielens/" target="_blank" rel="noreferrer">
          MovieLens ml-32m dataset
        </a>
        , GroupLens Research. Movie posters via TMDB in a later phase.
      </footer>
    </div>
  );
}
