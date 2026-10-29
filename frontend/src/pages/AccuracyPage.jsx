import { useEffect, useMemo, useState } from "react";

import * as api from "../api.js";

const NAMES = {
  global_mean: "Global mean",
  damped_item_mean: "Damped item mean",
  popularity: "Popularity",
  item_cf: "Item-based CF",
  user_cf: "User-based CF",
  svd: "SVD with fold-in",
  content: "Content-based",
  hybrid: "Hybrid (CF + content)",
};

const PERCENT = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 2,
});

function methodName(slug) {
  return NAMES[slug] ?? slug.replaceAll("_", " ");
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

function formatMetric(value, kind) {
  if (value == null) return "—";
  return kind === "percent" ? PERCENT.format(value) : value.toFixed(3);
}

// Below this many impressions per arm the gap between two variants is noise, so the page reports
// the tally without crowning a winner.
const ENGAGEMENT_MIN_IMPRESSIONS = 50;

function formatRate(value) {
  return value == null ? "—" : PERCENT.format(value);
}

function EngagementPanel() {
  const [state, setState] = useState({ status: "loading", experiment: null, error: null });

  useEffect(() => {
    const controller = new AbortController();
    api
      .listExperiments({ signal: controller.signal })
      .then((reports) => {
        const [experiment] = Array.isArray(reports) ? reports : [];
        setState({ status: "ready", experiment: experiment ?? null, error: null });
      })
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ status: "error", experiment: null, error: error.message });
      });
    return () => controller.abort();
  }, []);

  const experiment = state.experiment;
  const arms = experiment?.variants ?? [];
  const impressions = arms.reduce((total, arm) => total + arm.impressions, 0);
  const enoughTraffic = arms.every((arm) => arm.impressions >= ENGAGEMENT_MIN_IMPRESSIONS);
  const bestPositive = Math.max(...arms.map((arm) => arm.positive_rate ?? -1));
  const leader =
    enoughTraffic && bestPositive >= 0
      ? arms.find((arm) => arm.positive_rate === bestPositive)?.variant
      : null;

  return (
    <section className="engagement-card" aria-labelledby="engagement-title">
      <h2 id="engagement-title">Live engagement</h2>
      <p className="chart-note">
        {experiment?.description ??
          "Which model a signed-in user gets when they have not chosen one."}
      </p>

      {state.status === "error" ? (
        <p className="table-note">Engagement data is unavailable right now ({state.error}).</p>
      ) : null}

      {state.status !== "error" && impressions === 0 ? (
        <p className="table-note">
          No experiment traffic yet. Impressions are logged as signed-in users load the For you page
          without choosing a model.
        </p>
      ) : null}

      {arms.length > 0 && impressions > 0 ? (
        <>
          <div className="table-wrap">
            <table className="metrics-table">
              <caption className="sr-only">A/B engagement for each variant</caption>
              <thead>
                <tr>
                  <th scope="col">Arm</th>
                  <th scope="col" className="num">Users</th>
                  <th scope="col" className="num">Shown</th>
                  <th scope="col" className="num">Likes</th>
                  <th scope="col" className="num">Rejects</th>
                  <th scope="col" className="num">Like rate</th>
                  <th scope="col" className="num">Rated</th>
                </tr>
              </thead>
              <tbody>
                {arms.map((arm) => (
                  <tr key={arm.variant}>
                    <th scope="row" className="method">
                      {methodName(arm.variant)}
                    </th>
                    <td className="num">{arm.users}</td>
                    <td className="num">{arm.impressions}</td>
                    <td className="num">{arm.likes}</td>
                    <td className="num">{arm.rejects}</td>
                    <td className={`num ${arm.variant === leader ? "is-best" : ""}`}>
                      {arm.variant === leader ? (
                        <span className="sr-only">Leading arm: </span>
                      ) : null}
                      {formatRate(arm.positive_rate)}
                    </td>
                    <td className="num">{formatRate(arm.rated_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="table-note">
            Only lists the experiment assigned are counted; a model the user picked by hand is
            excluded. Like rate is likes over like-or-reject verdicts.
            {enoughTraffic
              ? ""
              : ` With fewer than ${ENGAGEMENT_MIN_IMPRESSIONS} shown picks per arm this is a tally, not a result.`}
          </p>
        </>
      ) : null}
    </section>
  );
}

function buildMetrics(k) {
  return [
    { key: "rmse", label: "RMSE", better: "lower", kind: "score" },
    { key: "mae", label: "MAE", better: "lower", kind: "score" },
    { key: "precision_at_k", label: `P@${k}`, better: "higher", kind: "percent" },
    { key: "recall_at_k", label: `R@${k}`, better: "higher", kind: "percent" },
    { key: "ndcg_at_k", label: `NDCG@${k}`, better: "higher", kind: "percent" },
    { key: "coverage", label: "Coverage", better: "higher", kind: "percent" },
  ];
}

function BarChart({ title, metric, kind, rows, better }) {
  const entries = rows
    .filter((row) => row[metric] != null)
    .sort((a, b) => (better === "lower" ? a[metric] - b[metric] : b[metric] - a[metric]));
  if (entries.length === 0) return null;
  const max = Math.max(...entries.map((row) => row[metric]), 0);
  const bestValue = entries[0][metric];

  return (
    <section className="chart-card" aria-labelledby={`chart-${metric}`}>
      <h2 id={`chart-${metric}`}>{title}</h2>
      <p className="chart-note">{better === "lower" ? "Lower is better" : "Higher is better"}</p>
      <ul className="chart-bars">
        {entries.map((row) => (
          <li key={row.id} className={`chart-row ${row[metric] === bestValue ? "is-best" : ""}`}>
            <span className="chart-label">{methodName(row.algorithm)}</span>
            <span className="chart-track" aria-hidden="true">
              <span
                className="chart-fill"
                style={{ width: `${max > 0 ? (row[metric] / max) * 100 : 0}%` }}
              />
            </span>
            <span className="chart-value">{formatMetric(row[metric], kind)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function AccuracyPage() {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "loading", runs: [], error: null });

  useEffect(() => {
    const controller = new AbortController();
    setState((previous) => ({ ...previous, status: previous.runs.length ? "refreshing" : "loading" }));
    api
      .listModelRuns({ signal: controller.signal })
      .then((runs) => setState({ status: "ready", runs, error: null }))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ status: "error", runs: [], error: error.message });
      });
    return () => controller.abort();
  }, [reloadKey]);

  const runs = state.runs;
  const k = runs[0]?.params?.k ?? 10;
  const metrics = useMemo(() => buildMetrics(k), [k]);

  const rows = useMemo(() => {
    const withCoverage = runs.map((run) => ({ ...run, coverage: run.params?.coverage ?? null }));
    return withCoverage.sort((a, b) => {
      if (a.ndcg_at_k == null && b.ndcg_at_k == null) return a.algorithm.localeCompare(b.algorithm);
      if (a.ndcg_at_k == null) return 1;
      if (b.ndcg_at_k == null) return -1;
      return b.ndcg_at_k - a.ndcg_at_k;
    });
  }, [runs]);

  // "Best" per metric: lowest error, highest ranking quality.
  const best = useMemo(() => {
    const result = {};
    for (const metric of metrics) {
      const values = rows.map((row) => row[metric.key]).filter((value) => value != null);
      if (values.length === 0) continue;
      result[metric.key] = metric.better === "lower" ? Math.min(...values) : Math.max(...values);
    }
    return result;
  }, [rows, metrics]);

  // The sample is shared by every method, but the first row can be the global mean, which scores
  // no users at all; take the largest recorded sample instead.
  const sampled = Math.max(0, ...runs.map((run) => Number(run.params?.sample_users ?? 0)));
  const pairs = runs.find((run) => run.params?.rated_pairs)?.params.rated_pairs;

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Evaluation</p>
        <h1>Measured, not guessed.</h1>
        <p className="lede">
          Every method is scored offline against a held-out slice of MovieLens ratings: the most
          recent 20% of each viewer&apos;s history, with 4★ or higher counting as relevant.
        </p>
      </section>

      {sampled || pairs ? (
        <div className="stat-line">
          {sampled ? (
            <span>
              <strong>{Number(sampled).toLocaleString("en-US")}</strong> users sampled
            </span>
          ) : null}
          {pairs ? (
            <span>
              <strong>{Number(pairs).toLocaleString("en-US")}</strong> held-out ratings
            </span>
          ) : null}
          <span>
            cut-off <strong>k = {k}</strong>
          </span>
        </div>
      ) : null}

      {state.status === "error" ? (
        <div className="panel empty">
          <h3>Evaluation results are unavailable.</h3>
          <p>{state.error}</p>
          <button type="button" className="button" onClick={() => setReloadKey((key) => key + 1)}>
            Try again
          </button>
        </div>
      ) : null}

      {state.status === "loading" ? <p className="page-loading">Loading evaluation results…</p> : null}

      {state.status !== "loading" && state.status !== "error" && rows.length === 0 ? (
        <div className="panel empty">
          <h3>No evaluation runs recorded yet.</h3>
          <p>
            Run the offline evaluation once and the results land here, alongside the README table:
          </p>
          <code className="command">uv run python scripts/evaluate_models.py</code>
        </div>
      ) : null}

      {rows.length > 0 ? (
        <>
          <div className="chart-grid">
            <BarChart title="RMSE" metric="rmse" kind="score" rows={rows} better="lower" />
            <BarChart
              title={`NDCG@${k}`}
              metric="ndcg_at_k"
              kind="percent"
              rows={rows}
              better="higher"
            />
          </div>

          <div className="table-wrap">
            <table className="metrics-table">
              <caption className="sr-only">
                Offline accuracy of each recommendation method at cut-off {k}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  {metrics.map((metric) => (
                    <th scope="col" key={metric.key} className="num">
                      {metric.label}
                      <span className={`metric-dir is-${metric.better}`} aria-hidden="true">
                        {metric.better === "lower" ? "↓" : "↑"}
                      </span>
                      <span className="sr-only">
                        ({metric.better === "lower" ? "lower" : "higher"} is better)
                      </span>
                    </th>
                  ))}
                  <th scope="col" className="num">
                    Trained
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((run) => (
                  <tr key={run.id}>
                    <th scope="row" className="method">
                      {methodName(run.algorithm)}
                    </th>
                    {metrics.map((metric) => {
                      const value = run[metric.key];
                      const isBest =
                        value != null &&
                        best[metric.key] != null &&
                        Math.abs(value - best[metric.key]) < 1e-9;
                      return (
                        <td
                          key={metric.key}
                          data-label={metric.label}
                          className={`num ${isBest ? "is-best" : ""}`}
                        >
                          {isBest ? <span className="sr-only">Best in table: </span> : null}
                          {formatMetric(value, metric.kind)}
                        </td>
                      );
                    })}
                    <td className="num date" data-label="Trained">
                      {formatDate(run.trained_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="table-note">
            Lower RMSE and MAE are better (rating prediction); higher precision, recall, NDCG and
            catalogue coverage are better (ranking). These are offline measurements on a fixed
            chronological split, not live A/B results — those are below.
          </p>
        </>
      ) : null}

      <EngagementPanel />
    </>
  );
}
