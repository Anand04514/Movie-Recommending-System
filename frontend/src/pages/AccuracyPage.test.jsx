import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  listModelRuns: vi.fn(),
}));

import * as api from "../api.js";
import AccuracyPage from "./AccuracyPage.jsx";

const RUNS = [
  {
    id: 4,
    algorithm: "item_cf",
    params: {
      k: 10,
      sample_users: 10_000,
      evaluated_users: 9_766,
      rated_pairs: 2_000_000,
      coverage: 0.1213,
    },
    rmse: 0.9354,
    mae: 0.7121,
    precision_at_k: 0.0791,
    recall_at_k: 0.0512,
    ndcg_at_k: 0.1048,
    trained_at: "2026-10-27T13:54:40",
  },
  {
    id: 5,
    algorithm: "user_cf",
    params: {
      k: 10,
      sample_users: 10_000,
      evaluated_users: 9_766,
      rated_pairs: 2_000_000,
      coverage: 0.2034,
    },
    rmse: 0.9395,
    mae: 0.7137,
    precision_at_k: 0.0825,
    recall_at_k: 0.0521,
    ndcg_at_k: 0.1082,
    trained_at: "2026-10-27T13:54:40",
  },
  {
    id: 2,
    algorithm: "global_mean",
    params: { k: 10, sample_users: 10_000, evaluated_users: 0, rated_pairs: 2_000_000 },
    rmse: 1.0625,
    mae: 0.831,
    precision_at_k: null,
    recall_at_k: null,
    ndcg_at_k: null,
    trained_at: "2026-10-27T13:54:40",
  },
];

async function findRow(name) {
  return screen.findByRole("row", { name: new RegExp(name) });
}

describe("AccuracyPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders every method with readable metrics", async () => {
    api.listModelRuns.mockResolvedValue(RUNS);
    render(<AccuracyPage />);

    const itemRow = await findRow("Item-based CF");
    expect(itemRow).toHaveTextContent("0.935");
    expect(itemRow).toHaveTextContent("7.91%");
    expect(itemRow).toHaveTextContent("Oct 27, 2026");
    expect(screen.getByRole("columnheader", { name: /P@10/i })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /Global mean/ })).toHaveTextContent("—");
  });

  it("marks the best value of each metric", async () => {
    api.listModelRuns.mockResolvedValue(RUNS);
    render(<AccuracyPage />);

    const itemRow = await findRow("Item-based CF");
    const itemBest = [...itemRow.querySelectorAll("td.is-best")].map((cell) => cell.textContent);
    expect(itemBest.some((text) => text.includes("0.935"))).toBe(true);

    const userRow = await findRow("User-based CF");
    const userBest = [...userRow.querySelectorAll("td.is-best")].map((cell) => cell.textContent);
    expect(userBest.some((text) => text.includes("10.82%"))).toBe(true);
  });

  it("charts RMSE and NDCG beside the table", async () => {
    api.listModelRuns.mockResolvedValue(RUNS);
    render(<AccuracyPage />);

    await findRow("Item-based CF");
    expect(screen.getByRole("heading", { name: "RMSE" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "NDCG@10" })).toBeInTheDocument();

    const charts = document.querySelectorAll(".chart-card");
    expect(charts).toHaveLength(2);
    expect(charts[0].querySelectorAll(".chart-row")).toHaveLength(RUNS.length);
    expect(charts[0].querySelector(".chart-row.is-best")).toHaveTextContent("Item-based CF");
    expect(charts[1].querySelector(".chart-row.is-best")).toHaveTextContent("User-based CF");
  });

  it("shows the shared sample size even when the first row is the global mean", async () => {
    api.listModelRuns.mockResolvedValue([RUNS[2], RUNS[0]]);
    render(<AccuracyPage />);

    await findRow("Global mean");
    expect(document.querySelector(".stat-line")).toHaveTextContent("10,000 users sampled");
  });

  it("orders the rows by ranking quality and shows the sample size", async () => {
    api.listModelRuns.mockResolvedValue(RUNS);
    render(<AccuracyPage />);

    await findRow("Item-based CF");
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("User-based CF");
    expect(rows.at(-1)).toHaveTextContent("Global mean");

    expect(document.querySelector(".stat-line")).toHaveTextContent("10,000 users sampled");
    expect(document.querySelector(".stat-line")).toHaveTextContent("2,000,000 held-out ratings");
  });

  it("points at the evaluation script when there is nothing recorded", async () => {
    api.listModelRuns.mockResolvedValue([]);
    render(<AccuracyPage />);

    expect(
      await screen.findByRole("heading", { name: "No evaluation runs recorded yet." }),
    ).toBeInTheDocument();
    expect(screen.getByText("uv run python scripts/evaluate_models.py")).toBeInTheDocument();
  });

  it("retries after a failed load", async () => {
    const user = userEvent.setup();
    api.listModelRuns
      .mockRejectedValueOnce(new Error("Cannot reach the API."))
      .mockResolvedValueOnce(RUNS);
    render(<AccuracyPage />);

    expect(await screen.findByText("Cannot reach the API.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() => expect(screen.getAllByText("Item-based CF").length).toBeGreaterThan(0));
    expect(api.listModelRuns).toHaveBeenCalledTimes(2);
  });
});
