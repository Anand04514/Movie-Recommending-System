import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  compareAlgorithms: vi.fn(),
}));

import * as api from "../api.js";
import ComparePage from "./ComparePage.jsx";

const USER = { id: 7, email: "ada@example.com", display_name: "Ada" };

function movie(id, title, year) {
  return { id, title, release_year: year, rating_count: 100, genres: ["Drama"] };
}

function pick(id, title, predicted = 4.5) {
  return {
    movie: movie(id, title, 2000),
    score: 1.2,
    predicted_rating: predicted,
    reason: "Because you rated Heat 5★",
    evidence: [],
  };
}

const ALPHA = pick(1, "Alpha", 4.7);
const BETA = pick(2, "Beta", 4.3);
const GAMMA = pick(3, "Gamma", 4.1);

function response(overrides = {}) {
  return {
    limit: 10,
    left: { algorithm: "item_cf", cold_start: false, recommendations: [ALPHA, BETA] },
    right: { algorithm: "user_cf", cold_start: false, recommendations: [BETA, GAMMA] },
    overlap: [BETA.movie.id],
    overlap_count: 1,
    ...overrides,
  };
}

function renderPage(overrides = {}) {
  const props = {
    user: USER,
    onRequireAuth: vi.fn(),
    onExplore: vi.fn(),
    ...overrides,
  };
  render(<ComparePage {...props} />);
  return props;
}

describe("ComparePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("asks signed-out visitors to sign in without comparing", async () => {
    const user = userEvent.setup();
    const props = renderPage({ user: null });

    expect(screen.getByRole("heading", { name: "Two methods, your taste." })).toBeInTheDocument();
    expect(api.compareAlgorithms).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(props.onRequireAuth).toHaveBeenCalledWith(
      "Sign in to compare algorithms made for you.",
    );
  });

  it("renders both algorithms with their picks and marks the overlap", async () => {
    api.compareAlgorithms.mockResolvedValue(response());
    renderPage();

    expect(
      await screen.findByRole("heading", { name: "Item-based CF" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "User-based CF" })).toBeInTheDocument();

    const columns = screen.getAllByRole("region");
    expect(within(columns[0]).getByText("Alpha")).toBeInTheDocument();
    expect(within(columns[1]).getByText("Gamma")).toBeInTheDocument();

    // Beta appears in both lists, so each column marks it (the lede also shows the badge).
    expect(within(columns[0]).getByText("Both")).toBeInTheDocument();
    expect(within(columns[1]).getByText("Both")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("1 of 2 picks appear in both lists");
  });

  it("says so when the two lists share nothing", async () => {
    api.compareAlgorithms.mockResolvedValue(
      response({
        right: { algorithm: "user_cf", cold_start: false, recommendations: [GAMMA] },
        overlap: [],
        overlap_count: 0,
      }),
    );
    renderPage();

    expect(
      await screen.findByText(/picked completely different films this time/i),
    ).toBeInTheDocument();
    for (const column of screen.getAllByRole("region")) {
      expect(within(column).queryByText("Both")).not.toBeInTheDocument();
    }
  });

  it("explains that a cold start collapses both lists into popularity", async () => {
    api.compareAlgorithms.mockResolvedValue(
      response({
        left: { algorithm: "popularity", cold_start: true, recommendations: [ALPHA] },
        right: { algorithm: "popularity", cold_start: true, recommendations: [ALPHA] },
        overlap: [ALPHA.movie.id],
        overlap_count: 1,
      }),
    );
    renderPage();

    expect(
      await screen.findByRole("heading", { name: /make this a real comparison/i }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { name: "Popularity baseline" })).toHaveLength(2);
  });

  it("lets each side pick its own model and keeps them distinct", async () => {
    const user = userEvent.setup();
    api.compareAlgorithms.mockResolvedValue(response());
    renderPage();

    await screen.findByText("Alpha");
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Left algorithm" }),
      "svd",
    );
    await waitFor(() =>
      expect(api.compareAlgorithms).toHaveBeenLastCalledWith(
        expect.objectContaining({ left: "svd", right: "user_cf" }),
      ),
    );

    // Choosing the algorithm the other side already uses swaps them rather than sending a
    // comparison the API would reject.
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Right algorithm" }),
      "svd",
    );
    await waitFor(() =>
      expect(api.compareAlgorithms).toHaveBeenLastCalledWith(
        expect.objectContaining({ left: "user_cf", right: "svd" }),
      ),
    );
  });

  it("refetches with the chosen list length", async () => {
    const user = userEvent.setup();
    api.compareAlgorithms.mockResolvedValue(response());
    renderPage();

    await screen.findByText("Alpha");
    await user.selectOptions(screen.getByRole("combobox", { name: /list length/i }), "15");

    await waitFor(() =>
      expect(api.compareAlgorithms).toHaveBeenLastCalledWith(expect.objectContaining({ limit: 15 })),
    );
  });

  it("retries after a failed comparison", async () => {
    const user = userEvent.setup();
    api.compareAlgorithms
      .mockRejectedValueOnce(new Error("Artifacts are missing."))
      .mockResolvedValueOnce(response());
    renderPage();

    expect(await screen.findByText("Artifacts are missing.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByText("Alpha")).toBeInTheDocument();
    expect(api.compareAlgorithms).toHaveBeenCalledTimes(2);
  });
});
