import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  getRecommendations: vi.fn(),
}));

import * as api from "../api.js";
import RecommendationsPage from "./RecommendationsPage.jsx";

const USER = { id: 7, email: "ada@example.com", display_name: "Ada" };
const MOVIE = {
  id: 2571,
  title: "The Matrix",
  release_year: 1999,
  rating_count: 27134,
  genres: ["Action", "Sci-Fi"],
};
const PICK = {
  movie: MOVIE,
  score: 3.42,
  predicted_rating: 4.8,
  reason: "Because you rated Heat 5★",
};

function buildProps(overrides = {}) {
  return {
    user: USER,
    refreshKey: 0,
    ratingsByMovie: new Map(),
    onRate: vi.fn().mockResolvedValue(undefined),
    onRequireAuth: vi.fn(),
    onExplore: vi.fn(),
    ...overrides,
  };
}

function renderPage(overrides = {}) {
  const props = buildProps(overrides);
  render(<RecommendationsPage {...props} />);
  return props;
}

function respond({ algorithm = "item_cf", coldStart = false, recommendations = [PICK] } = {}) {
  api.getRecommendations.mockResolvedValue({
    algorithm,
    cold_start: coldStart,
    recommendations,
  });
}

describe("RecommendationsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("asks signed-out visitors to sign in instead of fetching picks", async () => {
    const user = userEvent.setup();
    const props = renderPage({ user: null });

    expect(
      screen.getByRole("heading", { name: "Your picks live behind the door." }),
    ).toBeInTheDocument();
    expect(api.getRecommendations).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(props.onRequireAuth).toHaveBeenCalledWith("Sign in to see picks made for you.");
  });

  it("labels the popularity fallback and shows the cold-start onboarding", async () => {
    respond({
      algorithm: "popularity",
      coldStart: true,
      recommendations: [{ ...PICK, reason: "Popular with MovieLens users" }],
    });
    renderPage();

    expect(await screen.findByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
    expect(screen.getByText("Popularity baseline")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /this is the crowd/i })).toBeInTheDocument();
    expect(screen.getByText(/Popular with MovieLens users/)).toBeInTheDocument();
    expect(api.getRecommendations).toHaveBeenCalledTimes(1);
  });

  it("labels item-based CF and shows a reason with the predicted score", async () => {
    respond();
    renderPage();

    expect(await screen.findByText("Item-based collaborative filtering")).toBeInTheDocument();
    expect(screen.getByText(/Because you rated Heat 5★/)).toBeInTheDocument();
    expect(screen.getByText("Predicted 4.8★")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /this is the crowd/i })).not.toBeInTheDocument();
  });

  it("sends the clicked half-star rating for a recommended movie", async () => {
    const user = userEvent.setup();
    respond();
    const props = renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    await user.click(screen.getByRole("radio", { name: "4.5 out of 5 stars" }));

    expect(props.onRate).toHaveBeenCalledWith(MOVIE, 4.5);
  });

  it("surfaces a failed save without dropping the list", async () => {
    const user = userEvent.setup();
    respond();
    renderPage({ onRate: vi.fn().mockRejectedValue(new Error("Cannot save that rating.")) });

    await screen.findByRole("heading", { name: "The Matrix" });
    await user.click(screen.getByRole("radio", { name: "3 out of 5 stars" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot save that rating.");
    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
  });

  it("retries after a failed fetch", async () => {
    const user = userEvent.setup();
    api.getRecommendations
      .mockRejectedValueOnce(new Error("Artifacts are missing."))
      .mockResolvedValueOnce({
        algorithm: "item_cf",
        cold_start: false,
        recommendations: [PICK],
      });
    renderPage();

    expect(await screen.findByText("Artifacts are missing.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
    expect(api.getRecommendations).toHaveBeenCalledTimes(2);
  });

  it("refetches when the rating clock ticks", async () => {
    respond();
    const props = buildProps();
    const { rerender } = render(<RecommendationsPage {...props} refreshKey={0} />);

    await screen.findByRole("heading", { name: "The Matrix" });
    rerender(<RecommendationsPage {...props} refreshKey={1} />);

    await waitFor(() => expect(api.getRecommendations).toHaveBeenCalledTimes(2));
  });
});
