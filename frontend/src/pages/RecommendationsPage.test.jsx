import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  getRecommendations: vi.fn(),
  listGenres: vi.fn(),
  sendFeedback: vi.fn(),
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
  evidence: [
    {
      movie: {
        id: 949,
        title: "Heat",
        release_year: 1995,
        rating_count: 1200,
        genres: ["Action", "Crime"],
      },
      rating: 5,
      strength: 1.2,
      shared_genres: ["Action"],
    },
    {
      movie: {
        id: 27205,
        title: "Inception",
        release_year: 2010,
        rating_count: 900,
        genres: ["Action", "Sci-Fi"],
      },
      rating: 4.5,
      strength: 0.8,
      shared_genres: ["Action", "Sci-Fi"],
    },
  ],
};

function buildProps(overrides = {}) {
  return {
    user: USER,
    refreshKey: 0,
    ratingsByMovie: new Map(),
    onRate: vi.fn().mockResolvedValue(undefined),
    onRequireAuth: vi.fn(),
    onExplore: vi.fn(),
    onOnboard: vi.fn(),
    ...overrides,
  };
}

function renderPage(overrides = {}) {
  const props = buildProps(overrides);
  render(<RecommendationsPage {...props} />);
  return props;
}

function emptyResponse() {
  return {
    algorithm: "item_cf",
    cold_start: false,
    filters: {},
    candidates_considered: 0,
    recommendations: [],
  };
}

function respond({
  algorithm = "item_cf",
  coldStart = false,
  recommendations = [PICK],
  needsOnboarding = false,
} = {}) {
  api.getRecommendations.mockResolvedValue({
    algorithm,
    cold_start: coldStart,
    needs_onboarding: needsOnboarding,
    filters: {},
    candidates_considered: 24,
    recommendations,
  });
}

describe("RecommendationsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listGenres.mockResolvedValue(["Action", "Drama", "Sci-Fi"]);
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

  it("asks for the hybrid model by default and labels the response", async () => {
    respond();
    renderPage();

    expect(await screen.findByText("Item-based collaborative filtering")).toBeInTheDocument();
    expect(api.getRecommendations).toHaveBeenCalledWith(
      expect.objectContaining({ algorithm: "hybrid" }),
    );
    expect(screen.getByText(/Because you rated Heat 5★/)).toBeInTheDocument();
    expect(screen.getByText("Predicted 4.8★")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /this is the crowd/i })).not.toBeInTheDocument();
  });

  it("switches model without changing the filters", async () => {
    const user = userEvent.setup();
    respond({ algorithm: "svd" });
    renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Recommendation model" }),
      "svd",
    );

    await waitFor(() =>
      expect(api.getRecommendations).toHaveBeenLastCalledWith(
        expect.objectContaining({ algorithm: "svd", genres: [] }),
      ),
    );
    // The one-line explanation must describe the model that actually answered.
    expect(screen.getByText(/^Matrix factorization/)).toBeInTheDocument();
    expect(screen.queryByText(/^Films that people who rated/)).not.toBeInTheDocument();
  });

  it("removes a film immediately when it is rejected", async () => {
    const user = userEvent.setup();
    api.sendFeedback.mockResolvedValue({ movie_id: PICK.movie.id, action: "rejected" });
    respond();
    renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    await user.click(screen.getByRole("button", { name: "Not interested in The Matrix" }));

    expect(api.sendFeedback).toHaveBeenCalledWith(PICK.movie.id, "rejected", "item_cf");
    await waitFor(() =>
      expect(screen.queryByRole("heading", { name: "The Matrix" })).not.toBeInTheDocument(),
    );
    expect(await screen.findByText(/The Matrix will not be shown again/)).toBeInTheDocument();
  });

  it("likes a pick and keeps the list failure-free", async () => {
    const user = userEvent.setup();
    api.sendFeedback.mockResolvedValue({ movie_id: PICK.movie.id, action: "liked" });
    respond();
    renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    await user.click(screen.getByRole("button", { name: "Like The Matrix" }));

    expect(api.sendFeedback).toHaveBeenCalledWith(PICK.movie.id, "liked", "item_cf");
    expect(await screen.findByText(/we will look for more like it/)).toBeInTheDocument();
  });

  it("points a new user at onboarding", async () => {
    const user = userEvent.setup();
    respond({ needsOnboarding: true });
    const props = renderPage();

    expect(
      await screen.findByRole("heading", { name: "Five ratings in, this becomes yours." }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start onboarding" }));
    expect(props.onOnboard).toHaveBeenCalledTimes(1);
  });

  it("labels a back-of-list pick as discovery", async () => {
    respond({
      recommendations: [
        PICK,
        {
          ...PICK,
          movie: { ...MOVIE, id: 99, title: "A Weird Little Film", genres: ["Horror"] },
          discovery: true,
          reason: "Discovery — a Horror pick outside your usual Drama",
        },
      ],
    });
    renderPage();

    expect(await screen.findByText("Discovery")).toBeInTheDocument();
    expect(
      screen.getByText(/outside your usual Drama/),
    ).toBeInTheDocument();
  });

  it("shows the runner-up ratings behind a pick", async () => {
    respond();
    renderPage();

    expect(await screen.findByText("Also driven by")).toBeInTheDocument();
    const evidence = screen.getByRole("list", { name: "Other ratings behind this pick" });
    expect(within(evidence).getByText("Inception")).toBeInTheDocument();
    expect(within(evidence).getByText("4.5★")).toBeInTheDocument();
    expect(within(evidence).getByText("Action · Sci-Fi")).toBeInTheDocument();
    // The reason already names the strongest rating, so it is not repeated as a chip.
    expect(within(evidence).queryAllByRole("listitem")).toHaveLength(1);
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
        filters: {},
        candidates_considered: 10,
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

  it("turns a genre chip into a filtered request", async () => {
    const user = userEvent.setup();
    respond();
    renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    await user.click(screen.getByRole("button", { name: "Drama" }));

    await waitFor(() =>
      expect(api.getRecommendations).toHaveBeenLastCalledWith(
        expect.objectContaining({ genres: ["Drama"], yearMin: null, yearMax: null, minScore: null }),
      ),
    );
    expect(screen.getByRole("button", { name: "Drama" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Reset" })).toBeInTheDocument();
  });

  it("sends the year range and minimum predicted score", async () => {
    respond();
    renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    fireEvent.change(screen.getByLabelText("Earliest release year"), {
      target: { value: "1990" },
    });
    fireEvent.change(screen.getByLabelText("Latest release year"), { target: { value: "2005" } });
    fireEvent.change(screen.getByRole("slider", { name: "Minimum predicted rating" }), {
      target: { value: "4" },
    });

    await waitFor(() =>
      expect(api.getRecommendations).toHaveBeenLastCalledWith(
        expect.objectContaining({ yearMin: 1990, yearMax: 2005, minScore: 4 }),
      ),
    );
  });

  it("keeps the last list and explains an impossible year range", async () => {
    respond();
    renderPage();

    await screen.findByRole("heading", { name: "The Matrix" });
    fireEvent.change(screen.getByLabelText("Earliest release year"), {
      target: { value: "2000" },
    });
    fireEvent.change(screen.getByLabelText("Latest release year"), { target: { value: "1990" } });

    expect(await screen.findByText("The first year must come before the second.")).toBeVisible();
    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
    expect(api.getRecommendations).toHaveBeenCalledTimes(1);
  });

  it("offers a way out when the filters exclude every candidate", async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValue(emptyResponse());
    renderPage();

    expect(
      await screen.findByRole("heading", { name: "Nothing left to suggest." }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Drama" }));
    expect(
      await screen.findByRole("heading", { name: "No picks match those filters." }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await waitFor(() =>
      expect(api.getRecommendations).toHaveBeenLastCalledWith(
        expect.objectContaining({ genres: [], yearMin: null, yearMax: null, minScore: null }),
      ),
    );
  });
});
