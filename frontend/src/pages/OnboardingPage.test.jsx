import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  getOnboarding: vi.fn(),
}));

import * as api from "../api.js";
import OnboardingPage from "./OnboardingPage.jsx";

const USER = { id: 7, email: "ada@example.com", display_name: "Ada" };

function movie(id, title) {
  return {
    id,
    title,
    release_year: 1999,
    rating_count: 1200,
    genres: ["Drama"],
  };
}

const MOVIES = [
  movie(1, "Alpha"),
  movie(2, "Beta"),
  movie(3, "Gamma"),
  movie(4, "Delta"),
  movie(5, "Epsilon"),
  movie(6, "Zeta"),
];

function renderPage(overrides = {}) {
  const props = {
    user: USER,
    ratingsByMovie: new Map(),
    onRate: vi.fn().mockResolvedValue(undefined),
    onDone: vi.fn(),
    onSkip: vi.fn(),
    onRequireAuth: vi.fn(),
    ...overrides,
  };
  render(<OnboardingPage {...props} />);
  return props;
}

describe("OnboardingPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getOnboarding.mockResolvedValue({ target: 5, movies: MOVIES });
  });

  it("asks signed-out visitors to sign in", () => {
    renderPage({ user: null });
    expect(
      screen.getByRole("heading", { name: "Start with five films." }),
    ).toBeInTheDocument();
    expect(api.getOnboarding).not.toHaveBeenCalled();
  });

  it("shows the diverse shortlist with progress from existing ratings", async () => {
    renderPage({ ratingsByMovie: new Map([[1, 4.5], [2, 3]]) });

    expect(await screen.findByRole("heading", { name: "Alpha" })).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Onboarding progress" })).toHaveAttribute(
      "aria-valuenow",
      "2",
    );
    expect(screen.getByText(/Rate 3 more to unlock/)).toBeInTheDocument();
    expect(api.getOnboarding).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 12 }),
    );
  });

  it("rates a film through the shared rating handler", async () => {
    const user = userEvent.setup();
    const props = renderPage();
    const article = (await screen.findByRole("heading", { name: "Alpha" })).closest("article");

    await user.click(within(article).getByRole("radio", { name: "4.5 out of 5 stars" }));

    expect(props.onRate).toHaveBeenCalledWith(MOVIES[0], 4.5);
  });

  it("invites completion once five films are rated", async () => {
    const user = userEvent.setup();
    const props = renderPage({
      ratingsByMovie: new Map([[1, 5], [2, 4], [3, 3], [4, 2.5], [5, 1]]),
    });

    expect(
      await screen.findByRole("heading", { name: "That is enough to start." }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "See my recommendations" }));
    expect(props.onDone).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Skip for now" }));
    expect(props.onSkip).toHaveBeenCalledTimes(1);
  });

  it("retries after a failed fetch", async () => {
    const user = userEvent.setup();
    api.getOnboarding
      .mockRejectedValueOnce(new Error("Cannot reach the API."))
      .mockResolvedValueOnce({ target: 5, movies: MOVIES });
    renderPage();

    expect(await screen.findByText("Cannot reach the API.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Alpha" })).toBeInTheDocument();
    await waitFor(() => expect(api.getOnboarding).toHaveBeenCalledTimes(2));
  });

  it("reports a rating failure without losing the page", async () => {
    const user = userEvent.setup();
    renderPage({ onRate: vi.fn().mockRejectedValue(new Error("Cannot save that rating.")) });
    const article = (await screen.findByRole("heading", { name: "Alpha" })).closest("article");

    await user.click(within(article).getByRole("radio", { name: "3 out of 5 stars" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot save that rating.");
    expect(screen.getByRole("heading", { name: "Alpha" })).toBeInTheDocument();
  });
});
