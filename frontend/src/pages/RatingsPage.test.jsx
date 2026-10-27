import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import RatingsPage from "./RatingsPage.jsx";

const USER = { id: 7, email: "ada@example.com", display_name: "Ada" };
const MATRIX = {
  id: 2571,
  title: "The Matrix",
  release_year: 1999,
  rating_count: 93808,
  genres: ["Action", "Sci-Fi"],
};
const GODFATHER = {
  id: 858,
  title: "The Godfather",
  release_year: 1972,
  rating_count: 66440,
  genres: ["Crime", "Drama"],
};

function buildProps(overrides = {}) {
  return {
    user: USER,
    ratings: [{ movie: MATRIX, rating: 4.5 }, { movie: GODFATHER, rating: 5 }],
    loading: false,
    error: null,
    onRate: vi.fn().mockResolvedValue(undefined),
    onRemove: vi.fn().mockResolvedValue(undefined),
    onRequireAuth: vi.fn(),
    onExplore: vi.fn(),
    onRetry: vi.fn(),
    ...overrides,
  };
}

describe("RatingsPage", () => {
  it("asks signed-out visitors to sign in", async () => {
    const user = userEvent.setup();
    const props = buildProps({ user: null });
    render(<RatingsPage {...props} />);

    expect(
      screen.getByRole("heading", { name: "Your ratings, kept in one place." }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(props.onRequireAuth).toHaveBeenCalledWith("Sign in to see your ratings.");
  });

  it("shows skeletons while the library loads instead of claiming it is empty", () => {
    render(<RatingsPage {...buildProps({ ratings: [], loading: true })} />);

    expect(screen.getByText("Loading your ratings…")).toBeInTheDocument();
    expect(screen.queryByText("No ratings yet.")).not.toBeInTheDocument();
    expect(document.querySelector(".movie-grid")).toHaveAttribute("aria-busy", "true");
    expect(document.querySelectorAll(".movie-card.skeleton").length).toBe(3);
  });

  it("surfaces a load failure and retries on demand", async () => {
    const user = userEvent.setup();
    const props = buildProps({ ratings: [], error: "Cannot reach the API." });
    render(<RatingsPage {...props} />);

    expect(screen.getByRole("alert")).toHaveTextContent("We couldn't load your ratings.");
    expect(screen.getByText("Cannot reach the API.")).toBeInTheDocument();
    expect(screen.queryByText("No ratings yet.")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(props.onRetry).toHaveBeenCalledTimes(1);
  });

  it("invites an empty library to start rating", async () => {
    const user = userEvent.setup();
    const props = buildProps({ ratings: [] });
    render(<RatingsPage {...props} />);

    expect(screen.getByRole("heading", { name: "No ratings yet." })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Find films to rate" }));
    expect(props.onExplore).toHaveBeenCalledTimes(1);
  });

  it("lists rated films with the average", () => {
    render(<RatingsPage {...buildProps()} />);

    expect(screen.getByRole("heading", { name: "My ratings" })).toBeInTheDocument();
    expect(screen.getByText("2 films rated · average 4.75★")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The Godfather" })).toBeInTheDocument();
    expect(screen.getByText("4.5")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Remove" })).toHaveLength(2);
  });

  it("keeps the list when a removal fails and reports why", async () => {
    const user = userEvent.setup();
    const props = buildProps({
      onRemove: vi.fn().mockRejectedValue(new Error("Cannot remove that rating.")),
    });
    render(<RatingsPage {...props} />);

    await user.click(screen.getAllByRole("button", { name: "Remove" })[0]);

    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot remove that rating.");
    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
  });
});
