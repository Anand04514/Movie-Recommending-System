import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  searchMovies: vi.fn(),
  addTmdbMovie: vi.fn(),
}));

import * as api from "../api.js";
import SearchPage from "./SearchPage.jsx";

const USER = { id: 7, email: "ada@example.com", display_name: "Ada" };
const CATALOG_MOVIE = {
  id: 4382,
  title: "Dune",
  release_year: 1984,
  rating_count: 1204,
  in_model: true,
  genres: ["Sci-Fi"],
  source: "movielens",
};
const EXTERNAL_HIT = {
  tmdb_id: 424242,
  title: "Synthetic Future",
  release_year: 2025,
  genres: ["Sci-Fi", "Action"],
  overview: "A film newer than the dataset.",
  vote_count: 1200,
  poster_url: null,
};
const ADDED_MOVIE = {
  id: 10424242,
  title: "Synthetic Future",
  release_year: 2025,
  rating_count: 0,
  in_model: false,
  genres: ["Sci-Fi", "Action"],
  source: "tmdb",
};

function searchResponse({ external = [EXTERNAL_HIT], includeExternal = true } = {}) {
  return {
    query: "",
    normalized_query: "",
    count: 1,
    results: [CATALOG_MOVIE],
    external,
    external_source: includeExternal && external.length > 0 ? "tmdb" : null,
  };
}

function renderPage(overrides = {}) {
  const props = {
    user: USER,
    ratingsByMovie: new Map(),
    onRate: vi.fn().mockResolvedValue(undefined),
    onRequireAuth: vi.fn(),
    ...overrides,
  };
  render(<SearchPage {...props} />);
  return props;
}

describe("SearchPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.searchMovies.mockResolvedValue(searchResponse());
  });

  it("offers TMDB films the catalog does not hold as addable cards", async () => {
    renderPage();

    expect(await screen.findByRole("heading", { name: "Dune" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Newer than the dataset" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Synthetic Future" })).toBeInTheDocument();
    expect(screen.getByText("A film newer than the dataset.")).toBeInTheDocument();
    expect(screen.getByText("Not in catalog")).toBeInTheDocument();
  });

  it("adds a film once and makes it rateable in place", async () => {
    const user = userEvent.setup();
    api.addTmdbMovie.mockResolvedValue(ADDED_MOVIE);
    const props = renderPage();

    await screen.findByRole("heading", { name: "Synthetic Future" });
    await user.click(screen.getByRole("button", { name: "Add & rate" }));

    expect(api.addTmdbMovie).toHaveBeenCalledWith(EXTERNAL_HIT.tmdb_id);
    // The external shelf empties: the film is in the catalog now.
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Add & rate" })).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Added")).toBeInTheDocument();
    const ratingWidget = screen.getByRole("radiogroup", { name: "Rate Synthetic Future" });

    await user.click(within(ratingWidget).getByRole("radio", { name: "4 out of 5 stars" }));
    expect(props.onRate).toHaveBeenCalledWith(
      expect.objectContaining({ id: ADDED_MOVIE.id }),
      4,
    );
  });

  it("asks signed-out visitors to sign in before adding anything", async () => {
    const user = userEvent.setup();
    const props = renderPage({ user: null });

    await screen.findByRole("heading", { name: "Synthetic Future" });
    await user.click(screen.getByRole("button", { name: "Add & rate" }));

    expect(props.onRequireAuth).toHaveBeenCalledWith("Sign in to add a film and rate it.");
    expect(api.addTmdbMovie).not.toHaveBeenCalled();
  });

  it("reports why a film could not be added and keeps the search results", async () => {
    const user = userEvent.setup();
    api.addTmdbMovie.mockRejectedValue(
      new Error("TMDB is not configured on this server, so new films cannot be added."),
    );
    renderPage();

    await screen.findByRole("heading", { name: "Synthetic Future" });
    await user.click(screen.getByRole("button", { name: "Add & rate" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/TMDB is not configured/);
    expect(screen.getByRole("heading", { name: "Synthetic Future" })).toBeInTheDocument();
  });

  it("keeps the catalog results when TMDB has nothing to add", async () => {
    api.searchMovies.mockResolvedValue(searchResponse({ external: [], includeExternal: false }));
    renderPage();

    expect(await screen.findByRole("heading", { name: "Dune" })).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Newer than the dataset" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add & rate" })).not.toBeInTheDocument();
  });
});
