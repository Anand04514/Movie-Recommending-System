import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  getMovie: vi.fn(),
}));

import * as api from "../api.js";
import MovieDialog from "./MovieDialog.jsx";

const MOVIE = {
  id: 2571,
  title: "The Matrix",
  release_year: 1999,
  rating_count: 27_134,
  genres: ["Action", "Sci-Fi"],
};

const DETAIL = {
  ...MOVIE,
  genres: ["Action", "Sci-Fi", "Thriller"],
  overview: "A hacker discovers the world is a simulation.",
  poster_url: "https://image.tmdb.org/t/p/w342/matrix.jpg",
};

describe("MovieDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows the card's facts immediately and loads the synopsis", async () => {
    api.getMovie.mockResolvedValue(DETAIL);
    const { container } = render(<MovieDialog movie={MOVIE} onClose={vi.fn()} />);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
    expect(await screen.findByText(DETAIL.overview)).toBeInTheDocument();

    expect(api.getMovie).toHaveBeenCalledWith(2571, expect.objectContaining({ signal: expect.anything() }));
    const genres = within(screen.getByRole("list", { name: "Genres" }))
      .getAllByRole("listitem")
      .map((item) => item.textContent);
    expect(genres).toEqual(["Action", "Sci-Fi", "Thriller"]);
    expect(container.querySelector(".dialog-poster")).toHaveAttribute("src", DETAIL.poster_url);
    expect(screen.getByText(/poster and synopsis via TMDB/i)).toBeInTheDocument();
  });

  it("explains a missing synopsis instead of leaving a blank panel", async () => {
    api.getMovie.mockResolvedValue({ ...DETAIL, overview: null, poster_url: null });
    render(<MovieDialog movie={MOVIE} onClose={vi.fn()} />);

    expect(await screen.findByText(/no synopsis in the catalog yet/i)).toBeInTheDocument();
    expect(screen.queryByText(/via TMDB/i)).not.toBeInTheDocument();
  });

  it("keeps the card usable when details fail to load", async () => {
    api.getMovie.mockRejectedValue(new Error("Cannot reach the API."));
    render(<MovieDialog movie={MOVIE} onClose={vi.fn()} />);

    expect(await screen.findByText(/details are unavailable right now/i)).toBeInTheDocument();
  });

  it("passes a half-star rating back to the page", async () => {
    const user = userEvent.setup();
    api.getMovie.mockResolvedValue(DETAIL);
    const onRate = vi.fn();
    render(<MovieDialog movie={MOVIE} rating={3} onRate={onRate} onClose={vi.fn()} />);

    await screen.findByText(DETAIL.overview);
    await user.click(screen.getByRole("radio", { name: "4.5 out of 5 stars" }));

    expect(onRate).toHaveBeenCalledWith(DETAIL, 4.5);
  });

  it("closes from its own button", async () => {
    const user = userEvent.setup();
    api.getMovie.mockResolvedValue(DETAIL);
    const onClose = vi.fn();
    render(<MovieDialog movie={MOVIE} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: "Close details" }));
    expect(onClose).toHaveBeenCalled();
  });
});
