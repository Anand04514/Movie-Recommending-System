import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", () => ({
  getMovie: vi.fn(),
}));

import * as api from "../api.js";
import MovieCard from "./MovieCard.jsx";

const MOVIE = {
  id: 2571,
  title: "The Matrix",
  release_year: 1999,
  rating_count: 27134,
  genres: ["Action", "Sci-Fi", "Thriller", "Drama"],
};

describe("MovieCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the title, year, rating count, and at most three genres", () => {
    render(<MovieCard movie={MOVIE} />);

    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
    const card = screen.getByRole("article");
    expect(card).toHaveTextContent("1999");
    expect(card).toHaveTextContent(`${MOVIE.rating_count.toLocaleString()} ratings`);

    const genres = within(screen.getByRole("list", { name: "Genres" })).getAllByRole("listitem");
    expect(genres.map((item) => item.textContent)).toEqual(["Action", "Sci-Fi", "Thriller"]);
    expect(screen.queryByText("Drama")).not.toBeInTheDocument();
  });

  it("degrades gracefully when the release year is unknown", () => {
    render(<MovieCard movie={{ ...MOVIE, release_year: null }} />);

    expect(screen.getByRole("article")).toHaveTextContent("Year unknown");
  });

  it("shows the reason and predicted rating for a recommendation", () => {
    render(
      <MovieCard
        movie={MOVIE}
        reason="Because you rated Heat 5★"
        predictedRating={4.75}
        badge="New"
      />,
    );

    expect(screen.getByText(/Because you rated Heat 5★/)).toBeInTheDocument();
    expect(screen.getByText("Predicted 4.8★")).toBeInTheDocument();
    expect(screen.getByText("New")).toBeInTheDocument();
  });

  it("passes the movie and the clicked half-star value to onRate", async () => {
    const user = userEvent.setup();
    const onRate = vi.fn();
    render(<MovieCard movie={MOVIE} onRate={onRate} />);

    expect(screen.getByRole("radiogroup", { name: "Rate The Matrix" })).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "4.5 out of 5 stars" }));

    expect(onRate).toHaveBeenCalledTimes(1);
    expect(onRate).toHaveBeenCalledWith(MOVIE, 4.5);
  });

  it("offers remove only for movies already rated", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    const { rerender } = render(<MovieCard movie={MOVIE} onRemove={onRemove} />);

    expect(screen.queryByRole("button", { name: "Remove" })).not.toBeInTheDocument();

    rerender(<MovieCard movie={MOVIE} rating={3.5} onRemove={onRemove} />);
    await user.click(screen.getByRole("button", { name: "Remove" }));

    expect(onRemove).toHaveBeenCalledWith(MOVIE);
  });

  it("offers like and reject actions only when feedback is wired up", async () => {
    const user = userEvent.setup();
    const onFeedback = vi.fn();
    const { rerender } = render(<MovieCard movie={MOVIE} />);

    expect(
      screen.queryByRole("button", { name: "Not interested in The Matrix" }),
    ).not.toBeInTheDocument();

    rerender(<MovieCard movie={MOVIE} onFeedback={onFeedback} />);
    await user.click(screen.getByRole("button", { name: "Like The Matrix" }));
    await user.click(screen.getByRole("button", { name: "Not interested in The Matrix" }));

    expect(onFeedback).toHaveBeenNthCalledWith(1, MOVIE, "liked");
    expect(onFeedback).toHaveBeenNthCalledWith(2, MOVIE, "rejected");
  });

  it("disables feedback while the reaction is saving", () => {
    render(<MovieCard movie={MOVIE} onFeedback={vi.fn()} feedbackBusy />);

    expect(screen.getByRole("button", { name: "Like The Matrix" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Not interested in The Matrix" })).toBeDisabled();
  });

  it("locks the rating widget while a save is in flight", () => {
    render(<MovieCard movie={MOVIE} busy onRate={vi.fn()} />);

    expect(screen.getByRole("article")).toHaveAttribute("aria-busy", "true");
    expect(screen.getAllByRole("radio").every((radio) => radio.disabled)).toBe(true);
  });

  it("opens full details from the title", async () => {
    const user = userEvent.setup();
    api.getMovie.mockResolvedValue({
      ...MOVIE,
      overview: "A hacker learns the world is a simulation.",
      poster_url: null,
    });
    render(<MovieCard movie={MOVIE} onRate={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "The Matrix" }));

    expect(
      await screen.findByText("A hacker learns the world is a simulation."),
    ).toBeInTheDocument();
    expect(api.getMovie).toHaveBeenCalledWith(MOVIE.id, expect.anything());
  });

  it("shows the TMDB poster when the catalog has one", () => {
    const poster = "https://image.tmdb.org/t/p/w342/matrix.jpg";
    const { container } = render(<MovieCard movie={{ ...MOVIE, poster_url: poster }} />);

    const image = container.querySelector("img.poster");
    expect(image).toHaveAttribute("src", poster);
    expect(image).toHaveAttribute("loading", "lazy");
    expect(image).toHaveAttribute("alt", "");
  });

  it("drops a poster that fails to load instead of showing a broken image", () => {
    const { container } = render(
      <MovieCard movie={{ ...MOVIE, poster_url: "https://image.tmdb.org/t/p/w342/missing.jpg" }} />,
    );

    fireEvent.error(container.querySelector("img.poster"));
    expect(container.querySelector("img.poster")).toBeNull();
    expect(screen.getByRole("heading", { name: "The Matrix" })).toBeInTheDocument();
  });

  it("renders no poster element when the movie has no artwork", () => {
    const { container } = render(<MovieCard movie={MOVIE} />);

    expect(container.querySelector("img.poster")).toBeNull();
  });

  it("lists the runner-up ratings behind a recommendation", () => {
    const heat = { ...MOVIE, id: 949, title: "Heat" };
    const inception = { ...MOVIE, id: 27205, title: "Inception" };
    render(
      <MovieCard
        movie={MOVIE}
        reason="Because you rated Heat 5★"
        evidence={[
          { movie: heat, rating: 5, strength: 1.2, shared_genres: ["Action", "Crime"] },
          { movie: inception, rating: 4.5, strength: 0.8, shared_genres: ["Sci-Fi"] },
        ]}
      />,
    );

    expect(screen.getByText("Also driven by")).toBeInTheDocument();
    const list = screen.getByRole("list", { name: "Other ratings behind this pick" });
    expect(within(list).getByText("Inception")).toBeInTheDocument();
    expect(within(list).getByText("4.5★")).toBeInTheDocument();
    expect(within(list).getByText("Sci-Fi")).toBeInTheDocument();
    // Heat is already named in the reason, so it is not repeated as a chip.
    expect(within(list).queryByText("Heat")).not.toBeInTheDocument();
  });

  it("omits the evidence block when nothing else drove the pick", () => {
    render(<MovieCard movie={MOVIE} reason="Popular with MovieLens users" evidence={[]} />);

    expect(screen.queryByText("Also driven by")).not.toBeInTheDocument();
  });
});
