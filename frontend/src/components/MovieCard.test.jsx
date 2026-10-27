import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import MovieCard from "./MovieCard.jsx";

const MOVIE = {
  id: 2571,
  title: "The Matrix",
  release_year: 1999,
  rating_count: 27134,
  genres: ["Action", "Sci-Fi", "Thriller", "Drama"],
};

describe("MovieCard", () => {
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

  it("locks the rating widget while a save is in flight", () => {
    render(<MovieCard movie={MOVIE} busy onRate={vi.fn()} />);

    expect(screen.getByRole("article")).toHaveAttribute("aria-busy", "true");
    expect(screen.getAllByRole("radio").every((radio) => radio.disabled)).toBe(true);
  });
});
