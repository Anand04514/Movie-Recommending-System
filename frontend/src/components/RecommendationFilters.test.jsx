import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import RecommendationFilters, {
  EMPTY_FILTERS,
  isFiltered,
  toRequestFilters,
} from "./RecommendationFilters.jsx";

const GENRES = ["Action", "Drama", "Sci-Fi"];

function cleanValue(overrides = {}) {
  return { ...EMPTY_FILTERS, genres: [], ...overrides };
}

function setup(value = cleanValue(), overrides = {}) {
  const onChange = vi.fn();
  const onReset = vi.fn();
  const props = {
    genres: GENRES,
    value,
    onChange,
    onReset,
    error: null,
    ...overrides,
  };
  const { rerender } = render(<RecommendationFilters {...props} />);
  return { onChange, onReset, rerender, props };
}

describe("RecommendationFilters", () => {
  it("adds and removes a genre as a toggle", async () => {
    const user = userEvent.setup();
    const { onChange, rerender, props } = setup();

    await user.click(screen.getByRole("button", { name: "Drama" }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ genres: ["Drama"] }));

    onChange.mockClear();
    rerender(<RecommendationFilters {...props} value={cleanValue({ genres: ["Drama"] })} />);
    await user.click(screen.getByRole("button", { name: "Drama" }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ genres: [] }));
  });

  it("marks selected genres with aria-pressed", () => {
    setup(cleanValue({ genres: ["Drama"] }));

    expect(screen.getByRole("button", { name: "Drama" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Action" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("passes typed years through as raw strings", () => {
    const { onChange } = setup();

    fireEvent.change(screen.getByLabelText("Earliest release year"), {
      target: { value: "1999" },
    });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ yearMin: "1999" }));

    fireEvent.change(screen.getByLabelText("Latest release year"), { target: { value: "2010" } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ yearMax: "2010" }));
  });

  it("shows the chosen minimum score next to the slider", () => {
    const { onChange } = setup(cleanValue({ minScore: 3.5 }));

    expect(screen.getByText("3.5★ and up")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("slider", { name: "Minimum predicted rating" }), {
      target: { value: "4.5" },
    });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ minScore: 4.5 }));
  });

  it("labels an untouched slider as any score", () => {
    setup();

    expect(screen.getByText("Any score")).toBeInTheDocument();
  });

  it("offers reset only once a filter is set", async () => {
    const user = userEvent.setup();
    const { onReset, rerender, props } = setup();
    expect(screen.queryByRole("button", { name: "Reset" })).not.toBeInTheDocument();

    rerender(<RecommendationFilters {...props} value={cleanValue({ genres: ["Action"] })} />);
    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(onReset).toHaveBeenCalled();
  });

  it("collapses the controls but keeps a summary of the active filters", async () => {
    const user = userEvent.setup();
    setup(cleanValue({ genres: ["Drama"], minScore: 4 }));

    expect(screen.getByRole("button", { name: "Drama" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^Filters/ }));

    expect(screen.queryByRole("button", { name: "Drama" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Filters/ })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(screen.getByText("Drama · 4.0★+")).toBeInTheDocument();
  });

  it("reports a validation error under the controls", () => {
    setup(cleanValue(), { error: "The first year must come before the second." });

    expect(screen.getByText("The first year must come before the second.")).toHaveClass(
      "filter-error",
    );
  });

  it("explains when the genre list could not be loaded", () => {
    setup(cleanValue(), { genres: [] });

    expect(screen.getByText(/genre list unavailable/i)).toBeInTheDocument();
  });
});

describe("toRequestFilters", () => {
  it("turns blank years and a zero score into absent filters", () => {
    expect(toRequestFilters(cleanValue())).toEqual({
      genres: [],
      yearMin: null,
      yearMax: null,
      minScore: null,
      error: null,
    });
  });

  it("keeps genres and numeric bounds for the API", () => {
    const request = toRequestFilters(
      cleanValue({ genres: ["Sci-Fi", "Drama"], yearMin: "1990", yearMax: "2010", minScore: 4 }),
    );

    expect(request).toEqual({
      genres: ["Sci-Fi", "Drama"],
      yearMin: 1990,
      yearMax: 2010,
      minScore: 4,
      error: null,
    });
  });

  it("rejects a reversed range and non-numeric years", () => {
    expect(toRequestFilters(cleanValue({ yearMin: "2000", yearMax: "1990" })).error).toMatch(
      /before/i,
    );
    expect(toRequestFilters(cleanValue({ yearMin: "19x0" })).error).toMatch(/whole numbers/i);
  });

  it("counts any populated control as filtered", () => {
    expect(isFiltered(cleanValue())).toBe(false);
    expect(isFiltered(cleanValue({ genres: ["Action"] }))).toBe(true);
    expect(isFiltered(cleanValue({ yearMin: "1999" }))).toBe(true);
    expect(isFiltered(cleanValue({ minScore: 0.5 }))).toBe(true);
  });
});
