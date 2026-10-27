import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import StarRating from "./StarRating.jsx";

const OPTIONS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

function Harness({ initial = 0 }) {
  const [value, setValue] = useState(initial);
  return <StarRating value={value} onChange={setValue} label="Rate Dune" />;
}

function fillWidths(container) {
  return Array.from(container.querySelectorAll(".star-fill")).map((element) => element.style.width);
}

describe("StarRating", () => {
  it("offers ten half-star options in an accessible radio group", () => {
    render(<StarRating value={0} onChange={() => {}} label="Rate Dune" />);

    expect(screen.getByRole("radiogroup", { name: "Rate Dune" })).toBeInTheDocument();
    expect(screen.getAllByRole("radio")).toHaveLength(OPTIONS.length);
    for (const option of OPTIONS) {
      expect(screen.getByRole("radio", { name: `${option} out of 5 stars` })).toBeEnabled();
    }
    expect(screen.getByRole("radio", { name: "0.5 out of 5 stars" })).not.toBeChecked();
    expect(screen.getByText("Rate")).toBeInTheDocument();
  });

  it("checks the matching radio and shows the value once set", () => {
    render(<StarRating value={2.5} onChange={() => {}} label="Rate Dune" />);

    expect(screen.getByRole("radio", { name: "2.5 out of 5 stars" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "2 out of 5 stars" })).not.toBeChecked();
    expect(screen.getByText("2.5")).toBeInTheDocument();
  });

  it("reports the clicked half star and updates the widget", async () => {
    const user = userEvent.setup();
    render(<Harness initial={4} />);

    await user.click(screen.getByRole("radio", { name: "4.5 out of 5 stars" }));

    expect(screen.getByRole("radio", { name: "4.5 out of 5 stars" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "4 out of 5 stars" })).not.toBeChecked();
    expect(screen.getByText("4.5")).toBeInTheDocument();
  });

  it("fills stars in half-star steps", () => {
    const { container } = render(<StarRating value={2.5} onChange={() => {}} />);

    expect(fillWidths(container)).toEqual(["100%", "100%", "50%", "0%", "0%"]);
  });

  it("previews a hovered half star without committing the value", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container } = render(<StarRating value={2} onChange={onChange} />);

    await user.hover(screen.getByRole("radio", { name: "4.5 out of 5 stars" }));

    expect(fillWidths(container)).toEqual(["100%", "100%", "100%", "100%", "50%"]);
    expect(screen.getByText("2.0")).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("ignores interaction when disabled", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<StarRating value={3} onChange={onChange} disabled />);

    expect(screen.getAllByRole("radio").every((radio) => radio.disabled)).toBe(true);
    await user.click(screen.getByRole("radio", { name: "5 out of 5 stars" }));
    expect(onChange).not.toHaveBeenCalled();
  });
});
