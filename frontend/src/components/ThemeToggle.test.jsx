import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import ThemeToggle from "./ThemeToggle.jsx";
import { THEME_KEY } from "../theme.js";

function clearTheme() {
  delete document.documentElement.dataset.theme;
  window.localStorage.clear();
}

afterEach(() => {
  clearTheme();
  vi.unstubAllGlobals();
});

describe("ThemeToggle", () => {
  it("starts dark when nothing is stored and the system expresses no preference", () => {
    render(<ThemeToggle />);

    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(screen.getByRole("button", { name: "Switch to light mode" })).toBeInTheDocument();
  });

  it("follows a system preference for light on the first visit", () => {
    vi.stubGlobal("matchMedia", (query) => ({
      matches: query.includes("light"),
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    }));

    render(<ThemeToggle />);

    expect(document.documentElement.dataset.theme).toBe("light");
    expect(screen.getByRole("button", { name: "Switch to dark mode" })).toBeInTheDocument();
  });

  it("flips the palette and remembers the choice", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);

    await user.click(screen.getByRole("button", { name: "Switch to light mode" }));

    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem(THEME_KEY)).toBe("light");
    expect(screen.getByRole("button", { name: "Switch to dark mode" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Switch to dark mode" }));

    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem(THEME_KEY)).toBe("dark");
  });

  it("restores a stored preference over the system one", () => {
    window.localStorage.setItem(THEME_KEY, "light");

    render(<ThemeToggle />);

    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("ignores a corrupted stored value instead of rendering a broken theme", () => {
    window.localStorage.setItem(THEME_KEY, "purple");

    render(<ThemeToggle />);

    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});
