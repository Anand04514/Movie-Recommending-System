/**
 * Light/dark theme state.
 *
 * The preference is explicit once the user chooses one, and follows the operating system until
 * then. `document.documentElement[data-theme]` drives the CSS; `index.html` sets it before the
 * stylesheet loads so a dark-preferring reader never sees a flash of the light palette.
 */

export const THEME_KEY = "movie-recs.theme";

export function systemTheme() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "dark";
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function storedTheme() {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    // Private browsing can block storage; the session still gets a sensible theme.
    return null;
  }
}

export function resolveTheme() {
  return storedTheme() ?? systemTheme();
}

export function applyTheme(theme) {
  if (typeof document === "undefined") return;
  document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
}

export function saveTheme(theme) {
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Ignore storage failures; the choice lasts for this session.
  }
}
