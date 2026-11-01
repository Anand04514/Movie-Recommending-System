import { useEffect, useState } from "react";

import { applyTheme, resolveTheme, saveTheme } from "../theme.js";

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" aria-hidden="true" focusable="false">
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 2.4v2.2M12 19.4v2.2M4.2 12H2M22 12h-2.2M5.8 5.8 4.2 4.2M19.8 19.8l-1.6-1.6M18.2 5.8l1.6-1.6M4.2 19.8l1.6-1.6" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" aria-hidden="true" focusable="false">
      <path d="M20.5 14.3a8.6 8.6 0 0 1-10.8-10.8 8.7 8.7 0 1 0 10.8 10.8Z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(resolveTheme);

  // Applied on mount as well as on change: the inline script in index.html handles the first paint,
  // and this keeps the attribute correct for anything rendered outside that page (tests, embeds).
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  function toggle() {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    saveTheme(next);
  }

  const target = theme === "light" ? "dark" : "light";
  const label = `Switch to ${target} mode`;

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      aria-label={label}
      title={label}
    >
      {theme === "light" ? <MoonIcon /> : <SunIcon />}
    </button>
  );
}
