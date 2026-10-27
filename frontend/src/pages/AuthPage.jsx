import { useState } from "react";

import * as api from "../api.js";

export default function AuthPage({ initialMode = "login", message, onAuthenticated, onCancel }) {
  const [mode, setMode] = useState(initialMode);
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const isSignup = mode === "signup";

  function switchMode(next) {
    setMode(next);
    setError(null);
  }

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = isSignup
        ? await api.signup({ email, displayName, password })
        : await api.login({ email, password });
      api.storeToken(result.access_token);
      onAuthenticated(result.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="auth-wrap">
      <div className="auth-card">
        <div className="auth-tabs" aria-label="Account access">
          <button
            type="button"
            aria-pressed={!isSignup}
            className={!isSignup ? "is-active" : ""}
            onClick={() => switchMode("login")}
          >
            Sign in
          </button>
          <button
            type="button"
            aria-pressed={isSignup}
            className={isSignup ? "is-active" : ""}
            onClick={() => switchMode("signup")}
          >
            Create account
          </button>
        </div>

        <div className="auth-body">
          <p className="eyebrow">{isSignup ? "New here" : "Welcome back"}</p>
          <h1>{isSignup ? "Start your watchlist." : "Pick up where you left off."}</h1>
          {message ? <p className="auth-message">{message}</p> : null}

          <form onSubmit={submit} className="form">
            {isSignup ? (
              <label className="field">
                <span>Display name</span>
                <input
                  type="text"
                  autoComplete="name"
                  value={displayName}
                  onChange={(event) => setDisplayName(event.target.value)}
                  required
                  maxLength={80}
                  placeholder="Ada"
                />
              </label>
            ) : null}

            <label className="field">
              <span>Email</span>
              <input
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
                placeholder="you@example.com"
              />
            </label>

            <label className="field">
              <span>Password</span>
              <input
                type="password"
                autoComplete={isSignup ? "new-password" : "current-password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
                minLength={isSignup ? 8 : undefined}
                maxLength={128}
                placeholder={isSignup ? "At least 8 characters" : "Your password"}
              />
            </label>

            {error ? (
              <p className="form-error" role="alert">
                {error}
              </p>
            ) : null}

            <button type="submit" className="button submit" disabled={busy}>
              {busy ? "Working…" : isSignup ? "Create account" : "Sign in"}
            </button>
          </form>

          {onCancel ? (
            <button type="button" className="link-button back" onClick={onCancel}>
              ← Back to browsing
            </button>
          ) : null}
        </div>
      </div>

      <aside className="auth-aside" aria-label="About accounts">
        <p className="eyebrow">Why sign in</p>
        <ul>
          <li>Rate films in half-star steps and keep your list.</li>
          <li>Get picks computed from your ratings, not the crowd&apos;s.</li>
          <li>See the reason behind every recommendation.</li>
        </ul>
      </aside>
    </section>
  );
}
