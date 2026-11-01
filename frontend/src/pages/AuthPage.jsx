import { useState } from "react";

import * as api from "../api.js";

export default function AuthPage({
  initialMode = "login",
  message,
  onAuthenticated,
  onCancel,
}) {
  const [mode, setMode] = useState(initialMode);
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [resetSent, setResetSent] = useState(false);

  const isSignup = mode === "signup";
  const isForgot = mode === "forgot";

  function switchMode(next) {
    setMode(next);
    setError(null);
    setResetSent(false);
  }

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (isForgot) {
        // The server answers the same way whether or not the address exists, so the UI must not
        // imply anything either: it confirms the request, never the account.
        await api.requestPasswordReset(email);
        setResetSent(true);
        return;
      }
      const result = isSignup
        ? await api.signup({ email, displayName, password })
        : await api.login({ email, password });
      api.storeToken(result.access_token);
      onAuthenticated(result.user, { isNew: isSignup });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="auth-wrap">
      <div className="auth-card">
        {!isForgot ? (
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
        ) : null}

        <div className="auth-body">
          <p className="eyebrow">
            {isForgot ? "Password reset" : isSignup ? "New here" : "Welcome back"}
          </p>
          <h1>
            {isForgot
              ? "Get back in."
              : isSignup
                ? "Start your watchlist."
                : "Pick up where you left off."}
          </h1>
          {message && !isForgot ? <p className="auth-message">{message}</p> : null}

          {isForgot && resetSent ? (
            <div className="auth-sent" role="status">
              <p>
                If that address has an account, a reset link is on its way. The link works once
                and expires in about half an hour.
              </p>
              <p className="auth-note">
                Nothing arrived? This instance may not have email configured — ask whoever runs it
                for a reset link.
              </p>
              <button type="button" className="button" onClick={() => switchMode("login")}>
                Back to sign in
              </button>
            </div>
          ) : (
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

              {!isForgot ? (
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
              ) : (
                <p className="auth-note">
                  We will email a single-use link to choose a new password. Your current password
                  keeps working until you use it.
                </p>
              )}

              {error ? (
                <p className="form-error" role="alert">
                  {error}
                </p>
              ) : null}

              <button type="submit" className="button submit" disabled={busy}>
                {busy ? "Working…" : isForgot ? "Send reset link" : isSignup ? "Create account" : "Sign in"}
              </button>

              {!isSignup && !isForgot ? (
                <button
                  type="button"
                  className="link-button forgot"
                  onClick={() => switchMode("forgot")}
                >
                  Forgot your password?
                </button>
              ) : null}

              {isForgot ? (
                <button
                  type="button"
                  className="link-button forgot"
                  onClick={() => switchMode("login")}
                >
                  Back to sign in
                </button>
              ) : null}
            </form>
          )}

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
