import { useState } from "react";

import * as api from "../api.js";

export default function ResetPasswordPage({ token, onReset, onNeedNewLink }) {
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    // Checked here as well as on the server: a typo in the second field is the common mistake and
    // waiting for a round trip to say so is worse feedback.
    if (password.length < 8) {
      setError("Use at least 8 characters.");
      return;
    }
    if (password !== confirmation) {
      setError("The two passwords do not match.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.confirmPasswordReset(token, password);
      api.storeToken(result.access_token);
      onReset(result.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (!token) {
    return (
      <section className="auth-wrap">
        <div className="auth-card">
          <div className="auth-body">
            <p className="eyebrow">Password reset</p>
            <h1>This link is incomplete.</h1>
            <p className="auth-note">
              The address had no reset token in it, which usually means a mail client cut the link
              in half. Ask for a new one and open it in one click.
            </p>
            <button type="button" className="button" onClick={onNeedNewLink}>
              Request a new link
            </button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="auth-wrap">
      <div className="auth-card">
        <div className="auth-body">
          <p className="eyebrow">Password reset</p>
          <h1>Choose a new password.</h1>
          <p className="auth-note">
            Signing in will need this password from now on, and every other session on this account
            is signed out.
          </p>

          <form onSubmit={submit} className="form">
            <label className="field">
              <span>New password</span>
              <input
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
                minLength={8}
                maxLength={128}
                placeholder="At least 8 characters"
              />
            </label>

            <label className="field">
              <span>Repeat it</span>
              <input
                type="password"
                autoComplete="new-password"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                required
                maxLength={128}
                placeholder="Type it again"
              />
            </label>

            {error ? (
              <p className="form-error" role="alert">
                {error}
              </p>
            ) : null}

            <button type="submit" className="button submit" disabled={busy}>
              {busy ? "Saving…" : "Save password and sign in"}
            </button>

            <button type="button" className="link-button forgot" onClick={onNeedNewLink}>
              This link did not work — send another
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}
