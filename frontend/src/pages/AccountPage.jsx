import { useState } from "react";

import * as api from "../api.js";

const CONFIRMATION_WORD = "DELETE";

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
}

export default function AccountPage({ user, onDeleted }) {
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const armed = confirmation.trim().toUpperCase() === CONFIRMATION_WORD && password.length > 0;

  async function remove(event) {
    event.preventDefault();
    if (!armed) {
      setError(`Type ${CONFIRMATION_WORD} and your password to confirm.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.deleteAccount(password);
      onDeleted();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Account</p>
        <h1>Your account.</h1>
        <p className="lede">
          Signed in as {user.email}. Member since {formatDate(user.created_at)}.
        </p>
      </section>

      <section className="account-grid">
        <div className="panel">
          <h3>What this account holds</h3>
          <p>
            Your ratings, the films you liked or rejected, the picks you were shown, and any
            precomputed candidates for your profile. The movie catalog itself belongs to the
            dataset and is not yours to delete.
          </p>
          <p className="auth-note">
            Forgot the password? Signing out and choosing <em>Forgot your password?</em> sends a
            single-use reset link; the same reset can be started by whoever runs this instance if
            email is not configured.
          </p>
        </div>

        <div className="panel danger-zone">
          <h3>Delete this account</h3>
          <p>
            This removes the account and everything listed beside it, immediately and permanently.
            There is no undo, no grace period and no export — if you want your ratings, copy them
            from <strong>My ratings</strong> first.
          </p>

          <form onSubmit={remove} className="form">
            <label className="field">
              <span>Password</span>
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Confirm it is you"
                maxLength={128}
              />
            </label>

            <label className="field">
              <span>
                Type <strong>{CONFIRMATION_WORD}</strong> to confirm
              </span>
              <input
                type="text"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                placeholder={CONFIRMATION_WORD}
                autoComplete="off"
                maxLength={12}
              />
            </label>

            {error ? (
              <p className="form-error" role="alert">
                {error}
              </p>
            ) : null}

            <button type="submit" className="button danger" disabled={busy || !armed}>
              {busy ? "Deleting…" : "Delete my account"}
            </button>
          </form>
        </div>
      </section>
    </>
  );
}
