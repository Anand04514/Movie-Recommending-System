import { useCallback, useEffect, useMemo, useState } from "react";

import * as api from "./api.js";
import AccuracyPage from "./pages/AccuracyPage.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import ComparePage from "./pages/ComparePage.jsx";
import RatingsPage from "./pages/RatingsPage.jsx";
import RecommendationsPage from "./pages/RecommendationsPage.jsx";
import SearchPage from "./pages/SearchPage.jsx";

const NAV = [
  { id: "discover", label: "Discover" },
  { id: "for-you", label: "For you" },
  { id: "compare", label: "Compare" },
  { id: "ratings", label: "My ratings" },
  { id: "accuracy", label: "Accuracy" },
];
// Only the pages built on your own ratings require an account.
const PRIVATE_VIEWS = new Set(["for-you", "ratings", "compare"]);

export default function App() {
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(() => Boolean(api.getStoredToken()));
  const [view, setView] = useState("discover");
  const [pendingView, setPendingView] = useState("for-you");
  const [authMessage, setAuthMessage] = useState("");
  const [ratings, setRatings] = useState([]);
  const [ratingsStatus, setRatingsStatus] = useState({ loading: false, error: null });
  const [ratingsReloadKey, setRatingsReloadKey] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!api.getStoredToken()) {
      setCheckingSession(false);
      return undefined;
    }
    let cancelled = false;
    api
      .getMe()
      .then((profile) => {
        if (cancelled) return;
        setUser(profile);
        setCheckingSession(false);
      })
      .catch((error) => {
        if (cancelled) return;
        // Only a token the server actually rejected is worth discarding. A network blip must not
        // sign the user out for good, so the credential stays and the next load retries.
        if (error instanceof api.ApiError && (error.status === 401 || error.status === 403)) {
          api.clearToken();
        }
        setCheckingSession(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!user) {
      setRatings([]);
      setRatingsStatus({ loading: false, error: null });
      return undefined;
    }
    let cancelled = false;
    setRatingsStatus({ loading: true, error: null });
    api
      .listRatings()
      .then((rows) => {
        if (cancelled) return;
        setRatings(rows);
        setRatingsStatus({ loading: false, error: null });
      })
      .catch((error) => {
        if (cancelled) return;
        setRatings([]);
        setRatingsStatus({ loading: false, error: error.message });
      });
    return () => {
      cancelled = true;
    };
  }, [user, ratingsReloadKey]);

  const ratingsByMovie = useMemo(
    () => new Map(ratings.map((row) => [row.movie.id, row.rating])),
    [ratings],
  );

  const openAuth = useCallback(
    (message = "Sign in to keep going.", nextView) => {
      setAuthMessage(message);
      setPendingView(nextView ?? (view === "auth" ? "for-you" : view));
      setView("auth");
    },
    [view],
  );

  const navigate = useCallback(
    (id) => {
      if (PRIVATE_VIEWS.has(id) && !user) {
        const message =
          id === "for-you"
            ? "Sign in to see picks made for you."
            : id === "ratings"
              ? "Sign in to see your ratings."
              : "Sign in to compare algorithms made for you.";
        openAuth(message, id);
        return;
      }
      setView(id);
    },
    [user, openAuth],
  );

  const handleRate = useCallback(
    async (movie, value) => {
      if (!user) {
        openAuth("Sign in to save your ratings.");
        return;
      }
      const saved = await api.saveRating(movie.id, value);
      setRatings((previous) => [
        saved,
        ...previous.filter((row) => row.movie.id !== movie.id),
      ]);
      setRefreshKey((key) => key + 1);
    },
    [user, openAuth],
  );

  const handleRemove = useCallback(async (movie) => {
    await api.deleteRating(movie.id);
    setRatings((previous) => previous.filter((row) => row.movie.id !== movie.id));
    setRefreshKey((key) => key + 1);
  }, []);

  function handleAuthenticated(profile) {
    setUser(profile);
    setAuthMessage("");
    const next = pendingView ?? "for-you";
    setPendingView("for-you");
    setView(next);
  }

  function signOut() {
    api.clearToken();
    setUser(null);
    setRatings([]);
    setView("discover");
  }

  const explore = () => setView("discover");

  return (
    <div className="shell">
      <header className="topbar">
        <button
          className="brand"
          type="button"
          onClick={() => setView("discover")}
          aria-label="Movie Recs home"
        >
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          movie recs
        </button>

        <nav className="nav" aria-label="Primary">
          {NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-link ${view === item.id ? "is-active" : ""}`}
              aria-current={view === item.id ? "page" : undefined}
              onClick={() => navigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="account">
          {user ? (
            <>
              <button
                type="button"
                className="account-name"
                onClick={() => setView("ratings")}
                title="Your ratings"
              >
                {user.display_name}
              </button>
              <button type="button" className="link-button" onClick={signOut}>
                Sign out
              </button>
            </>
          ) : (
            <button
              type="button"
              className="button small"
              onClick={() => openAuth("Sign in to rate films and get picks.", "for-you")}
            >
              Sign in
            </button>
          )}
        </div>
      </header>

      <main>
        {checkingSession ? <p className="page-loading">Restoring your session…</p> : null}

        {!checkingSession && view === "discover" ? (
          <SearchPage
            user={user}
            ratingsByMovie={ratingsByMovie}
            onRate={handleRate}
            onRequireAuth={openAuth}
          />
        ) : null}

        {!checkingSession && view === "for-you" ? (
          <RecommendationsPage
            user={user}
            refreshKey={refreshKey}
            ratingsByMovie={ratingsByMovie}
            onRate={handleRate}
            onRequireAuth={openAuth}
            onExplore={explore}
          />
        ) : null}

        {!checkingSession && view === "compare" ? (
          <ComparePage
            user={user}
            onRequireAuth={openAuth}
            onExplore={explore}
          />
        ) : null}

        {!checkingSession && view === "accuracy" ? <AccuracyPage /> : null}

        {!checkingSession && view === "ratings" ? (
          <RatingsPage
            user={user}
            ratings={ratings}
            loading={ratingsStatus.loading}
            error={ratingsStatus.error}
            onRate={handleRate}
            onRemove={handleRemove}
            onRequireAuth={openAuth}
            onExplore={explore}
            onRetry={() => setRatingsReloadKey((key) => key + 1)}
          />
        ) : null}

        {!checkingSession && view === "auth" ? (
          <AuthPage
            message={authMessage}
            onAuthenticated={handleAuthenticated}
            onCancel={() => setView("discover")}
          />
        ) : null}
      </main>

      <footer className="site-footer">
        <span>Movie Recs — a MovieLens ml-32m project.</span>
        <span>
          Ratings and titles from GroupLens Research. This product uses the TMDB API but is not
          endorsed or certified by TMDB.
        </span>
      </footer>
    </div>
  );
}
