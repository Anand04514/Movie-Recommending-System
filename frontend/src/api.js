const TOKEN_KEY = "movie-recs.token";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getStoredToken() {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token) {
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Private browsing can block storage; the in-memory session still works.
  }
}

export function clearToken() {
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Ignore storage failures.
  }
}

function formatDetail(detail, status) {
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter((message) => typeof message === "string" && message.length > 0);
    if (messages.length > 0) return messages.join(". ");
  }
  if (typeof detail === "string" && detail.length > 0) return detail;
  return `Something went wrong (HTTP ${status}).`;
}

async function request(path, { method = "GET", body, auth = false, signal } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const token = auth ? getStoredToken() : null;
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("Cannot reach the API. Is the backend running?", 0);
  }

  if (response.status === 204) return null;
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(formatDetail(payload?.detail, response.status), response.status);
  }
  return payload;
}

export function searchMovies(query, { limit = 18, includeExternal = true, signal } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (query.trim()) params.set("q", query.trim());
  if (!includeExternal) params.set("include_external", "false");
  return request(`/movies/search?${params.toString()}`, { signal });
}

export function getMovie(movieId, { signal } = {}) {
  return request(`/movies/${movieId}`, { signal });
}

// Adds a film TMDB knows to the catalog, once, and returns it as a normal movie.
export function addTmdbMovie(tmdbId) {
  return request(`/movies/tmdb/${tmdbId}`);
}

export function getNewReleases({ limit = 6, signal } = {}) {
  return request(`/recommend/new-releases?limit=${limit}`, { auth: true, signal });
}

export function signup({ email, displayName, password }) {
  return request("/auth/signup", {
    method: "POST",
    body: { email, display_name: displayName, password },
  });
}

export function login({ email, password }) {
  return request("/auth/login", { method: "POST", body: { email, password } });
}

export function getMe() {
  return request("/auth/me", { auth: true });
}

// The answer is deliberately identical for a registered address, an unknown one and a failed send.
export function requestPasswordReset(email) {
  return request("/auth/password-reset", { method: "POST", body: { email } });
}

export function confirmPasswordReset(token, password) {
  return request("/auth/password-reset/confirm", {
    method: "POST",
    body: { token, password },
  });
}

export function deleteAccount(password) {
  return request("/auth/delete-account", {
    method: "POST",
    body: { password },
    auth: true,
  });
}

export function listRatings() {
  return request("/ratings", { auth: true });
}

export function saveRating(movieId, rating) {
  return request("/ratings", {
    method: "POST",
    body: { movie_id: movieId, rating },
    auth: true,
  });
}

export function deleteRating(movieId) {
  return request(`/ratings/${movieId}`, { method: "DELETE", auth: true });
}

export function getRecommendations({
  limit = 12,
  algorithm = null,
  genres = [],
  yearMin = null,
  yearMax = null,
  minScore = null,
  signal,
} = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (algorithm) params.set("algorithm", algorithm);
  if (genres.length > 0) params.set("genres", genres.join(","));
  if (yearMin != null) params.set("year_min", String(yearMin));
  if (yearMax != null) params.set("year_max", String(yearMax));
  if (minScore != null) params.set("min_score", String(minScore));
  return request(`/recommend?${params.toString()}`, { auth: true, signal });
}

export function listGenres({ signal } = {}) {
  return request("/movies/genres", { signal });
}

export function compareAlgorithms({ limit = 10, left = null, right = null, signal } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (left) params.set("left", left);
  if (right) params.set("right", right);
  return request(`/compare?${params.toString()}`, { auth: true, signal });
}

export function getOnboarding({ limit = 12, signal } = {}) {
  return request(`/onboarding?limit=${limit}`, { signal });
}

export function sendFeedback(movieId, action, sourceAlgorithm = null) {
  return request("/feedback", {
    method: "POST",
    body: {
      movie_id: movieId,
      action,
      source_algorithm: sourceAlgorithm,
    },
    auth: true,
  });
}

export function listModelRuns({ signal } = {}) {
  return request("/model-runs", { signal });
}

export function listExperiments({ signal } = {}) {
  return request("/experiments", { signal });
}

export function getNewReleaseReport({ signal } = {}) {
  return request("/new-releases/report", { signal });
}
