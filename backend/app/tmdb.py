"""TMDB enrichment: posters and overviews, fetched lazily and cached locally.

Per PLAN.md §4.5 the catalog's ``tmdbId`` (from ``links.csv``) is the key. Per §5.13 the API key
lives on the server, responses are cached, and TMDB problems never break the app: if the key is
missing, the network is down, or the quota is exhausted, every function here returns ``None`` and
the caller renders without a poster.

Attribution is required by TMDB's terms and is shown in the UI footer; a deployment must also add
the TMDB logo (see README).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.app.config import get_settings

API_BASE_URL = "https://api.themoviedb.org/3/movie/"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."
# TMDB's documented poster sizes; w342 covers a card, w780 a detail view.
POSTER_SIZES = ("w92", "w154", "w185", "w342", "w500", "w780", "original")


def poster_url(poster_path: str | None, *, size: str | None = None) -> str | None:
    """Absolute image URL for a stored poster path (TMDB returns paths, not URLs)."""
    if not poster_path:
        return None
    if poster_path.startswith(("http://", "https://")):
        return poster_path
    chosen = size or get_settings().tmdb_poster_size
    if chosen not in POSTER_SIZES:
        chosen = "w342"
    return f"{IMAGE_BASE_URL}{chosen}{poster_path}"


@dataclass(frozen=True)
class TmdbMovie:
    tmdb_id: int
    poster_path: str | None
    overview: str | None


class TmdbClient:
    """One endpoint, an on-disk cache, a polite request rate and no exceptions to the caller."""

    def __init__(
        self,
        *,
        api_key: str = "",
        read_token: str = "",
        cache_dir: str | Path | None = None,
        timeout: float = 8.0,
        max_calls_per_second: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.read_token = read_token.strip()
        self.cache_dir = Path(cache_dir or get_settings().tmdb_cache_dir)
        self.timeout = timeout
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self._transport = transport
        self._memory: dict[int, dict[str, str | None] | None] = {}
        self._last_call = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.read_token)

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.read_token:
            headers["Authorization"] = f"Bearer {self.read_token}"
        return headers

    def _params(self, tmdb_id: int) -> dict[str, str]:
        params = {"language": "en-US"}
        if not self.read_token and self.api_key:
            params["api_key"] = self.api_key
        return params

    def _cache_path(self, tmdb_id: int) -> Path:
        return self.cache_dir / f"{tmdb_id}.json"

    def _read_cache(self, tmdb_id: int) -> tuple[bool, dict[str, str | None] | None]:
        """Return ``(was_cached, payload)``; a cached miss is remembered too."""
        if tmdb_id in self._memory:
            return True, self._memory[tmdb_id]
        path = self._cache_path(tmdb_id)
        if not path.is_file():
            return False, None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, None
        if not isinstance(payload, dict):
            return False, None
        entry = None if payload.get("missing") else payload
        self._memory[tmdb_id] = entry
        return True, entry

    def _write_cache(self, tmdb_id: int, entry: dict[str, str | None] | None) -> None:
        self._memory[tmdb_id] = entry
        payload = entry if entry is not None else {"missing": True}
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(tmdb_id).write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            # A read-only or missing cache directory must not break a request.
            pass

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def fetch(self, tmdb_id: int) -> TmdbMovie | None:
        """Poster path and overview for a TMDB id, or ``None`` if unavailable."""
        if not self.enabled or tmdb_id <= 0:
            return None
        cached, entry = self._read_cache(tmdb_id)
        if cached:
            return self._to_movie(tmdb_id, entry)
        payload = self._request(tmdb_id)
        if payload is None:
            return None  # transient failure: do not poison the cache
        self._write_cache(tmdb_id, payload)
        return self._to_movie(tmdb_id, payload)

    def _to_movie(self, tmdb_id: int, entry: dict[str, str | None] | None) -> TmdbMovie | None:
        if entry is None:
            return None
        return TmdbMovie(
            tmdb_id=tmdb_id,
            poster_path=entry.get("poster_path"),
            overview=entry.get("overview"),
        )

    def _request(self, tmdb_id: int, *, attempt: int = 0) -> dict[str, str | None] | None:
        url = f"{API_BASE_URL}{tmdb_id}"
        try:
            self._throttle()
            with httpx.Client(
                transport=self._transport,
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = client.get(
                    url, headers=self._headers(), params=self._params(tmdb_id)
                )
        except httpx.HTTPError:
            return None

        if response.status_code == 429 and attempt == 0:
            delay = self._retry_after(response)
            if delay is not None:
                time.sleep(delay)
                return self._request(tmdb_id, attempt=1)
            return None
        if response.status_code == 404:
            self._write_cache(tmdb_id, None)  # TMDB has no such id: remember that
            return None
        if response.status_code >= 400:
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        if not isinstance(body, dict):
            return None
        overview = body.get("overview")
        cleaned = overview.strip() if isinstance(overview, str) else ""
        return {
            "poster_path": body.get("poster_path") or None,
            "overview": cleaned or None,
        }

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After", "")
        try:
            delay = float(raw)
        except ValueError:
            return None
        return delay if 0 <= delay <= 5 else None


def build_tmdb_client() -> TmdbClient:
    settings = get_settings()
    return TmdbClient(
        api_key=settings.tmdb_api_key,
        read_token=settings.tmdb_read_access_token,
        cache_dir=settings.tmdb_cache_dir,
    )
