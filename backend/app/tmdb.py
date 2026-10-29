"""TMDB enrichment: posters and overviews, fetched lazily and cached locally.

Per PLAN.md §4.5 the catalog's ``tmdbId`` (from ``links.csv``) is the key. Per §5.13 the API key
lives on the server, responses are cached, and TMDB problems never break the app: if the key is
missing, the network is down, or the quota is exhausted, every function here returns ``None`` and
the caller renders without a poster.

Attribution is required by TMDB's terms and is shown in the UI footer; a deployment must also add
the TMDB logo (see README).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.app.config import get_settings

API_BASE_URL = "https://api.themoviedb.org/3/movie/"
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."
# TMDB's documented poster sizes; w342 covers a card, w780 a detail view.
POSTER_SIZES = ("w92", "w154", "w185", "w342", "w500", "w780", "original")

# TMDB genre names mapped onto the MovieLens labels the catalog already uses. Names that have no
# MovieLens equivalent (TV Movie, History, ...) are dropped rather than invented: a wrong genre
# would feed the content model the wrong taste.
GENRE_MAP = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "comedy": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "family": "Children",
    "fantasy": "Fantasy",
    "history": "Drama",
    "horror": "Horror",
    "music": "Musical",
    "mystery": "Mystery",
    "romance": "Romance",
    "science fiction": "Sci-Fi",
    "tv movie": "Drama",
    "thriller": "Thriller",
    "war": "War",
    "western": "Western",
}


# TMDB's stable genre ids, used by /search which returns ids rather than names.
_GENRE_IDS_TO_NAMES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


def map_genres(names: list[str] | tuple[str, ...]) -> list[str]:
    """MovieLens genre labels for TMDB genre names, de-duplicated and in a stable order."""
    mapped: list[str] = []
    for name in names:
        label = GENRE_MAP.get(str(name).strip().lower())
        if label and label not in mapped:
            mapped.append(label)
    return mapped


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def year_from_release_date(value: str | None) -> int | None:
    """``2019-05-17`` -> ``2019``; anything unparseable is simply unknown."""
    if not value or len(value) < 4:
        return None
    try:
        year = int(value[:4])
    except ValueError:
        return None
    return year if 1880 <= year <= 2100 else None


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


@dataclass(frozen=True)
class TmdbSearchResult:
    """One search hit, before anything is written to the catalog."""

    tmdb_id: int
    title: str
    release_year: int | None
    overview: str | None
    poster_path: str | None
    genres: tuple[str, ...]
    vote_count: int
    popularity: float


@dataclass(frozen=True)
class TmdbDetails:
    """Everything ingestion needs to create a catalog row."""

    tmdb_id: int
    title: str
    release_year: int | None
    overview: str | None
    poster_path: str | None
    genres: tuple[str, ...]
    keywords: tuple[str, ...]
    imdb_id: int | None
    vote_count: int


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
        search_cache_minutes: int = 1440,
    ) -> None:
        self.api_key = api_key.strip()
        self.read_token = read_token.strip()
        self.cache_dir = Path(cache_dir or get_settings().tmdb_cache_dir)
        self.timeout = timeout
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self._transport = transport
        self.search_cache_minutes = max(0, int(search_cache_minutes))
        self._memory: dict[int, dict[str, str | None] | None] = {}
        self._search_memory: dict[str, tuple[float, list[TmdbSearchResult]]] = {}
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
        status, payload = self._request(tmdb_id)
        if status == 404:
            # The id does not exist; remembering that costs one file and saves a request forever.
            self._write_cache(tmdb_id, None)
            return None
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

    def _get_json(
        self, url: str, params: dict[str, str], *, attempt: int = 0
    ) -> tuple[int | None, dict | None]:
        """One throttled GET with a single retry on 429, returning ``(status, body)``.

        The status is what separates "TMDB says this does not exist" (a fact worth caching) from
        "TMDB could not answer" (a blip that must be retried next time), so it is never discarded.
        ``None`` means the request never reached a server.
        """
        try:
            self._throttle()
            with httpx.Client(
                transport=self._transport,
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = client.get(url, headers=self._headers(), params=params)
        except httpx.HTTPError:
            return None, None

        if response.status_code == 429 and attempt == 0:
            delay = self._retry_after(response)
            if delay is not None:
                time.sleep(delay)
                return self._get_json(url, params, attempt=1)
            return response.status_code, None
        if response.status_code >= 400:
            return response.status_code, None
        try:
            body = response.json()
        except ValueError:
            return response.status_code, None
        return response.status_code, body if isinstance(body, dict) else None

    def _request(self, tmdb_id: int) -> tuple[int | None, dict[str, str | None] | None]:
        status, body = self._get_json(f"{API_BASE_URL}{tmdb_id}", self._params(tmdb_id))
        if body is None:
            return status, None
        overview = body.get("overview")
        cleaned = overview.strip() if isinstance(overview, str) else ""
        return status, {
            "poster_path": body.get("poster_path") or None,
            "overview": cleaned or None,
        }

    # ---------------------------------------------------------------- search (5.10)

    def search(self, query: str, *, limit: int = 8) -> list[TmdbSearchResult] | None:
        """Catalog candidates TMDB knows about, newest cache win.

        ``None`` means "could not ask" (disabled, offline, rate-limited); an empty list means TMDB
        answered and knew nothing, which the caller is allowed to cache and show as "no matches".
        """
        cleaned = query.strip()
        if not self.enabled or not cleaned or limit < 1:
            return None

        key = f"{cleaned.lower()}|{int(limit)}"
        cached = self._read_search_cache(key)
        if cached is not None:
            return cached

        params = self._params(0)
        params.update({"query": cleaned, "include_adult": "false", "page": "1"})
        _, body = self._get_json(SEARCH_URL, params)
        if body is None:
            return None

        results = [
            result
            for result in (self._to_search_result(raw) for raw in body.get("results", []))
            if result is not None
        ][:limit]
        self._write_search_cache(key, results)
        return results

    @staticmethod
    def _to_search_result(raw: object) -> TmdbSearchResult | None:
        if not isinstance(raw, dict):
            return None
        try:
            tmdb_id = int(raw.get("id") or 0)
        except (TypeError, ValueError):
            return None
        title = str(raw.get("title") or raw.get("original_title") or "").strip()
        if tmdb_id <= 0 or not title:
            return None
        overview = raw.get("overview")
        genre_ids = raw.get("genre_ids")
        return TmdbSearchResult(
            tmdb_id=tmdb_id,
            title=title,
            release_year=year_from_release_date(raw.get("release_date")),
            overview=overview.strip() if isinstance(overview, str) and overview.strip() else None,
            poster_path=raw.get("poster_path") or None,
            genres=tuple(
                map_genres(
                    [
                        _GENRE_IDS_TO_NAMES[gid]
                        for gid in (_as_int(entry) for entry in genre_ids)
                        if gid is not None
                    ]
                )
            )
            if isinstance(genre_ids, list)
            else (),
            vote_count=int(raw.get("vote_count") or 0),
            popularity=float(raw.get("popularity") or 0.0),
        )

    def _search_cache_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / "search" / f"{digest}.json"

    def _read_search_cache(self, key: str) -> list[TmdbSearchResult] | None:
        entry = self._search_memory.get(key)
        now = time.time()
        if entry is not None:
            stored_at, results = entry
            if self.search_cache_minutes <= 0 or now - stored_at < self.search_cache_minutes * 60:
                return list(results)
            self._search_memory.pop(key, None)

        path = self._search_cache_path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stored_at = float(payload.get("stored_at", 0.0))
            raw_results = payload.get("results", [])
        except (OSError, ValueError, AttributeError):
            return None
        if self.search_cache_minutes <= 0 or now - stored_at >= self.search_cache_minutes * 60:
            return None
        results = [
            result
            for result in (self._to_search_result(raw) for raw in raw_results)
            if result is not None
        ]
        self._search_memory[key] = (stored_at, results)
        return list(results)

    def _write_search_cache(self, key: str, results: list[TmdbSearchResult]) -> None:
        self._search_memory[key] = (time.time(), results)
        payload = {
            "stored_at": time.time(),
            "results": [
                {
                    "id": result.tmdb_id,
                    "title": result.title,
                    "release_date": (
                        f"{result.release_year}-01-01" if result.release_year else None
                    ),
                    "overview": result.overview,
                    "poster_path": result.poster_path,
                    "genre_ids": [],
                    "vote_count": result.vote_count,
                    "popularity": result.popularity,
                }
                for result in results
            ],
        }
        try:
            self._search_cache_path(key).parent.mkdir(parents=True, exist_ok=True)
            self._search_cache_path(key).write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

    # ---------------------------------------------------------------- details (5.10)

    def details(self, tmdb_id: int) -> TmdbDetails | None:
        """Genres, keywords, release date and ids for one film, cached like a poster fetch."""
        if not self.enabled or tmdb_id <= 0:
            return None
        path = self.cache_dir / "details" / f"{int(tmdb_id)}.json"
        payload: dict | None = None
        if path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                payload = stored if isinstance(stored, dict) else None
            except (OSError, ValueError):
                payload = None
        if payload is None:
            params = self._params(tmdb_id)
            params["append_to_response"] = "keywords"
            status, payload = self._get_json(f"{API_BASE_URL}{tmdb_id}", params)
            if status == 404:
                # Same rule as a poster fetch: a film TMDB does not know stays unknown.
                payload = {"missing": True}
            elif payload is None:
                return None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass
        if payload.get("missing"):
            return None
        return self._to_details(tmdb_id, payload)

    @staticmethod
    def _to_details(tmdb_id: int, payload: dict) -> TmdbDetails | None:
        title = str(payload.get("title") or payload.get("original_title") or "").strip()
        if not title:
            return None
        genres = payload.get("genres")
        genre_names = [
            str(entry.get("name", ""))
            for entry in (genres if isinstance(genres, list) else [])
            if isinstance(entry, dict)
        ]
        keywords_block = payload.get("keywords")
        keyword_entries = (
            keywords_block.get("keywords", []) if isinstance(keywords_block, dict) else []
        )
        keywords = tuple(
            str(entry.get("name", "")).strip().lower()
            for entry in (keyword_entries if isinstance(keyword_entries, list) else [])
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        )
        overview = payload.get("overview")
        imdb_raw = str(payload.get("imdb_id") or "")
        imdb_id = (
            int(imdb_raw[2:])
            if imdb_raw.startswith("tt") and imdb_raw[2:].isdigit()
            else None
        )
        return TmdbDetails(
            tmdb_id=int(tmdb_id),
            title=title,
            release_year=year_from_release_date(payload.get("release_date")),
            overview=overview.strip() if isinstance(overview, str) and overview.strip() else None,
            poster_path=payload.get("poster_path") or None,
            genres=tuple(map_genres(genre_names)),
            keywords=keywords,
            imdb_id=imdb_id,
            vote_count=int(payload.get("vote_count") or 0),
        )

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
