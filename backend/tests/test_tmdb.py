"""TMDB enrichment: cache behaviour, failure tolerance and the poster URL helper."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.deps import get_tmdb_client
from backend.app.main import app
from backend.app.tmdb import IMAGE_BASE_URL, TmdbClient, poster_url


def make_client(tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]) -> TmdbClient:
    return TmdbClient(
        api_key="test-key",
        cache_dir=tmp_path / "tmdb_cache",
        max_calls_per_second=0,  # no sleeping in tests
        transport=httpx.MockTransport(handler),
    )


def json_handler(payload: dict, *, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def test_poster_url_is_built_on_the_image_cdn(monkeypatch: pytest.MonkeyPatch) -> None:
    assert poster_url("/abc.jpg") == f"{IMAGE_BASE_URL}w342/abc.jpg"
    assert poster_url("/abc.jpg", size="w780") == f"{IMAGE_BASE_URL}w780/abc.jpg"
    assert poster_url("/abc.jpg", size="not-a-size") == f"{IMAGE_BASE_URL}w342/abc.jpg"
    assert poster_url("https://example.com/x.jpg") == "https://example.com/x.jpg"
    assert poster_url(None) is None
    assert poster_url("") is None


def test_client_is_disabled_without_credentials(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"poster_path": "/p.jpg"})

    disabled = TmdbClient(
        api_key="",
        read_token="",
        cache_dir=tmp_path / "cache",
        transport=httpx.MockTransport(handler),
    )
    assert disabled.enabled is False
    assert disabled.fetch(603) is None
    assert calls == []  # no key means no HTTP call at all
    assert TmdbClient(api_key="k").min_interval == pytest.approx(1 / 20)
    assert TmdbClient(api_key="k", max_calls_per_second=0).min_interval == 0


def test_fetch_caches_the_result_and_reuses_it(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"poster_path": "/poster.jpg", "overview": "  A chess prodigy grows up.  "},
        )

    first = make_client(tmp_path, handler)
    movie = first.fetch(603)
    assert movie is not None
    assert movie.poster_path == "/poster.jpg"
    assert movie.overview == "A chess prodigy grows up."  # trimmed
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/movie/603")
    assert calls[0].url.params["api_key"] == "test-key"

    # A second client (fresh process) finds the cache on disk and makes no request.
    second = make_client(tmp_path, handler)
    assert second.fetch(603).poster_path == "/poster.jpg"
    assert len(calls) == 1


def test_a_read_access_token_is_sent_as_a_bearer_header(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"poster_path": "/p.jpg"})

    client = TmdbClient(
        api_key="ignored-when-a-token-exists",
        read_token="token",
        cache_dir=tmp_path / "cache",
        max_calls_per_second=0,
        transport=httpx.MockTransport(handler),
    )
    client.fetch(603)
    assert seen[0].headers["authorization"] == "Bearer token"
    assert "api_key" not in seen[0].url.params


def test_a_missing_movie_is_remembered_so_it_is_not_asked_again(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(404, json={"status_message": "Not found"})

    client = make_client(tmp_path, handler)
    assert client.fetch(999999) is None
    assert client.fetch(999999) is None
    assert len(calls) == 1  # the 404 is a fact about the id, so it is cached
    assert (tmp_path / "tmdb_cache" / "999999.json").is_file()


def test_transport_failures_are_not_cached(tmp_path: Path) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise httpx.ConnectError("network down", request=request)
        return httpx.Response(500)

    client = make_client(tmp_path, handler)
    assert client.fetch(603) is None
    assert not (tmp_path / "tmdb_cache" / "603.json").exists()  # a blip must not poison the cache
    assert client.fetch(603) is None  # the server error is still tried again
    assert len(attempts) == 2


def test_rate_limited_requests_retry_once_after_the_documented_delay(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"poster_path": "/p.jpg"})

    client = make_client(tmp_path, handler)
    assert client.fetch(603).poster_path == "/p.jpg"
    assert len(calls) == 2


def test_a_long_retry_after_is_not_waited_out(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(429, headers={"Retry-After": "120"})

    client = make_client(tmp_path, handler)
    assert client.fetch(603) is None  # never sleeps for two minutes inside a request
    assert len(calls) == 1


def test_a_corrupt_cache_entry_is_refetched(tmp_path: Path) -> None:
    cache_dir = tmp_path / "tmdb_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "603.json").write_text("{not json", encoding="utf-8")

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(200, json={"poster_path": "/p.jpg"})

    assert make_client(tmp_path, handler).fetch(603).poster_path == "/p.jpg"
    assert len(calls) == 1


def test_non_object_payloads_and_blank_fields_are_treated_as_absent(tmp_path: Path) -> None:
    client = make_client(tmp_path, json_handler({"poster_path": "", "overview": "   "}))
    movie = client.fetch(603)
    assert movie is not None
    assert movie.poster_path is None
    assert movie.overview is None


@pytest.fixture
def tmdb(tmp_path: Path) -> Iterator[Callable[[Callable], list[httpx.Request]]]:
    """Override the app's TMDB dependency with a mock-transport client, counting requests."""
    calls: list[httpx.Request] = []

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list[httpx.Request]:
        client = TmdbClient(
            api_key="test-key",
            cache_dir=tmp_path / "cache",
            max_calls_per_second=0,
            transport=httpx.MockTransport(
                lambda request: (calls.append(request), handler(request))[1]
            ),
        )
        app.dependency_overrides[get_tmdb_client] = lambda: client
        return calls

    yield install
    app.dependency_overrides.pop(get_tmdb_client, None)


def test_movie_detail_fills_a_poster_once_and_stores_it(
    client: TestClient, add_movies, tmdb
) -> None:
    calls = tmdb(json_handler({"poster_path": "/poster.jpg", "overview": "A chess prodigy."}))
    (movie_id,) = add_movies(
        {"id": 238, "title": "The Godfather (1972)", "genres": "Crime", "tmdb_id": 238}
    )

    body = client.get(f"/api/movies/{movie_id}").json()
    assert body["poster_path"] == "/poster.jpg"
    assert body["overview"] == "A chess prodigy."
    assert len(calls) == 1

    # Stored on the row, so the next request is served from the database with no TMDB call.
    again = client.get(f"/api/movies/{movie_id}").json()
    assert again["poster_path"] == "/poster.jpg"
    assert len(calls) == 1


def test_movie_detail_is_unaffected_when_tmdb_is_unavailable(
    client: TestClient, add_movies, tmdb
) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    calls = tmdb(failing)
    (movie_id,) = add_movies(
        {"id": 680, "title": "Pulp Fiction (1994)", "tmdb_id": 680}
    )

    response = client.get(f"/api/movies/{movie_id}")
    assert response.status_code == 200  # TMDB trouble never breaks the catalog
    body = response.json()
    assert body["poster_path"] is None
    assert len(calls) == 1
    assert client.get(f"/api/movies/{movie_id}").status_code == 200
    assert len(calls) == 2  # a failure is not cached, so it is retried later


def test_movies_without_a_tmdb_id_never_call_tmdb(client: TestClient, add_movies, tmdb) -> None:
    calls = tmdb(json_handler({"poster_path": "/poster.jpg"}))
    (movie_id,) = add_movies({"id": 1, "title": "Toy Story (1995)"})

    assert client.get(f"/api/movies/{movie_id}").json()["poster_path"] is None
    assert calls == []


def test_stored_posters_are_never_refetched(client: TestClient, add_movies, tmdb) -> None:
    calls = tmdb(json_handler({"poster_path": "/new.jpg"}))
    (movie_id,) = add_movies(
        {
            "id": 1,
            "title": "Toy Story (1995)",
            "tmdb_id": 862,
            "poster_path": "/stored.jpg",
        }
    )

    body = client.get(f"/api/movies/{movie_id}").json()
    assert body["poster_path"] == "/stored.jpg"  # the stored value wins
    assert calls == []
