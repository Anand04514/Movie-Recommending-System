"""Batch poster prefetch: planning, writing, and the command line."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from backend.app.db import get_session_factory
from backend.app.models import Movie
from backend.app.tmdb import TmdbClient, TmdbMovie
from scripts.fetch_posters import main, plan_fetch, run_fetch, store_fetched


def make_client(tmp_path: Path, payloads: dict[int, dict]) -> tuple[TmdbClient, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tmdb_id = int(request.url.path.rstrip("/").rsplit("/", 1)[-1])
        calls.append(tmdb_id)
        payload = payloads.get(tmdb_id)
        if payload is None:
            return httpx.Response(404, json={"status_message": "Not found"})
        return httpx.Response(200, json=payload)

    client = TmdbClient(
        api_key="test-key",
        cache_dir=tmp_path / "cache",
        max_calls_per_second=0,
        transport=httpx.MockTransport(handler),
    )
    return client, calls


def test_plan_prefers_popular_movies_and_skips_complete_rows(
    add_movies: Callable[..., list[int]],
) -> None:
    add_movies(
        {"id": 1, "title": "Popular (1999)", "tmdb_id": 11, "rating_count": 900},
        {
            "id": 2,
            "title": "Complete (1998)",
            "tmdb_id": 12,
            "rating_count": 800,
            "poster_path": "/p.jpg",
            "overview": "Already enriched.",
        },
        {
            "id": 3,
            "title": "Needs Overview (1997)",
            "tmdb_id": 13,
            "rating_count": 700,
            "poster_path": "/p.jpg",
        },
        {"id": 4, "title": "No Tmdb (1996)", "rating_count": 600},
        {"id": 5, "title": "Obscure (1995)", "tmdb_id": 15, "rating_count": 2},
        {"id": 6, "title": "Newest (1994)", "tmdb_id": 16, "rating_count": 100},
    )

    with get_session_factory()() as session:
        assert plan_fetch(session, limit=10) == [(1, 11), (3, 13), (6, 16), (5, 15)]
        assert plan_fetch(session, limit=2) == [(1, 11), (3, 13)]
        assert plan_fetch(session, limit=10, min_rating_count=50) == [(1, 11), (3, 13), (6, 16)]
        with pytest.raises(ValueError):
            plan_fetch(session, limit=0)


def test_store_fetched_writes_only_what_came_back(add_movies: Callable[..., list[int]]) -> None:
    (movie_id,) = add_movies({"id": 7, "title": "Untouched (2001)", "tmdb_id": 70})

    with get_session_factory()() as session:
        assert store_fetched(session, movie_id, None) is False
        assert store_fetched(session, movie_id, TmdbMovie(70, None, None)) is False

        assert store_fetched(session, movie_id, TmdbMovie(70, "/poster.jpg", None)) is True
        movie = session.get(Movie, movie_id)
        assert movie.poster_path == "/poster.jpg"
        assert movie.overview is None  # never invented

        assert store_fetched(session, movie_id, TmdbMovie(70, "/poster.jpg", "A plot.")) is True
        assert session.get(Movie, movie_id).overview == "A plot."

        # Same data again changes nothing, so a re-run reports honestly.
        assert store_fetched(session, movie_id, TmdbMovie(70, "/poster.jpg", "A plot.")) is False
        assert store_fetched(session, 999_999, TmdbMovie(1, "/x.jpg", None)) is False
        session.commit()


def test_run_fetch_counts_unavailable_movies_and_keeps_going(
    tmp_path: Path, add_movies: Callable[..., list[int]]
) -> None:
    add_movies(
        {"id": 10, "title": "One (2000)", "tmdb_id": 101, "rating_count": 5},
        {"id": 11, "title": "Two (2001)", "tmdb_id": 102, "rating_count": 4},
        {"id": 12, "title": "Three (2002)", "tmdb_id": 103, "rating_count": 3},
    )
    client, calls = make_client(
        tmp_path,
        {
            101: {"poster_path": "/one.jpg", "overview": "First."},
            102: {},  # TMDB knows the id but has nothing to offer
        },
    )
    seen: list[tuple[int, int, int]] = []

    with get_session_factory()() as session:
        pairs = plan_fetch(session, limit=10)
        assert [movie_id for movie_id, _ in pairs] == [10, 11, 12]
        counters = run_fetch(
            session, client, pairs, batch_size=1, progress=lambda *args: seen.append(args)
        )

    assert counters == {"planned": 3, "stored": 1, "missing": 2}
    assert calls == [101, 102, 103]
    assert seen == [(1, 1, 0), (2, 1, 1), (3, 1, 2)]

    with get_session_factory()() as session:
        assert session.get(Movie, 10).poster_path == "/one.jpg"
        assert session.get(Movie, 10).overview == "First."
        assert session.get(Movie, 11).poster_path is None
        assert session.get(Movie, 12).overview is None


def test_a_rerun_uses_the_cache_and_the_database(
    tmp_path: Path, add_movies: Callable[..., list[int]]
) -> None:
    (movie_id,) = add_movies({"id": 20, "title": "Cached (2003)", "tmdb_id": 201})
    payloads = {201: {"poster_path": "/cached.jpg", "overview": "Cached plot."}}

    first_client, first_calls = make_client(tmp_path, payloads)
    with get_session_factory()() as session:
        run_fetch(session, first_client, [(movie_id, 201)])
    assert first_calls == [201]

    # A fresh process: the row is complete, so nothing is even planned.
    with get_session_factory()() as session:
        assert plan_fetch(session, limit=10) == []

    # And if the row were cleared, the on-disk cache supplies the data with no HTTP call.
    with get_session_factory()() as session:
        session.get(Movie, movie_id).poster_path = None
        session.commit()
    second_client, second_calls = make_client(tmp_path, payloads)
    with get_session_factory()() as session:
        counters = run_fetch(session, second_client, [(movie_id, 201)])
    assert counters["stored"] == 1
    assert second_calls == []


def test_dry_run_needs_no_credentials(
    add_movies: Callable[..., list[int]], capsys: pytest.CaptureFixture[str]
) -> None:
    add_movies({"id": 30, "title": "Planned (2004)", "tmdb_id": 301, "rating_count": 3})

    assert main(["--dry-run"], client_factory=lambda: pytest.fail("must not build a client")) == 0
    output = capsys.readouterr().out
    assert "1 movies are missing a poster" in output
    assert "would fetch movie 30 (tmdb 301)" in output


def test_main_reports_missing_credentials_and_does_not_crash(
    add_movies: Callable[..., list[int]], capsys: pytest.CaptureFixture[str]
) -> None:
    add_movies({"id": 31, "title": "Unconfigured (2005)", "tmdb_id": 311})

    disabled = TmdbClient(api_key="", read_token="")
    assert main([], client_factory=lambda: disabled) == 1
    assert "TMDB is not configured" in capsys.readouterr().err


def test_main_stores_posters_for_the_planned_movies(
    tmp_path: Path,
    add_movies: Callable[..., list[int]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    add_movies(
        {"id": 40, "title": "Stored (2006)", "tmdb_id": 401, "rating_count": 9},
        {"id": 41, "title": "Gone (2007)", "tmdb_id": 402, "rating_count": 8},
    )
    client, _ = make_client(tmp_path, {401: {"poster_path": "/stored.jpg", "overview": "Plot."}})

    assert main(["--batch-size", "1"], client_factory=lambda: client) == 0
    output = capsys.readouterr().out
    assert "2 movies are missing a poster" in output
    assert "1 movies updated, 1 unavailable" in output

    with get_session_factory()() as session:
        assert session.get(Movie, 40).poster_path == "/stored.jpg"
        assert session.get(Movie, 41).poster_path is None


def test_main_rejects_a_nonsense_batch_size() -> None:
    assert main(["--batch-size", "0"]) == 2
