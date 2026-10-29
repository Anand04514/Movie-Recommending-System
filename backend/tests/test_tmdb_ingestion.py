"""Phase 5.10–5.12: TMDB search, on-demand ingestion, content cold start and recency.

No test here touches the network: every TMDB request goes through an ``httpx.MockTransport``, so a
missing key, a rate limit or an outage is exercised as a normal case rather than as a happy path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse
from sqlalchemy import select

from backend.app.config import Settings
from backend.app.db import get_session_factory
from backend.app.deps import get_tmdb_client
from backend.app.freshness import age_in_days, boost_for, is_new_release, release_date_for
from backend.app.ingest import TMDB_ID_BASE, ingest, movie_id_for
from backend.app.main import app
from backend.app.models import Movie, Rating
from backend.app.recommendations import (
    ContentService,
    HybridService,
    RecommendationBatch,
    RecommendationService,
    ScoredItem,
    SvdService,
    UserCfService,
    get_content_recommender,
    get_hybrid_recommender,
    get_recommender,
    get_svd_recommender,
    get_user_cf_recommender,
)
from backend.app.routers import recommendations as rec_router
from backend.app.tmdb import TmdbClient
from recsys.content import build_content_features, feature_row, load_features, save_features
from recsys.item_cf import build_item_neighbors, save_neighbors
from recsys.svd import save_model, train_svd
from recsys.user_cf import build_user_neighbors, save_user_neighbors

# A fixture film released inside the recency window. Derived from today's date so the suite does
# not quietly start failing two years from now.
RECENT_YEAR = date.today().year - 1
RECENT_DATE = f"{RECENT_YEAR}-03-14"
TODAY = date(2026, 10, 29)

MOVIE_IDS = [10, 11, 12, 13, 14, 15, 16, 17, 18]
# Two taste groups, plus id 18: a recent release inside the model universe, for the recency tests.
MATRIX = np.array(
    [
        [5.0, 4.0, 1.0, 1.0, 4.0, 3.0, 1.0, 2.0, 3.5],
        [4.0, 5.0, 2.0, 1.0, 4.5, 3.5, 1.0, 2.0, 4.0],
        [5.0, 4.0, 1.0, 2.0, 3.5, 3.0, 1.0, 2.5, 3.0],
        [1.0, 2.0, 5.0, 4.0, 2.0, 2.5, 5.0, 4.0, 2.5],
        [1.0, 1.0, 4.0, 5.0, 1.5, 2.0, 4.5, 5.0, 2.0],
        [2.0, 1.0, 5.0, 4.0, 2.5, 2.0, 5.0, 4.0, 3.0],
    ],
    dtype="float32",
)
MOVIES: list[dict[str, Any]] = [
    {"id": 10, "title": "Alpha (2000)", "genres": "Action|Thriller", "rating_count": 6},
    {"id": 11, "title": "Beta (2001)", "genres": "Action|Sci-Fi", "rating_count": 6},
    {"id": 12, "title": "Gamma (2002)", "genres": "Drama", "rating_count": 6},
    {"id": 13, "title": "Delta (2003)", "genres": "Drama|Romance", "rating_count": 6},
    {"id": 14, "title": "Epsilon (1990)", "genres": "Action|Adventure", "rating_count": 6},
    {"id": 15, "title": "Zeta (1995)", "genres": "Comedy", "rating_count": 6},
    {"id": 16, "title": "Eta (2010)", "genres": "Horror", "rating_count": 6},
    {"id": 17, "title": "Theta (2012)", "genres": "Sci-Fi", "rating_count": 6},
    {"id": 18, "title": f"Omega ({RECENT_YEAR})", "genres": "Action|Sci-Fi", "rating_count": 6},
]
RATED = ((10, 5.0), (11, 4.0), (12, 1.0), (13, 1.0))

# A TMDB film that matches this profile through Sci-Fi/Action, released inside the window.
TMDB_ID = 424242
DETAILS_PAYLOAD = {
    "id": TMDB_ID,
    "title": "Synthetic Future",
    "release_date": RECENT_DATE,
    "overview": "A film newer than the dataset.",
    "poster_path": "/future.jpg",
    "imdb_id": "tt4242420",
    "vote_count": 1200,
    "genres": [{"id": 878, "name": "Science Fiction"}, {"id": 28, "name": "Action"}],
    "keywords": {"keywords": [{"id": 1, "name": "Dystopia"}, {"id": 2, "name": "Space"}]},
}
SEARCH_PAYLOAD = {
    "page": 1,
    "results": [
        {
            "id": TMDB_ID,
            "title": "Synthetic Future",
            "release_date": RECENT_DATE,
            "overview": "A film newer than the dataset.",
            "poster_path": "/future.jpg",
            "genre_ids": [878, 28],
            "vote_count": 1200,
            "popularity": 42.0,
        },
        {
            # Already in the catalog (a MovieLens film), so it must not be offered as external.
            "id": 603,
            "title": "The Matrix",
            "release_date": "1999-03-31",
            "overview": "Neo.",
            "poster_path": "/matrix.jpg",
            "genre_ids": [28, 878],
            "vote_count": 24000,
            "popularity": 90.0,
        },
    ],
}


def write_artifacts(directory: Path) -> None:
    matrix = sparse.csr_matrix(MATRIX)
    save_neighbors(
        build_item_neighbors(matrix, k=4, block_size=5), directory / "item_neighbors.npz"
    )
    save_user_neighbors(
        build_user_neighbors(matrix, k=3, block_size=3), directory / "user_neighbors.npz"
    )
    sparse.save_npz(directory / "train_matrix.npz", matrix)
    pd.DataFrame({"movieId": MOVIE_IDS, "movie_index": list(range(len(MOVIE_IDS)))}).to_parquet(
        directory / "movie_id_map.parquet", index=False
    )
    save_model(train_svd(matrix, rank=3), directory / "svd_model.npz")
    catalog = pd.DataFrame(
        [
            {
                "movie_id": spec["id"],
                "genres": spec["genres"],
                "release_year": int(spec["title"][-5:-1]),
            }
            for spec in MOVIES
        ]
    )
    save_features(build_content_features(catalog), directory / "content_features.npz")


@pytest.fixture
def services(tmp_path: Path) -> Iterator[dict[str, Any]]:
    write_artifacts(tmp_path)
    item = RecommendationService(tmp_path)
    user = UserCfService(tmp_path)
    content = ContentService(tmp_path, fallback=item)
    svd = SvdService(tmp_path, fallback=item)
    hybrid = HybridService(item, user, content, processed_dir=tmp_path, base="user_cf", weight=0.8)
    overrides = {
        get_recommender: item,
        get_user_cf_recommender: user,
        get_content_recommender: content,
        get_svd_recommender: svd,
        get_hybrid_recommender: hybrid,
    }

    def provider(service: Any):
        def _service():
            return service

        return _service

    for dependency, service in overrides.items():
        app.dependency_overrides[dependency] = provider(service)
    try:
        yield {"item": item, "user": user, "content": content, "hybrid": hybrid}
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)


@pytest.fixture
def customer(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], add_movies
) -> tuple[dict[str, str], dict]:
    add_movies(*MOVIES)
    headers, profile = signup()
    for movie_id, rating in RATED:
        response = client.post(
            "/api/ratings", json={"movie_id": movie_id, "rating": rating}, headers=headers
        )
        assert response.status_code == 200, response.text
    return headers, profile


def tmdb_transport(
    *, details_status: int = 200, search_status: int = 200, calls: list[str] | None = None
) -> httpx.MockTransport:
    """A transport that routes by path, so search and details are exercised together."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path.endswith("/search/movie"):
            return httpx.Response(search_status, json=SEARCH_PAYLOAD)
        if request.url.path.endswith(f"/movie/{TMDB_ID}"):
            return httpx.Response(details_status, json=DETAILS_PAYLOAD)
        return httpx.Response(404, json={"status_message": "Not found"})

    return httpx.MockTransport(handler)


def build_client(tmp_path: Path, transport: httpx.MockTransport) -> TmdbClient:
    return TmdbClient(
        api_key="test-key",
        cache_dir=tmp_path / "tmdb_cache",
        max_calls_per_second=0,
        search_cache_minutes=60,
        transport=transport,
    )


def with_tmdb(monkeypatch: pytest.MonkeyPatch, client: TmdbClient) -> None:
    """Serve this client wherever the app asks for one; monkeypatch removes it afterwards."""
    monkeypatch.setitem(app.dependency_overrides, get_tmdb_client, lambda: client)


# ------------------------------------------------------------------ client: search and details


def test_search_maps_genres_and_caches_the_answer(tmp_path: Path) -> None:
    calls: list[str] = []
    client = build_client(tmp_path, tmdb_transport(calls=calls))

    results = client.search("synthetic", limit=5)
    assert results is not None
    first = results[0]
    assert first.tmdb_id == TMDB_ID
    assert first.title == "Synthetic Future"
    assert first.release_year == RECENT_YEAR
    assert first.genres == ("Sci-Fi", "Action")  # TMDB ids mapped onto MovieLens labels
    assert first.poster_path == "/future.jpg"

    again = client.search("synthetic", limit=5)
    assert [r.tmdb_id for r in again] == [r.tmdb_id for r in results]
    assert len(calls) == 1  # cache, not a second request


def test_search_failures_are_not_cached(tmp_path: Path) -> None:
    calls: list[str] = []
    client = build_client(tmp_path, tmdb_transport(search_status=503, calls=calls))

    assert client.search("anything") is None  # "could not ask", not "nothing found"
    assert client.search("anything") is None
    assert len(calls) == 2  # a server error is retried rather than remembered


def test_search_is_unavailable_without_credentials(tmp_path: Path) -> None:
    calls: list[str] = []
    disabled = TmdbClient(
        api_key="",
        read_token="",
        cache_dir=tmp_path / "cache",
        transport=tmdb_transport(calls=calls),
    )
    assert disabled.search("synthetic") is None
    assert calls == []


def test_details_map_keywords_year_and_imdb_id(tmp_path: Path) -> None:
    client = build_client(tmp_path, tmdb_transport())
    details = client.details(TMDB_ID)

    assert details is not None
    assert details.genres == ("Sci-Fi", "Action")
    assert details.keywords == ("dystopia", "space")
    assert details.release_year == RECENT_YEAR
    assert details.imdb_id == 4242420
    assert details.poster_path == "/future.jpg"


def test_details_remembers_a_film_tmdb_does_not_have(tmp_path: Path) -> None:
    calls: list[str] = []
    client = build_client(tmp_path, tmdb_transport(details_status=404, calls=calls))

    assert client.details(TMDB_ID) is None
    assert client.details(TMDB_ID) is None
    assert len(calls) == 1  # a 404 is a fact about the id, not an outage


def test_details_failure_returns_none_and_is_not_cached(tmp_path: Path) -> None:
    calls: list[str] = []
    client = build_client(tmp_path, tmdb_transport(details_status=500, calls=calls))

    assert client.details(TMDB_ID) is None
    assert client.details(TMDB_ID) is None
    assert len(calls) == 2


# ------------------------------------------------------------------ ingestion


def test_ingestion_reserves_an_id_range_and_stores_what_tmdb_knows(tmp_path: Path) -> None:
    tmdb = build_client(tmp_path, tmdb_transport())
    with get_session_factory()() as session:
        movie = ingest(session, tmdb, TMDB_ID)
        assert movie is not None

        assert movie.id == TMDB_ID_BASE + TMDB_ID == movie_id_for(TMDB_ID)
        assert movie.source == "tmdb"
        assert movie.in_model is False  # no collaborative model has seen it
        assert movie.rating_count == 0  # TMDB votes are not MovieLens ratings
        assert movie.release_year == RECENT_YEAR
        assert movie.genres == "Sci-Fi|Action"
        assert movie.tmdb_keywords == "dystopia,space"
        assert movie.imdb_id == 4242420
        assert movie.title_search == "synthetic future"


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    calls: list[str] = []
    tmdb = build_client(tmp_path, tmdb_transport(calls=calls))
    with get_session_factory()() as session:
        first = ingest(session, tmdb, TMDB_ID)
        second = ingest(session, tmdb, TMDB_ID)

        assert first is not None and second is not None
        assert first.id == second.id
        assert len(session.scalars(select(Movie)).all()) == 1
    assert calls == [f"/3/movie/{TMDB_ID}"]  # the second call never hit the network


def test_ingestion_reuses_a_movielens_film_with_the_same_tmdb_id(
    add_movies, tmp_path: Path
) -> None:
    add_movies({"id": 603, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi", "tmdb_id": 603})
    calls: list[str] = []
    tmdb = build_client(tmp_path, tmdb_transport(calls=calls))
    with get_session_factory()() as session:
        movie = ingest(session, tmdb, 603)
        assert movie is not None
        assert movie.id == 603  # the existing row, not 10_000_603
        assert len(session.scalars(select(Movie)).all()) == 1
    assert calls == []  # the link already existed, so TMDB was never asked


def test_ingestion_matches_an_existing_film_by_imdb_id(tmp_path: Path) -> None:
    with get_session_factory()() as session:
        session.add(
            Movie(
                id=777,
                title_display="Synthetic Future",
                title_search="synthetic future",
                release_year=RECENT_YEAR,
                genres="Sci-Fi",
                imdb_id=4242420,
                tmdb_id=None,
            )
        )
        session.commit()

    tmdb = build_client(tmp_path, tmdb_transport())
    with get_session_factory()() as session:
        movie = ingest(session, tmdb, TMDB_ID)
        assert movie is not None
        assert movie.id == 777
        assert movie.tmdb_id == TMDB_ID  # the link is filled in rather than a duplicate created
        assert len(session.scalars(select(Movie)).all()) == 1


def test_ingestion_gives_up_quietly_when_tmdb_fails(tmp_path: Path) -> None:
    tmdb = build_client(tmp_path, tmdb_transport(details_status=500))
    with get_session_factory()() as session:
        assert ingest(session, tmdb, TMDB_ID) is None
        assert session.scalars(select(Movie)).all() == []


# ------------------------------------------------------------------ search API


def test_search_offers_tmdb_films_the_catalog_lacks(
    client: TestClient, add_movies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_movies({"id": 603, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi", "tmdb_id": 603})
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport()))

    body = client.get("/api/movies/search", params={"q": "synthetic"}).json()

    assert body["count"] == 0
    assert body["external_source"] == "tmdb"
    assert [item["tmdb_id"] for item in body["external"]] == [TMDB_ID]
    external = body["external"][0]
    assert external["title"] == "Synthetic Future"
    assert external["genres"] == ["Sci-Fi", "Action"]
    assert external["poster_url"].startswith("https://image.tmdb.org/")


def test_search_marks_a_known_tmdb_film_as_local_not_external(
    client: TestClient, add_movies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The catalog already holds TMDB id 603, so the hit for The Matrix must be suppressed.
    add_movies({"id": 603, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi", "tmdb_id": 603})
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport()))

    body = client.get("/api/movies/search", params={"q": "matrix"}).json()
    assert body["count"] == 1
    assert [movie["id"] for movie in body["results"]] == [603]
    # TMDB's own hit for the same film must not be offered as a second, addable copy.
    assert 603 not in {hit["tmdb_id"] for hit in body["external"]}


def test_search_skips_tmdb_when_the_catalog_is_already_useful(
    client: TestClient, add_movies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    add_movies(
        *[
            {"id": 100 + index, "title": f"Star Wars {index} (1999)", "genres": "Action"}
            for index in range(6)
        ]
    )
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport(calls=calls)))

    body = client.get("/api/movies/search", params={"q": "star wars"}).json()

    assert body["count"] == 6
    assert body["external"] == []
    assert calls == []  # five or more local hits means TMDB is not asked at all


def test_search_survives_a_tmdb_outage(
    client: TestClient, add_movies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_movies({"id": 603, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi", "tmdb_id": 603})
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport(search_status=500)))

    response = client.get("/api/movies/search", params={"q": "matrix"})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["external"] == []


def test_search_with_tmdb_disabled_still_works(
    client: TestClient, add_movies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_movies({"id": 603, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi", "tmdb_id": 603})
    disabled = TmdbClient(api_key="", read_token="", cache_dir=tmp_path / "cache")
    with_tmdb(monkeypatch, disabled)

    body = client.get("/api/movies/search", params={"q": "matrix"}).json()
    assert body["count"] == 1
    assert body["external"] == []
    assert body["external_source"] is None


# ------------------------------------------------------------------ ingestion endpoint


def test_opening_a_tmdb_film_adds_it_once(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport(calls=calls)))

    first = client.get(f"/api/movies/tmdb/{TMDB_ID}")
    second = client.get(f"/api/movies/tmdb/{TMDB_ID}")

    assert first.status_code == 200
    body = first.json()
    assert body["id"] == TMDB_ID_BASE + TMDB_ID
    assert body["source"] == "tmdb"
    assert body["title"] == "Synthetic Future"
    assert body["release_year"] == RECENT_YEAR
    assert second.json()["id"] == body["id"]
    assert len(calls) == 1
    with get_session_factory()() as session:
        assert len(session.scalars(select(Movie)).all()) == 1


def test_ingest_endpoint_reports_a_missing_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_tmdb(monkeypatch, TmdbClient(api_key="", read_token=""))

    response = client.get(f"/api/movies/tmdb/{TMDB_ID}")

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_ingest_endpoint_404s_when_tmdb_does_not_know_the_film(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport(details_status=404)))

    response = client.get(f"/api/movies/tmdb/{TMDB_ID}")

    assert response.status_code == 404
    with get_session_factory()() as session:
        assert session.scalars(select(Movie)).all() == []


def test_an_ingested_film_can_be_rated_like_any_other(
    client: TestClient, signup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport()))
    movie = client.get(f"/api/movies/tmdb/{TMDB_ID}").json()

    headers, _ = signup()
    saved = client.post(
        "/api/ratings",
        json={"movie_id": movie["id"], "rating": 4.5},
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json()["movie"]["source"] == "tmdb"


# ------------------------------------------------------------------ cold start (5.11)


@pytest.fixture
def ingested(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    with_tmdb(monkeypatch, build_client(tmp_path, tmdb_transport()))
    return client.get(f"/api/movies/tmdb/{TMDB_ID}").json()


def test_new_releases_are_scored_by_content_only(
    client: TestClient, customer, services, ingested
) -> None:
    headers, _ = customer
    picks = client.get("/api/recommend/new-releases", headers=headers).json()

    assert picks, "a recent Sci-Fi film must be recommendable to a Sci-Fi-heavy profile"
    pick = next(item for item in picks if item["movie"]["id"] == ingested["id"])
    assert pick["movie"]["source"] == "tmdb"
    assert pick["reason"].startswith("New release matching your taste")
    assert pick["new_release"] is True
    assert 0.5 <= pick["predicted_rating"] <= 5.0


def test_new_releases_exclude_what_the_user_already_rated(
    client: TestClient, customer, services, ingested
) -> None:
    headers, _ = customer
    client.post(
        "/api/ratings", json={"movie_id": ingested["id"], "rating": 5.0}, headers=headers
    )
    picks = client.get("/api/recommend/new-releases", headers=headers).json()
    assert ingested["id"] not in {pick["movie"]["id"] for pick in picks}


def test_new_releases_are_honest_for_a_user_without_signal(
    client: TestClient, signup, add_movies, services, ingested
) -> None:
    add_movies(*MOVIES)
    headers, _ = signup(email="cold@example.com")
    picks = client.get("/api/recommend/new-releases", headers=headers).json()

    assert picks
    assert picks[0]["reason"] == "New in the catalog — no personal taste signal yet"


def test_new_releases_are_empty_without_any_recent_film(
    client: TestClient, signup, add_movies, services
) -> None:
    add_movies(*[spec for spec in MOVIES if spec["id"] != 18])
    headers, _ = signup(email="old@example.com")
    assert client.get("/api/recommend/new-releases", headers=headers).json() == []


def test_new_releases_require_auth(client: TestClient, services) -> None:
    assert client.get("/api/recommend/new-releases").status_code == 401


def test_rating_a_new_film_does_not_enter_the_trained_model(
    client: TestClient, customer, services, ingested
) -> None:
    headers, _ = customer
    client.post(
        "/api/ratings", json={"movie_id": ingested["id"], "rating": 5.0}, headers=headers
    )

    with get_session_factory()() as session:
        movie = session.get(Movie, ingested["id"])
        assert movie is not None
        assert movie.in_model is False
        # The rating is stored like any other, so a later retrain can pick the film up.
        assert session.scalar(select(Rating).where(Rating.movie_id == movie.id)) is not None


# ------------------------------------------------------------------ recency (5.12)


def test_recency_window_maths() -> None:
    assert release_date_for(2025) == date(2025, 7, 1)
    assert release_date_for(None) is None
    assert release_date_for(1200) is None  # outside the sane range
    assert age_in_days(2025, today=TODAY) == (TODAY - date(2025, 7, 1)).days
    assert age_in_days(None, today=TODAY) is None

    assert is_new_release(2026, today=TODAY, years=2) is True
    assert is_new_release(2025, today=TODAY, years=2) is True
    assert is_new_release(2020, today=TODAY, years=2) is False
    assert is_new_release(None, today=TODAY, years=2) is False
    assert is_new_release(2026, today=TODAY, years=0) is False

    # A bonus decays with age and vanishes outside the window.
    assert boost_for(2020, today=TODAY, years=2, weight=1.0) == 0.0
    assert boost_for(2026, today=TODAY, years=2, weight=1.0) > boost_for(
        2025, today=TODAY, years=2, weight=1.0
    )
    assert boost_for(2026, today=TODAY, years=2, weight=0.0) == 0.0


def test_the_default_bonus_is_small_enough_to_be_a_tiebreak() -> None:
    settings = Settings()
    assert 0.0 < settings.new_release_boost <= 0.5
    assert settings.new_release_years >= 1


def _batch() -> RecommendationBatch:
    """Two candidates: a clearly better old film and a weaker recent one."""

    def scored(movie_id: int, score: float) -> ScoredItem:
        return ScoredItem(
            movie_id=movie_id,
            score=score,
            predicted_rating=min(5.0, 3.0 + score),
            source_movie_id=None,
            source_rating=None,
        )

    return RecommendationBatch(
        algorithm="item_cf", cold_start=False, items=[scored(10, 0.9), scored(18, 0.5)]
    )


def test_a_small_bonus_does_not_upset_a_clearly_better_pick(
    add_movies, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_movies(*MOVIES)
    monkeypatch.setattr(
        rec_router, "get_settings", lambda: Settings(new_release_boost=0.15, new_release_years=2)
    )

    with get_session_factory()() as session:
        items, _ = rec_router._build_items(
            session,
            _batch(),
            {},
            genres=[],
            year_min=None,
            year_max=None,
            min_score=None,
            limit=5,
        )

    assert [item.movie.id for item in items] == [10, 18]
    assert items[0].new_release is False  # 2000 is long past the window
    assert items[1].new_release is True
    # A re-rank, not a rewrite: the reported score is still the model's.
    assert items[1].score == 0.5


def test_a_large_bonus_lifts_a_new_release_to_the_top(
    add_movies, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_movies(*MOVIES)
    monkeypatch.setattr(
        rec_router, "get_settings", lambda: Settings(new_release_boost=5.0, new_release_years=2)
    )

    with get_session_factory()() as session:
        items, _ = rec_router._build_items(
            session,
            _batch(),
            {},
            genres=[],
            year_min=None,
            year_max=None,
            min_score=None,
            limit=5,
        )

    assert [item.movie.id for item in items] == [18, 10]
    assert items[0].new_release is True
    assert items[0].score == 0.5


def test_a_film_outside_the_window_never_earns_the_badge(
    client: TestClient, customer, services
) -> None:
    headers, _ = customer
    body = client.get("/api/recommend", params={"limit": 8}, headers=headers).json()
    for item in body["recommendations"]:
        year = item["movie"]["release_year"]
        if year is None or year < RECENT_YEAR - 1:
            assert item["new_release"] is False


# ------------------------------------------------------------------ pure feature row (5.11)


def test_feature_row_stays_inside_the_frozen_vocabulary(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    features = load_features(tmp_path / "content_features.npz")
    row = feature_row(
        features,
        genres=["Action", "Sci-Fi", "Kaiju"],  # Kaiju has never been seen
        release_year=2055,
        tags=["dystopia", "space", "unknown-keyword"],
    )

    assert row.shape == (1, features.n_features)
    names = set(features.feature_names)
    # Known labels contribute; unknown ones are ignored so the vocabulary never shifts.
    assert "genre:Action" in names and "genre:Kaiju" not in names
    assert "decade:2050" not in names
    assert row.nnz == 2  # the two known genres and nothing else
    assert float(np.sqrt(row.multiply(row).sum())) == pytest.approx(1.0)


def test_feature_row_without_known_labels_is_empty_not_random() -> None:
    features = build_content_features(
        pd.DataFrame([{"movie_id": 1, "genres": "Drama", "release_year": 1999}])
    )
    row = feature_row(features, genres=["Nope"], release_year=None, tags=["nope"])
    assert row.nnz == 0
