"""Phase 6: one container serves the React build and the API from the same origin."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.config import Settings


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    directory = tmp_path / "dist"
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text(
        "<!doctype html><title>Movie Recs</title><div id=root></div>", encoding="utf-8"
    )
    (directory / "assets" / "app-abc123.js").write_text("console.log('app');", encoding="utf-8")
    return directory


@pytest.fixture
def spa_client(dist: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = Settings(serve_frontend=True, frontend_dist=str(dist))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.create_app()) as client:
        yield client


def test_the_shell_is_served_at_the_root(spa_client: TestClient) -> None:
    response = spa_client.get("/")
    assert response.status_code == 200
    assert "Movie Recs" in response.text
    assert response.headers["content-type"].startswith("text/html")
    # The shell must be revalidated, or a deploy leaves a cached old bundle in place.
    assert response.headers["cache-control"] == "no-cache"


def test_hashed_assets_are_served(spa_client: TestClient) -> None:
    response = spa_client.get("/assets/app-abc123.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_an_unknown_page_returns_the_shell_for_client_side_routing(spa_client: TestClient) -> None:
    response = spa_client.get("/for-you")
    assert response.status_code == 200
    assert "Movie Recs" in response.text


def test_the_api_keeps_precedence_over_the_shell(spa_client: TestClient) -> None:
    health = spa_client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # An unknown API path is a JSON 404, never the HTML shell.
    missing = spa_client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/json")


def test_a_path_outside_the_build_directory_is_not_served(spa_client: TestClient) -> None:
    # Traversal attempts collapse into the shell rather than reading the filesystem.
    response = spa_client.get("/../../etc/passwd")
    assert response.status_code == 200
    assert response.text.startswith("<!doctype html>")


def test_the_api_still_answers_when_the_build_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(serve_frontend=True, frontend_dist=str(tmp_path / "nope"))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.create_app()) as client:
        assert client.get("/api/health").status_code == 200
    assert "serving the API only" in caplog.text
