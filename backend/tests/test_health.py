"""API smoke tests for Phase 0."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_root_advertises_the_api() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/api/health"


def test_health_reports_database_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"]["status"] == "ok"
    # Tests must never depend on a developer's PostgreSQL instance.
    assert body["database"]["dialect"] == "sqlite"


def test_cors_allows_the_vite_dev_origin() -> None:
    with TestClient(app) as client:
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unknown_route_is_404() -> None:
    with TestClient(app) as client:
        response = client.get("/api/does-not-exist")
    assert response.status_code == 404
