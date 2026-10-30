"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from backend.app.config import Settings, get_settings
from backend.app.db import database_health, init_database
from backend.app.routers import (
    auth,
    feedback,
    movies,
    onboarding,
    ratings,
    recommendations,
)

logger = logging.getLogger(__name__)

# Paths the SPA fallback must never answer with HTML: an unknown API route is a 404, not a page.
RESERVED_PREFIXES = ("api/", "docs", "redoc", "openapi.json")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepare the database, but let the API start (and report degraded) if it is down."""
    try:
        init_database()
    except SQLAlchemyError:
        logger.warning("Database initialization failed; /api/health will report it", exc_info=True)
    yield


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built React app from the API's own origin, when a build is present.

    Production then needs one container, one port and no CORS: the SPA calls ``/api`` relatively,
    exactly as it does behind the Vite dev proxy. The API keeps precedence because its routes are
    registered first, and an unknown ``/api`` path still answers a JSON 404 rather than the shell.

    Missing build output is a warning, not a crash: the API is still useful on its own.
    """
    directory = Path(settings.frontend_dist)
    index = directory / "index.html"
    if not index.is_file():
        logger.warning(
            "SERVE_FRONTEND is on but %s was not found; serving the API only", index
        )
        return

    assets = directory / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def spa_root() -> FileResponse:
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    # Registered after every router, so an API path always wins over the shell.
    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> FileResponse:
        if path.startswith(RESERVED_PREFIXES):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (directory / path).resolve()
        # Only files inside the build directory; a crafted ``../`` path must not read elsewhere.
        if candidate.is_file() and directory.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="MovieLens-based recommender API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        auth.router,
        movies.router,
        ratings.router,
        recommendations.router,
        feedback.router,
        onboarding.router,
    ):
        app.include_router(router)

    @app.get("/api/health", tags=["meta"])
    def health() -> JSONResponse:
        database = database_health()
        healthy = database["status"] == "ok"
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "app": settings.app_name,
                "version": settings.version,
                "database": database,
            },
        )

    if settings.serve_frontend:
        # Mounted last so every API route, including /api/health, matches before the shell's
        # catch-all. The app owns "/" in production; the docs and health endpoints remain.
        _mount_frontend(app, settings)
    else:

        @app.get("/", tags=["meta"])
        def read_root() -> dict[str, str]:
            return {
                "name": settings.app_name,
                "version": settings.version,
                "docs": "/docs",
                "health": "/api/health",
            }

    return app


app = create_app()
