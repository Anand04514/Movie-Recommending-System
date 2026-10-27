"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.app.config import get_settings
from backend.app.db import database_health, init_database
from backend.app.routers import auth, movies, ratings, recommendations

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepare the database, but let the API start (and report degraded) if it is down."""
    try:
        init_database()
    except SQLAlchemyError:
        logger.warning("Database initialization failed; /api/health will report it", exc_info=True)
    yield


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

    for router in (auth.router, movies.router, ratings.router, recommendations.router):
        app.include_router(router)

    @app.get("/", tags=["meta"])
    def read_root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/api/health",
        }

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

    return app


app = create_app()
