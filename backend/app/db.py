"""Database engine, sessions, and startup checks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import get_settings

logger = logging.getLogger(__name__)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """SQLite ignores foreign keys unless each connection opts in."""
    if type(dbapi_connection).__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@lru_cache
def get_engine() -> Engine:
    """Create the process-wide engine, with the connection options each backend needs."""
    url = make_url(get_settings().database_url)
    connect_args = {"check_same_thread": False} if url.get_backend_name() == "sqlite" else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def dialect_name() -> str:
    return get_engine().dialect.name


def init_database() -> None:
    """Enable the extensions the app relies on. Safe to run on every startup.

    ``pg_trgm`` powers partial-title search on PostgreSQL. SQLite has no server extensions, so
    local development simply skips it; search uses ``LIKE`` there until PostgreSQL is used.
    """
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        logger.info("Database dialect is %s; skipping pg_trgm setup", engine.dialect.name)
        return
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    logger.info("pg_trgm extension is available")


def database_health() -> dict[str, str]:
    """Probe the database without leaking connection details."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.warning("Database health check failed", exc_info=True)
        return {"status": "unavailable", "dialect": dialect_name()}
    return {"status": "ok", "dialect": dialect_name()}
