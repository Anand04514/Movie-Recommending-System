"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import User
from backend.app.security import decode_access_token
from backend.app.tmdb import TmdbClient, build_tmdb_client

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache
def get_tmdb_client() -> TmdbClient:
    """One client per process, so its cache and request throttle are shared."""
    return build_tmdb_client()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to continue.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    decoded = decode_access_token(credentials.credentials)
    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is invalid or has expired. Sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id, token_version = decoded
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if token_version != user.token_version:
        # The password changed since this token was minted, so the session it represents is over.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your password changed. Sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
