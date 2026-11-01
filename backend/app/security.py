"""Password hashing and JWT access tokens."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from backend.app.config import get_settings


def _prepare_password(password: str) -> bytes:
    """Pre-hash so bcrypt's 72-byte limit cannot silently truncate a long password."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare_password(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare_password(password), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        # A seeded user's unusable hash ("!") must fail closed, not raise.
        return False


def create_access_token(user_id: int, token_version: int = 0) -> str:
    """A signed bearer token carrying the account's session generation.

    ``ver`` mirrors ``users.token_version``. A password reset or change bumps the stored version,
    which retires every token minted before it without keeping a session table.
    """
    settings = get_settings()
    issued = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "ver": int(token_version),
        "iat": issued,
        "exp": issued + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> tuple[int, int] | None:
    """The token's user id and session version, or ``None`` when it is unusable.

    A token minted before versions existed has no ``ver`` claim; it is read as version 0 so the
    upgrade does not sign anyone out mid-session.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        subject = payload.get("sub")
        if subject is None:
            return None
        version = payload.get("ver", 0)
        return int(subject), int(version)
    except (jwt.InvalidTokenError, TypeError, ValueError):
        return None
