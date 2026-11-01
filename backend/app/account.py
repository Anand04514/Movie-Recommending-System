"""Account lifecycle: password reset grants, password changes, and deletion.

Everything here is deliberately free of HTTP concerns so the security rules can be tested on their
own: a reset token is a random 256-bit value stored only as a SHA-256 digest, it is single-use, it
expires in minutes, and redeeming it retires every session the account had open.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.models import (
    ExperimentAssignment,
    Feedback,
    PasswordResetToken,
    PrecomputedRec,
    Rating,
    RecommendationLog,
    User,
    utcnow,
)
from backend.app.security import hash_password

TOKEN_BYTES = 32


def hash_reset_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; treat them as UTC rather than raising on comparison."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def issue_reset_token(
    db: Session,
    user: User,
    *,
    ttl_minutes: int,
    now: datetime | None = None,
) -> str:
    """Mint a reset token for ``user`` and return the raw value to put in the link.

    Any token still outstanding for that account is dropped first, so asking twice leaves exactly
    one working link — the newest one. The caller commits.
    """
    issued = now or utcnow()
    db.execute(
        delete(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
    )
    raw = secrets.token_urlsafe(TOKEN_BYTES)
    db.add(
        PasswordResetToken(
            user_id=user.id,
            token_hash=hash_reset_token(raw),
            created_at=issued,
            expires_at=issued + timedelta(minutes=max(1, ttl_minutes)),
        )
    )
    return raw


def consume_reset_token(
    db: Session,
    raw_token: str,
    *,
    now: datetime | None = None,
) -> User | None:
    """Redeem a token, or return ``None`` for unknown, spent or expired ones.

    The token is marked used immediately, so a link cannot be replayed even if the password update
    that follows fails and the caller rolls back afterwards.
    """
    moment = now or utcnow()
    row = db.scalar(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == hash_reset_token(raw_token)
        )
    )
    if row is None or row.used_at is not None:
        return None
    expires_at = _as_utc(row.expires_at)
    if expires_at is None or expires_at <= moment:
        return None

    user = db.get(User, row.user_id)
    if user is None:
        return None

    row.used_at = moment
    return user


def set_password(db: Session, user: User, password: str) -> None:
    """Store a new password and retire every session and outstanding reset link.

    The version bump is what makes a reset mean something: an attacker holding a token from before
    the change loses it, and the user is signed out everywhere except the browser doing the reset.
    """
    user.password_hash = hash_password(password)
    user.password_changed_at = utcnow()
    user.token_version = (user.token_version or 0) + 1
    # The redeemed token's ``used_at`` is still pending here; flushing keeps the bulk delete below
    # from removing a row the session then tries to UPDATE, which SQLAlchemy reports as stale data.
    db.flush()
    db.execute(
        delete(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
    )


def reset_link(settings: Settings, raw_token: str) -> str:
    """The URL a user clicks. Built from ``APP_BASE_URL``, never from the request's Host header."""
    return f"{settings.app_base_url.rstrip('/')}/?reset={raw_token}"


def delete_account(db: Session, user: User) -> None:
    """Remove the account and everything it produced, in an order that respects the foreign keys.

    Deletion is not deactivation: ratings, feedback, impressions and cached candidates all go, so a
    deleted account leaves no behavioural trace. The A/B counters lose that user's impressions with
    them, which is the correct outcome for data the user asked to be removed.
    """
    for model in (
        Rating,
        Feedback,
        RecommendationLog,
        ExperimentAssignment,
        PrecomputedRec,
        PasswordResetToken,
    ):
        db.execute(delete(model).where(model.user_id == user.id))
    db.delete(user)
    db.commit()
