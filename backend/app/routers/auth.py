"""Signup, login, password reset, account deletion, and the signed-in user's profile."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.account import (
    consume_reset_token,
    delete_account,
    issue_reset_token,
    reset_link,
    set_password,
)
from backend.app.config import get_settings
from backend.app.db import get_db
from backend.app.deps import get_current_user
from backend.app.mailer import EmailDeliveryError, EmailNotConfigured, reset_email, send_email
from backend.app.models import User
from backend.app.schemas import (
    AccountDeleteRequest,
    LoginRequest,
    MessageOut,
    PasswordResetConfirm,
    PasswordResetRequest,
    SignupRequest,
    TokenResponse,
    UserOut,
)
from backend.app.security import create_access_token, hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# The same sentence for a known address, an unknown one and a seeded demo account. Anything that
# varies here turns the endpoint into a way to test whether somebody has an account.
RESET_REQUESTED_MESSAGE = (
    "If that address has an account, a reset link is on its way. "
    "The link works once and expires shortly."
)


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id, user.token_version or 0),
        user=UserOut.model_validate(user),
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if db.scalar(select(User.id).where(User.email == payload.email)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    user = User(
        email=payload.email,
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from None
    db.refresh(user)
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_response(user)


@router.get("/me", response_model=UserOut)
def read_me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post(
    "/password-reset",
    response_model=MessageOut,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_password_reset(
    payload: PasswordResetRequest, db: Session = Depends(get_db)
) -> MessageOut:
    """Start a reset without revealing whether the address is registered.

    A real account gets a single-use, time-boxed link by email. An unknown address, a seeded demo
    account and a failed send all answer exactly the same way, and the failure is logged for the
    operator rather than shown to the caller.
    """
    settings = get_settings()
    user = db.scalar(select(User).where(User.email == payload.email))

    if user is not None and not user.is_seeded:
        raw_token = issue_reset_token(
            db, user, ttl_minutes=settings.password_reset_ttl_minutes
        )
        db.commit()
        try:
            send_email(
                settings,
                reset_email(
                    user.email,
                    link=reset_link(settings, raw_token),
                    ttl_minutes=settings.password_reset_ttl_minutes,
                    app_name=settings.app_name,
                ),
            )
        except EmailNotConfigured:
            # Deliberately no token in the log: the log is not a private channel, and the link is
            # a credential. The operator mints one with scripts/reset_password.py instead.
            logger.warning(
                "Password reset requested for user %s but SMTP is not configured; no mail was "
                "sent. Run scripts/reset_password.py --email <address> to mint a link.",
                user.id,
            )
        except EmailDeliveryError:
            logger.warning("Password reset email for user %s could not be delivered", user.id)

    return MessageOut(message=RESET_REQUESTED_MESSAGE)


@router.post("/password-reset/confirm", response_model=TokenResponse)
def confirm_password_reset(
    payload: PasswordResetConfirm, db: Session = Depends(get_db)
) -> TokenResponse:
    """Redeem a reset link: set the new password and sign this browser in.

    The response carries a fresh token because the reset bumped the session version, which retired
    every token issued before it — including any the user had on another device.
    """
    user = consume_reset_token(db, payload.token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This reset link is invalid, already used, or expired. Request a new one.",
        )
    set_password(db, user, payload.password)
    db.commit()
    db.refresh(user)
    return _token_response(user)


@router.post("/delete-account", status_code=status.HTTP_204_NO_CONTENT)
def delete_own_account(
    payload: AccountDeleteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Delete the signed-in account, its ratings, feedback and impressions.

    Re-authenticating means a borrowed session cannot destroy the account; a wrong password is a
    403 and changes nothing.
    """
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="That password does not match this account.",
        )
    delete_account(db, user)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
