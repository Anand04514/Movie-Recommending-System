"""Phase 2.4: JWT auth."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.db import get_session_factory
from backend.app.models import User


def test_signup_returns_a_token_and_normalized_profile(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    headers, user = signup(email="  Ada@Example.COM ")
    assert user["email"] == "ada@example.com"
    assert user["display_name"] == "Ada"
    assert "password_hash" not in user
    assert "password" not in user

    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == user["id"]


def test_duplicate_email_is_rejected(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    signup(email="ada@example.com")
    response = client.post(
        "/api/auth/signup",
        json={
            "email": "ADA@example.com",
            "display_name": "Someone else",
            "password": "another password",
        },
    )
    assert response.status_code == 409


def test_login_and_wrong_password(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    signup(email="ada@example.com", password="correct horse battery")

    ok = client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "correct horse battery"},
    )
    assert ok.status_code == 200
    assert ok.json()["token_type"] == "bearer"

    bad = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": "wrong password"}
    )
    assert bad.status_code == 401

    unknown = client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": "whatever"}
    )
    assert unknown.status_code == 401


def test_me_requires_a_valid_token(client: TestClient) -> None:
    assert client.get("/api/auth/me").status_code == 401
    response = client.get(
        "/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_password_and_email_validation(client: TestClient) -> None:
    short = client.post(
        "/api/auth/signup",
        json={"email": "ada@example.com", "display_name": "Ada", "password": "short"},
    )
    assert short.status_code == 422

    bad_email = client.post(
        "/api/auth/signup",
        json={"email": "not-an-email", "display_name": "Ada", "password": "long enough"},
    )
    assert bad_email.status_code == 422

    blank_name = client.post(
        "/api/auth/signup",
        json={"email": "ada@example.com", "display_name": "   ", "password": "long enough"},
    )
    assert blank_name.status_code == 422


def test_seeded_accounts_cannot_sign_in(client: TestClient) -> None:
    with get_session_factory()() as session:
        session.add(
            User(
                email="demo-1@movielens.invalid",
                display_name="MovieLens user 1",
                password_hash="!",
                is_seeded=True,
            )
        )
        session.commit()
        seeded = session.scalar(
            select(User).where(User.email == "demo-1@movielens.invalid")
        )
        assert seeded is not None and seeded.is_seeded

    response = client.post(
        "/api/auth/login",
        json={"email": "demo-1@movielens.invalid", "password": "any password"},
    )
    assert response.status_code == 401
