"""The account lifecycle: password reset grants and account deletion.

The claims these tests defend are the ones a reader would otherwise have to take on trust:
a reset link works once, stops working when it expires, retires every older session, and cannot be
used to discover whether an address has an account. Deletion removes the account's data, not just
its row.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.app.config import get_settings
from backend.app.db import get_session_factory
from backend.app.mailer import Email, EmailNotConfigured, send_email
from backend.app.models import (
    ExperimentAssignment,
    Feedback,
    PasswordResetToken,
    PrecomputedRec,
    Rating,
    RecommendationLog,
    User,
)

PASSWORD = "correct horse battery"
NEW_PASSWORD = "a different long password"


class RecordedSMTP:
    """Stands in for smtplib.SMTP so a test can read the message instead of sending it."""

    sent: list[dict[str, object]] = []

    def __init__(self, host, port, timeout=None) -> None:
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def starttls(self) -> None:
        pass

    def login(self, username, password) -> None:
        self.username = username

    def send_message(self, message, from_addr=None, to_addrs=None) -> None:
        RecordedSMTP.sent.append(
            {"text": message.get_content(), "to": to_addrs, "from": from_addr}
        )


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Configure a fake mail server for one test and return the outbox."""
    settings = get_settings()
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_from", "Movie Recs <no-reply@example.com>")
    monkeypatch.setattr(settings, "app_base_url", "https://recs.example.com")
    RecordedSMTP.sent = []
    monkeypatch.setattr("backend.app.mailer.smtplib.SMTP", RecordedSMTP)
    return RecordedSMTP.sent


def token_from(outbox: list[dict[str, object]]) -> str:
    match = re.search(r"\?reset=([A-Za-z0-9_\-]+)", str(outbox[-1]["text"]))
    assert match is not None, outbox[-1]["text"]
    return match.group(1)


def user_id_for(email: str) -> int:
    with get_session_factory()() as session:
        row = session.scalar(select(User).where(User.email == email))
        assert row is not None
        return row.id


def test_reset_request_never_reveals_whether_an_account_exists(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    signup(email="ada@example.com")

    known = client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    unknown = client.post("/api/auth/password-reset", json={"email": "nobody@example.com"})

    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()
    assert "reset" not in known.text.lower().replace("reset link", "")
    # One message went out, for the account that exists.
    assert len(smtp) == 1
    assert smtp[0]["to"] == ["ada@example.com"]


def test_the_emailed_link_resets_the_password_and_signs_the_user_in(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    headers, _ = signup(email="ada@example.com", password=PASSWORD)
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    token = token_from(smtp)

    response = client.post(
        "/api/auth/password-reset/confirm", json={"token": token, "password": NEW_PASSWORD}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["access_token"]
    assert body["user"]["email"] == "ada@example.com"

    # The new token works, the old password is gone, the new one signs in.
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    fresh = {"Authorization": f"Bearer {body['access_token']}"}
    assert client.get("/api/auth/me", headers=fresh).status_code == 200
    assert (
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": NEW_PASSWORD}
        ).status_code
        == 200
    )


def test_a_reset_link_works_exactly_once(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    signup(email="ada@example.com")
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    token = token_from(smtp)

    first = client.post(
        "/api/auth/password-reset/confirm", json={"token": token, "password": NEW_PASSWORD}
    )
    second = client.post(
        "/api/auth/password-reset/confirm", json={"token": token, "password": "yet another one"}
    )

    assert first.status_code == 200
    assert second.status_code == 400
    assert "invalid" in second.json()["detail"].lower()


def test_an_expired_link_is_refused(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    signup(email="ada@example.com")
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    token = token_from(smtp)

    with get_session_factory()() as session:
        row = session.scalar(select(PasswordResetToken))
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        session.commit()

    response = client.post(
        "/api/auth/password-reset/confirm", json={"token": token, "password": NEW_PASSWORD}
    )
    assert response.status_code == 400


def test_asking_twice_leaves_only_the_newest_link_working(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    signup(email="ada@example.com")
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    first = token_from(smtp)
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    second = token_from(smtp)

    assert first != second
    stale = client.post(
        "/api/auth/password-reset/confirm", json={"token": first, "password": NEW_PASSWORD}
    )
    current = client.post(
        "/api/auth/password-reset/confirm", json={"token": second, "password": NEW_PASSWORD}
    )
    assert stale.status_code == 400
    assert current.status_code == 200


def test_a_reset_retires_sessions_on_other_devices(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    laptop, _ = signup(email="ada@example.com", password=PASSWORD)
    phone = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    ).json()["access_token"]
    phone_headers = {"Authorization": f"Bearer {phone}"}

    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    confirmed = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token_from(smtp), "password": NEW_PASSWORD},
    )
    assert confirmed.status_code == 200

    for headers in (laptop, phone_headers):
        response = client.get("/api/auth/me", headers=headers)
        assert response.status_code == 401
        assert "password changed" in response.json()["detail"].lower()


def test_the_token_is_never_stored_in_the_clear(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    signup(email="ada@example.com")
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    token = token_from(smtp)

    with get_session_factory()() as session:
        row = session.scalar(select(PasswordResetToken))
        assert row is not None
        assert row.token_hash != token
        assert token not in row.token_hash


def test_a_seeded_demo_account_gets_no_link(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
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

    response = client.post(
        "/api/auth/password-reset", json={"email": "demo-1@movielens.invalid"}
    )
    assert response.status_code == 202
    assert smtp == []
    with get_session_factory()() as session:
        assert session.scalar(select(func.count()).select_from(PasswordResetToken)) == 0


def test_without_smtp_the_request_still_answers_and_sends_nothing(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    signup(email="ada@example.com")
    settings = get_settings()
    assert not settings.smtp_configured

    response = client.post("/api/auth/password-reset", json={"email": "ada@example.com"})

    assert response.status_code == 202
    assert "reset link" in response.json()["message"]
    # The token exists (an operator can mint another), but it is not handed back over HTTP.
    assert "token" not in response.text.lower()


def test_the_mailer_refuses_to_pretend_when_unconfigured() -> None:
    settings = get_settings()
    with pytest.raises(EmailNotConfigured):
        send_email(settings, Email(to="ada@example.com", subject="Hi", text="Hello"))


def test_deletion_requires_the_password(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    headers, user = signup(email="ada@example.com", password=PASSWORD)

    response = client.post(
        "/api/auth/delete-account", json={"password": "not my password"}, headers=headers
    )
    assert response.status_code == 403
    assert client.get("/api/auth/me", headers=headers).json()["id"] == user["id"]


def test_deletion_removes_the_account_and_its_traces(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    add_movies: Callable[..., list[int]],
) -> None:
    headers, user = signup(email="ada@example.com", password=PASSWORD)
    movie_id = add_movies({"id": 1, "title": "Heat", "year": 1995})[0]

    assert (
        client.post("/api/ratings", json={"movie_id": movie_id, "rating": 4.5}, headers=headers)
        .status_code
        == 200
    )
    client.post("/api/feedback", json={"movie_id": movie_id, "action": "liked"}, headers=headers)
    # An impression and an experiment assignment, as the recommendation path would write them.
    with get_session_factory()() as session:
        session.add_all(
            [
                RecommendationLog(
                    user_id=user["id"], movie_id=movie_id, algorithm="hybrid", score=1.0
                ),
                ExperimentAssignment(
                    user_id=user["id"], experiment="default_model_v1", variant="hybrid"
                ),
                PrecomputedRec(
                    user_id=user["id"],
                    algorithm="hybrid",
                    movie_id=movie_id,
                    score=1.0,
                    rank=1,
                    predicted_rating=4.0,
                ),
            ]
        )
        session.commit()

    response = client.post(
        "/api/auth/delete-account", json={"password": PASSWORD}, headers=headers
    )
    assert response.status_code == 204, response.text

    with get_session_factory()() as session:
        for model in (
            User,
            Rating,
            Feedback,
            RecommendationLog,
            ExperimentAssignment,
            PrecomputedRec,
            PasswordResetToken,
        ):
            remaining = session.scalar(
                select(func.count())
                .select_from(model)
                .where(model.user_id == user["id"])
                if model is not User
                else select(func.count()).select_from(User).where(User.id == user["id"])
            )
            assert remaining == 0, f"{model.__name__} rows survived deletion"

    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert (
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
        ).status_code
        == 401
    )
    # The catalog is untouched: deleting a user never removes films.
    assert client.get(f"/api/movies/{movie_id}").status_code == 200


def test_a_new_account_starts_at_session_version_zero(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]]
) -> None:
    headers, _ = signup(email="ada@example.com")
    with get_session_factory()() as session:
        row = session.scalar(select(User).where(User.email == "ada@example.com"))
        assert row is not None and row.token_version == 0
    assert client.get("/api/auth/me", headers=headers).status_code == 200


def test_reset_confirm_rejects_a_weak_password(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], smtp: list
) -> None:
    signup(email="ada@example.com")
    client.post("/api/auth/password-reset", json={"email": "ada@example.com"})

    response = client.post(
        "/api/auth/password-reset/confirm", json={"token": token_from(smtp), "password": "short"}
    )
    assert response.status_code == 422
