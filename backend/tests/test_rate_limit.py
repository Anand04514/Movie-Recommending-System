"""Rate limiting on signup, login and password reset.

What these tests pin down: a client or account that keeps going gets a 429 with a usable
``Retry-After``, the budget recovers when the window passes, separate callers do not share a bucket,
a forged ``X-Forwarded-For`` cannot buy a fresh budget, and a correct password is never blocked by
somebody else's failed guesses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app import ratelimit
from backend.app.config import get_settings
from backend.app.db import get_session_factory
from backend.app.models import User
from backend.app.ratelimit import Rule, check, parse_rule

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def restore_clock() -> Iterator[None]:
    """Any test that moves time must not leave the module's clock displaced."""
    yield
    ratelimit.set_clock(__import__("time").monotonic)


def configured(monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    settings = get_settings()
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)


def test_rule_parsing() -> None:
    rule = parse_rule("10/60", "login")
    assert rule == Rule(scope="login", limit=10, window_seconds=60.0)
    assert rule.enabled

    # A typo in an environment variable disables the rule instead of taking the page down.
    for bad in ("", "10", "ten/60", None):
        assert not parse_rule(bad, "login").enabled  # type: ignore[arg-type]


def test_sliding_window_allows_then_recovers() -> None:
    rule = Rule(scope="t", limit=2, window_seconds=10)
    assert check(("t", "a"), rule, now=0.0) is None
    assert check(("t", "a"), rule, now=1.0) is None

    wait = check(("t", "a"), rule, now=2.0)
    assert wait == pytest.approx(8.0)  # the first hit ages out at t=10

    # A rejected call is not recorded, so the wait does not grow by retrying.
    assert check(("t", "a"), rule, now=3.0) == pytest.approx(7.0)
    assert check(("t", "a"), rule, now=10.0) is None


def test_a_correct_password_is_never_blocked_by_failed_guesses(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], monkeypatch
) -> None:
    signup(email="ada@example.com", password=PASSWORD)
    # Small identity budget, generous client budget, so only the failure counter can bite.
    configured(
        monkeypatch,
        rate_limit_login="100/60",
        rate_limit_login_identity="2/600",
    )

    first = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "a"})
    second = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "b"})
    assert (first.status_code, second.status_code) == (401, 401)

    # Two failures have spent the identity budget; a third guess is refused without being checked.
    third = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "c"})
    assert third.status_code == 429

    # The owner still gets in: successes were never counted against that budget.
    owner = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    )
    assert owner.status_code == 200


def test_login_is_limited_per_client(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], monkeypatch
) -> None:
    signup(email="ada@example.com", password=PASSWORD)
    configured(monkeypatch, rate_limit_login="3/60", rate_limit_login_identity="100/600")

    codes = [
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": "guess"}
        ).status_code
        for _ in range(4)
    ]
    assert codes[:3] == [401, 401, 401]
    assert codes[3] == 429

    refused = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    )
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) > 0
    assert "Too many attempts" in refused.json()["detail"]
    assert "Try again in" in refused.json()["detail"]


def test_signup_is_limited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(monkeypatch, rate_limit_signup="2/600")

    first = client.post(
        "/api/auth/signup",
        json={"email": "one@example.com", "display_name": "One", "password": PASSWORD},
    )
    second = client.post(
        "/api/auth/signup",
        json={"email": "two@example.com", "display_name": "Two", "password": PASSWORD},
    )
    third = client.post(
        "/api/auth/signup",
        json={"email": "three@example.com", "display_name": "Three", "password": PASSWORD},
    )

    assert (first.status_code, second.status_code) == (201, 201)
    assert third.status_code == 429


def test_password_reset_is_limited_per_address_but_not_across_addresses(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signup(email="ada@example.com")
    signup(email="grace@example.com")
    configured(
        monkeypatch,
        rate_limit_password_reset="50/900",
        rate_limit_password_reset_identity="2/3600",
    )

    first_two = [
        client.post("/api/auth/password-reset", json={"email": "ada@example.com"}).status_code
        for _ in range(2)
    ]
    blocked = client.post("/api/auth/password-reset", json={"email": "ada@example.com"})
    someone_else = client.post("/api/auth/password-reset", json={"email": "grace@example.com"})

    assert first_two == [202, 202]
    assert blocked.status_code == 429
    # The identity budget is per mailbox, so flooding one inbox cannot consume another's allowance.
    # (With the shipped 5/900 client rule the same machine would still be throttled; this test
    # raises that rule to isolate the identity dimension.)
    assert someone_else.status_code == 202


def test_separate_clients_do_not_share_a_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    configured(monkeypatch, rate_limit_signup="1/600")
    body = {"display_name": "X", "password": PASSWORD}

    first = TestClient(client_app := _app(), client=("10.0.0.1", 1111))
    second = TestClient(client_app, client=("10.0.0.2", 2222))

    one = first.post("/api/auth/signup", json={**body, "email": "a@example.com"})
    two = first.post("/api/auth/signup", json={**body, "email": "b@example.com"})
    other = second.post("/api/auth/signup", json={**body, "email": "c@example.com"})

    assert (one.status_code, two.status_code) == (201, 429)
    assert other.status_code == 201


def test_forwarded_for_is_ignored_until_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured(monkeypatch, rate_limit_signup="1/600", trust_forwarded_for=False)
    body = {"display_name": "X", "password": PASSWORD}
    client = TestClient(_app(), client=("10.0.0.9", 9999))

    first = client.post(
        "/api/auth/signup",
        json={**body, "email": "d@example.com"},
        headers={"X-Forwarded-For": "203.0.113.1"},
    )
    # Same forged header, but the limiter keys on the real peer, so the second call is refused.
    second = client.post(
        "/api/auth/signup",
        json={**body, "email": "e@example.com"},
        headers={"X-Forwarded-For": "203.0.113.2"},
    )
    assert (first.status_code, second.status_code) == (201, 429)

    # Behind a proxy, the first hop is the real client and each caller gets its own budget.
    ratelimit.reset()
    configured(monkeypatch, trust_forwarded_for=True)
    third = client.post(
        "/api/auth/signup",
        json={**body, "email": "f@example.com"},
        headers={"X-Forwarded-For": "203.0.113.3, 10.0.0.9"},
    )
    fourth = client.post(
        "/api/auth/signup",
        json={**body, "email": "g@example.com"},
        headers={"X-Forwarded-For": "203.0.113.4, 10.0.0.9"},
    )
    assert (third.status_code, fourth.status_code) == (201, 201)


def test_limits_can_be_switched_off(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], monkeypatch
) -> None:
    signup(email="ada@example.com", password=PASSWORD)
    configured(
        monkeypatch,
        rate_limit_enabled=False,
        rate_limit_login="1/60",
        rate_limit_login_identity="1/600",
    )

    codes = [
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": "wrong"}
        ).status_code
        for _ in range(5)
    ]
    assert codes == [401] * 5


def test_the_window_passing_restores_access(
    client: TestClient, signup: Callable[..., tuple[dict[str, str], dict]], monkeypatch
) -> None:
    signup(email="ada@example.com", password=PASSWORD)
    configured(monkeypatch, rate_limit_login="1/60", rate_limit_login_identity="100/600")

    now = [1000.0]
    ratelimit.set_clock(lambda: now[0])

    assert (
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": "wrong"}
        ).status_code
        == 401
    )
    blocked = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    )
    assert blocked.status_code == 429

    now[0] += 61
    assert (
        client.post(
            "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
        ).status_code
        == 200
    )


def _app():
    from backend.app.main import app

    return app


def test_a_throttled_login_leaves_the_account_intact(
    client: TestClient,
    signup: Callable[..., tuple[dict[str, str], dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 must not be a successful sign-in, and must not disturb the account behind it."""
    headers, user = signup(email="ada@example.com", password=PASSWORD)
    configured(monkeypatch, rate_limit_login="1/60")

    client.post("/api/auth/login", json={"email": "ada@example.com", "password": "wrong"})
    blocked = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    )

    assert blocked.status_code == 429
    assert "access_token" not in blocked.json()
    assert client.get("/api/auth/me", headers=headers).status_code == 200
    with get_session_factory()() as session:
        assert session.get(User, user["id"]) is not None
