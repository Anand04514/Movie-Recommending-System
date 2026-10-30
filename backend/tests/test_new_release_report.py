"""PLAN 5.12: the recency row's share and outcomes, measured from logged impressions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from backend.app.db import get_session_factory
from backend.app.models import Feedback, Rating, RecommendationLog, User


def _user() -> User:
    with get_session_factory()() as session:
        user = User(email="report@example.com", password_hash="x", display_name="Report")
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def _log(
    user_id: int,
    movie_id: int,
    *,
    algorithm: str = "new_releases",
    shown_at: datetime,
    score: float = 0.8,
) -> RecommendationLog:
    return RecommendationLog(
        user_id=user_id, movie_id=movie_id, algorithm=algorithm, score=score, shown_at=shown_at
    )


def test_an_empty_log_reports_zeroes_rather_than_inventing_a_rate(client: TestClient) -> None:
    body = client.get("/api/new-releases/report").json()

    assert body["impressions"] == 0
    assert body["total_impressions"] == 0
    assert body["share_of_impressions"] is None
    assert body["like_rate"] is None
    assert body["rated_rate"] is None


def test_share_verdicts_and_ratings_come_from_logged_impressions(
    client: TestClient, add_movies: Callable[..., list[int]]
) -> None:
    add_movies(
        {"id": 1, "title": "New One (2025)", "in_model": False},
        {"id": 2, "title": "New Two (2025)", "in_model": False},
        {"id": 3, "title": "Old Three (1999)", "in_model": True},
    )
    user = _user()
    now = datetime.now(UTC)
    shown = now - timedelta(hours=2)

    with get_session_factory()() as session:
        session.add_all(
            [
                _log(user.id, 1, shown_at=shown),
                _log(user.id, 2, shown_at=shown),
                # A main-list impression: it counts towards the total, never towards the share.
                _log(user.id, 3, algorithm="hybrid", shown_at=shown),
                Feedback(user_id=user.id, movie_id=1, action="liked", created_at=now),
                Feedback(user_id=user.id, movie_id=2, action="rejected", created_at=now),
                # Liked before it was shown: not an outcome of this recommendation.
                Feedback(
                    user_id=user.id,
                    movie_id=3,
                    action="liked",
                    created_at=shown - timedelta(hours=1),
                ),
                Rating(user_id=user.id, movie_id=1, rating=4.5, updated_at=now),
                # Rated before the impression: must not be counted as a response.
                Rating(
                    user_id=user.id,
                    movie_id=2,
                    rating=3.0,
                    updated_at=shown - timedelta(hours=1),
                ),
            ]
        )
        session.commit()

    body = client.get("/api/new-releases/report").json()

    assert body["impressions"] == 2
    assert body["total_impressions"] == 3
    assert body["share_of_impressions"] == 2 / 3
    assert body["users"] == 1
    assert body["films"] == 2
    assert body["likes"] == 1
    assert body["rejects"] == 1
    assert body["clicks"] == 0
    assert body["rated_after"] == 1
    assert body["like_rate"] == 0.5
    assert body["rated_rate"] == 0.5


def test_clicks_are_counted_but_do_not_make_a_verdict(
    client: TestClient, add_movies: Callable[..., list[int]]
) -> None:
    add_movies({"id": 4, "title": "New Four (2025)", "in_model": False})
    user = _user()
    now = datetime.now(UTC)

    with get_session_factory()() as session:
        session.add(_log(user.id, 4, shown_at=now - timedelta(minutes=5)))
        session.add(
            Feedback(user_id=user.id, movie_id=4, action="clicked", created_at=now)
        )
        session.commit()

    body = client.get("/api/new-releases/report").json()

    assert body["clicks"] == 1
    assert body["likes"] == 0
    assert body["rejects"] == 0
    # A click is interest, not a verdict: with no like or reject the rate stays unset.
    assert body["like_rate"] is None


def test_the_report_is_public_and_carries_no_user_identifiers(client: TestClient) -> None:
    response = client.get("/api/new-releases/report")

    assert response.status_code == 200
    assert "user_id" not in response.text
    assert "email" not in response.text
