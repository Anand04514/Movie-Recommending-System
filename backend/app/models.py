"""SQLAlchemy models for users, the movie catalog, and ratings."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    is_seeded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    ratings: Mapped[list[Rating]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Movie(Base):
    """The full catalog. ``id`` is the MovieLens ``movieId`` so links stay stable."""

    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title_display: Mapped[str] = mapped_column(String(512), nullable=False)
    title_search: Mapped[str] = mapped_column(String(512), nullable=False)
    release_year: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    genres: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    imdb_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tmdb_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    poster_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    tmdb_keywords: Mapped[str | None] = mapped_column(Text, nullable=True)
    rating_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    in_model: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="movielens", nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_movies_title_search", "title_search"),
        Index("ix_movies_rating_count", "rating_count"),
        Index(
            "ix_movies_title_search_trgm",
            "title_search",
            postgresql_using="gin",
            postgresql_ops={"title_search": "gin_trgm_ops"},
        ),
    )


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="ratings")
    movie: Mapped[Movie] = relationship()

    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),)


class Feedback(Base):
    """A like, reject or click on a recommended film, used to refine the next list immediately.

    Rejections and likes are both exclusions from future recommendations: either way the user has
    now formed an opinion, and showing the film again wastes a slot. The action is kept for the
    engagement statistics in ``recommendation_logs`` and the accuracy dashboard.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    source_algorithm: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", "action", name="uq_feedback_user_movie_action"),
        Index("ix_feedback_user_action", "user_id", "action"),
    )


class RecommendationLog(Base):
    """An impression: this film was shown to this user by this method at this rank.

    One row per shown film lets the accuracy dashboard answer "how often do shown picks get
    rated or liked?" without joining to a mutable current recommendation list.
    """

    __tablename__ = "recommendation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    shown_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True, nullable=False
    )

    __table_args__ = (
        Index("ix_recommendation_logs_user_shown", "user_id", "shown_at"),
        Index("ix_recommendation_logs_algorithm", "algorithm"),
    )


class ModelRun(Base):
    """One offline evaluation of one method, so results outlive any single README edit.

    ``params`` is a JSON blob holding the sample sizes and settings that are not first-class
    columns: ``{"k": 10, "damping": 4.0, "coverage": 0.12, "evaluated_users": 10000, ...}``.
    """

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    params: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
