"""Initial users, movies, and ratings schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _trgm_available(connection) -> bool:
    """Install ``pg_trgm`` when the server allows it, and report whether it worked.

    The extension is an optimisation for search, not a dependency of the schema: a PostgreSQL
    build without contrib, or a role without permission to install extensions, must still migrate
    and serve. The attempt runs inside a SAVEPOINT because a failed statement aborts the whole
    PostgreSQL transaction that Alembic is running in.
    """
    if connection.dialect.name != "postgresql":
        return False
    try:
        with connection.begin_nested():
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    except sa.exc.DBAPIError:
        pass
    return (
        connection.exec_driver_sql(
            "SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"
        ).first()
        is not None
    )


def upgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"
    has_trgm = _trgm_available(op.get_bind()) if is_postgres else False

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=80), nullable=False),
        sa.Column("is_seeded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "movies",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("title_display", sa.String(length=512), nullable=False),
        sa.Column("title_search", sa.String(length=512), nullable=False),
        sa.Column("release_year", sa.SmallInteger(), nullable=True),
        sa.Column("genres", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("imdb_id", sa.Integer(), nullable=True),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("poster_path", sa.String(length=255), nullable=True),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("tmdb_keywords", sa.Text(), nullable=True),
        sa.Column("rating_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("in_model", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="movielens"),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_movies_title_search", "movies", ["title_search"])
    op.create_index("ix_movies_rating_count", "movies", ["rating_count"])
    if has_trgm:
        # gin_trgm_ops only exists when the extension does, so the index follows the extension.
        op.create_index(
            "ix_movies_title_search_trgm",
            "movies",
            ["title_search"],
            postgresql_using="gin",
            postgresql_ops={"title_search": "gin_trgm_ops"},
        )

    op.create_table(
        "ratings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
    )
    op.create_index("ix_ratings_movie_id", "ratings", ["movie_id"])
    op.create_index("ix_ratings_user_id", "ratings", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_ratings_user_id", table_name="ratings")
    op.drop_index("ix_ratings_movie_id", table_name="ratings")
    op.drop_table("ratings")
    if op.get_bind().dialect.name == "postgresql":
        # The trigram index is skipped when the extension was unavailable, so drop tolerantly.
        op.execute("DROP INDEX IF EXISTS ix_movies_title_search_trgm")
    op.drop_index("ix_movies_rating_count", table_name="movies")
    op.drop_index("ix_movies_title_search", table_name="movies")
    op.drop_table("movies")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
