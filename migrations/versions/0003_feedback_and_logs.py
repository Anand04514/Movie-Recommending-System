"""Add feedback and recommendation_logs for the real-time feedback loop.

Revision ID: 0003_feedback_logs
Revises: 0002_model_runs
Create Date: 2026-10-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_feedback_logs"
down_revision: str | None = "0002_model_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("source_algorithm", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "movie_id", "action", name="uq_feedback_user_movie_action"
        ),
    )
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])
    op.create_index("ix_feedback_movie_id", "feedback", ["movie_id"])
    op.create_index("ix_feedback_user_action", "feedback", ["user_id", "action"])

    op.create_table(
        "recommendation_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("shown_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recommendation_logs_user_id", "recommendation_logs", ["user_id"])
    op.create_index("ix_recommendation_logs_movie_id", "recommendation_logs", ["movie_id"])
    op.create_index("ix_recommendation_logs_shown_at", "recommendation_logs", ["shown_at"])
    op.create_index(
        "ix_recommendation_logs_user_shown", "recommendation_logs", ["user_id", "shown_at"]
    )
    op.create_index(
        "ix_recommendation_logs_algorithm", "recommendation_logs", ["algorithm"]
    )


def downgrade() -> None:
    op.drop_index("ix_recommendation_logs_algorithm", table_name="recommendation_logs")
    op.drop_index("ix_recommendation_logs_user_shown", table_name="recommendation_logs")
    op.drop_index("ix_recommendation_logs_shown_at", table_name="recommendation_logs")
    op.drop_index("ix_recommendation_logs_movie_id", table_name="recommendation_logs")
    op.drop_index("ix_recommendation_logs_user_id", table_name="recommendation_logs")
    op.drop_table("recommendation_logs")

    op.drop_index("ix_feedback_user_action", table_name="feedback")
    op.drop_index("ix_feedback_movie_id", table_name="feedback")
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_table("feedback")
