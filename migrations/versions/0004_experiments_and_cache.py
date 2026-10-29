"""Add A/B experiment assignments, variant attribution and precomputed candidates.

Revision ID: 0004_experiments_cache
Revises: 0003_feedback_logs
Create Date: 2026-10-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_experiments_cache"
down_revision: str | None = "0003_feedback_logs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Attribution columns on impressions: null for a list the user asked for explicitly.
    with op.batch_alter_table("recommendation_logs") as batch:
        batch.add_column(sa.Column("experiment", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("variant", sa.String(length=32), nullable=True))
    op.create_index(
        "ix_recommendation_logs_experiment",
        "recommendation_logs",
        ["experiment", "variant"],
    )

    op.create_table(
        "experiment_assignments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("experiment", sa.String(length=64), nullable=False),
        sa.Column("variant", sa.String(length=32), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "experiment", name="uq_experiment_assignments_user"),
    )
    op.create_index(
        "ix_experiment_assignments_user_id", "experiment_assignments", ["user_id"]
    )
    op.create_index(
        "ix_experiment_assignments_experiment",
        "experiment_assignments",
        ["experiment", "variant"],
    )

    op.create_table(
        "precomputed_recs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("predicted_rating", sa.Float(), nullable=False),
        sa.Column("source_movie_id", sa.Integer(), nullable=True),
        sa.Column("source_rating", sa.Float(), nullable=True),
        sa.Column("neighbors_rated", sa.Integer(), nullable=True),
        sa.Column("contributions", sa.Text(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "algorithm", "movie_id", name="uq_precomputed_user_movie"
        ),
    )
    op.create_index("ix_precomputed_recs_user_id", "precomputed_recs", ["user_id"])
    op.create_index("ix_precomputed_recs_movie_id", "precomputed_recs", ["movie_id"])
    op.create_index(
        "ix_precomputed_recs_user_algorithm",
        "precomputed_recs",
        ["user_id", "algorithm", "rank"],
    )
    op.create_index("ix_precomputed_recs_computed_at", "precomputed_recs", ["computed_at"])


def downgrade() -> None:
    op.drop_index("ix_precomputed_recs_computed_at", table_name="precomputed_recs")
    op.drop_index("ix_precomputed_recs_user_algorithm", table_name="precomputed_recs")
    op.drop_index("ix_precomputed_recs_movie_id", table_name="precomputed_recs")
    op.drop_index("ix_precomputed_recs_user_id", table_name="precomputed_recs")
    op.drop_table("precomputed_recs")

    op.drop_index(
        "ix_experiment_assignments_experiment", table_name="experiment_assignments"
    )
    op.drop_index("ix_experiment_assignments_user_id", table_name="experiment_assignments")
    op.drop_table("experiment_assignments")

    op.drop_index("ix_recommendation_logs_experiment", table_name="recommendation_logs")
    with op.batch_alter_table("recommendation_logs") as batch:
        batch.drop_column("variant")
        batch.drop_column("experiment")
