"""Add the model_runs table for offline evaluation results.

Revision ID: 0002_model_runs
Revises: 0001_initial
Create Date: 2026-10-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_model_runs"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        sa.Column("params", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("precision_at_k", sa.Float(), nullable=True),
        sa.Column("recall_at_k", sa.Float(), nullable=True),
        sa.Column("ndcg_at_k", sa.Float(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_runs_algorithm", "model_runs", ["algorithm"])


def downgrade() -> None:
    op.drop_index("ix_model_runs_algorithm", table_name="model_runs")
    op.drop_table("model_runs")
