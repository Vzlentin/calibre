from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_lifecycle_store"
down_revision = "0004_tuning_runs"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "fit_records",
        sa.Column("fit_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("sku_set", json_type, nullable=False),
        sa.Column("forecaster_config", json_type, nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("freq", sa.String(), nullable=False),
        sa.Column("history", json_type, nullable=False),
        sa.Column("future_x", json_type, nullable=True),
        sa.Column("conformal_config", json_type, nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("artifact_urls", json_type, nullable=False),
        sa.Column("last_forecast", json_type, nullable=True),
        sa.Column("last_calibrated", json_type, nullable=True),
        sa.Column("last_orders", json_type, nullable=True),
        sa.Column("conformal_state", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("fit_id"),
    )
    op.create_index("ix_fit_records_session_id", "fit_records", ["session_id"])
    op.create_index("ix_fit_records_tenant", "fit_records", ["tenant"])

    op.create_table(
        "tune_records",
        sa.Column("study_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("sku_set", json_type, nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("best_candidates", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("study_id"),
    )
    op.create_index("ix_tune_records_session_id", "tune_records", ["session_id"])
    op.create_index("ix_tune_records_tenant", "tune_records", ["tenant"])


def downgrade() -> None:
    op.drop_index("ix_tune_records_tenant", table_name="tune_records")
    op.drop_index("ix_tune_records_session_id", table_name="tune_records")
    op.drop_table("tune_records")
    op.drop_index("ix_fit_records_tenant", table_name="fit_records")
    op.drop_index("ix_fit_records_session_id", table_name="fit_records")
    op.drop_table("fit_records")
