from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_lifecycle_store"
down_revision = "0004_tuning_runs"
branch_labels = None
depends_on = None


def _json() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "lifecycle_fit_records",
        sa.Column("fit_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("sku_set", _json(), nullable=False),
        sa.Column("forecaster_config", _json(), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("freq", sa.String(), nullable=False),
        sa.Column("conformal_config", _json(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("artifact_urls", _json(), nullable=False),
        sa.Column("frame_uris", _json(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("fit_id"),
    )
    op.create_index("ix_lifecycle_fit_records_session_id", "lifecycle_fit_records", ["session_id"])
    op.create_index("ix_lifecycle_fit_records_tenant", "lifecycle_fit_records", ["tenant"])
    op.create_table(
        "lifecycle_conformal_state",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("partition", sa.String(), nullable=False),
        sa.Column("state", _json(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("session_id", "partition"),
    )
    op.create_table(
        "lifecycle_tune_records",
        sa.Column("study_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("sku_set", _json(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("best_candidates", _json(), nullable=False),
        sa.PrimaryKeyConstraint("study_id"),
    )


def downgrade() -> None:
    op.drop_table("lifecycle_tune_records")
    op.drop_table("lifecycle_conformal_state")
    op.drop_index("ix_lifecycle_fit_records_tenant", table_name="lifecycle_fit_records")
    op.drop_index("ix_lifecycle_fit_records_session_id", table_name="lifecycle_fit_records")
    op.drop_table("lifecycle_fit_records")
