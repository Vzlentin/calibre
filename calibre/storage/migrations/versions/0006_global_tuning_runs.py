from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006_global_tuning_runs"
down_revision = "0005_lifecycle_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "global_tuning_runs",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("config_signature", sa.String(), nullable=False),
        sa.Column(
            "candidate",
            sa.JSON().with_variant(JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )


def downgrade() -> None:
    op.drop_table("global_tuning_runs")
