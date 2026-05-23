"""Move lifecycle conformal state out of fit_records into its own table.

Per the PR #38 review (FIX.md item 1), conformal state is session-owned
mutable data and must not be duplicated on every fit row. This migration
creates ``lifecycle_conformal_state`` keyed by ``(session_id, partition)``
and drops the ``conformal_state`` JSON column from ``fit_records``.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008_lifecycle_conformal_state"
down_revision = "0007_tune_oracle_cost"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "lifecycle_conformal_state",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("partition", sa.String(), nullable=False),
        sa.Column("state", json_type, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("session_id", "partition"),
    )
    with op.batch_alter_table("fit_records") as batch_op:
        batch_op.drop_column("conformal_state")


def downgrade() -> None:
    with op.batch_alter_table("fit_records") as batch_op:
        batch_op.add_column(
            sa.Column(
                "conformal_state",
                json_type,
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )
    op.drop_table("lifecycle_conformal_state")
