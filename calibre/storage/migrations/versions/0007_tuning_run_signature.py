from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_tuning_run_signature"
down_revision = "0006_global_tuning_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tuning_runs",
        sa.Column(
            "config_signature",
            sa.String(),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("tuning_runs", "config_signature")
