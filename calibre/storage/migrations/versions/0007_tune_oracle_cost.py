from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_tune_oracle_cost"
down_revision = "0006_data_plane_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tune_records", sa.Column("oracle_cost", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("tune_records", "oracle_cost")
