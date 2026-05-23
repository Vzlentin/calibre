from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006_data_plane_tables"
down_revision = "0005_lifecycle_store"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "inventory_snapshots",
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("unique_id", sa.String(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_inventory", sa.Float(), nullable=False),
        sa.Column("pipeline", json_type, nullable=False),
        sa.Column("lead_time_depth", sa.Integer(), nullable=False),
        sa.Column("cumulative_costs", json_type, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("tenant", "unique_id", "as_of"),
    )

    op.create_table(
        "sales_records",
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("unique_id", sa.String(), nullable=False),
        sa.Column("ds", sa.DateTime(timezone=True), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("tenant", "unique_id", "ds"),
    )

    op.create_table(
        "order_records",
        sa.Column("order_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("unique_id", sa.String(), nullable=False),
        sa.Column("forecast_origin", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ds", sa.DateTime(timezone=True), nullable=True),
        sa.Column("order_qty", sa.Float(), nullable=False),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("placed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("order_id"),
    )
    op.create_index("ix_order_records_tenant", "order_records", ["tenant"])
    op.create_index("ix_order_records_session_id", "order_records", ["session_id"])
    op.create_index("ix_order_records_unique_id", "order_records", ["unique_id"])


def downgrade() -> None:
    op.drop_index("ix_order_records_unique_id", table_name="order_records")
    op.drop_index("ix_order_records_session_id", table_name="order_records")
    op.drop_index("ix_order_records_tenant", table_name="order_records")
    op.drop_table("order_records")
    op.drop_table("sales_records")
    op.drop_table("inventory_snapshots")
