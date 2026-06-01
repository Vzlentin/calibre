from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008_orders_and_sales"
down_revision = "0007_tuning_run_signature"
branch_labels = None
depends_on = None


def _json() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "sales",
        sa.Column("unique_id", sa.String(), nullable=False),
        sa.Column("ds", sa.DateTime(timezone=True), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("unique_id", "ds"),
    )
    op.create_table(
        "orders",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("unique_id", sa.String(), nullable=False),
        sa.Column("forecast_origin", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("order_qty", sa.Float(), nullable=False),
        sa.Column("detail", _json(), nullable=False),
        sa.Column(
            "placed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("session_id", "unique_id", "forecast_origin", "model_name"),
    )
    op.create_index("ix_orders_tenant_unique_id", "orders", ["tenant", "unique_id"])


def downgrade() -> None:
    op.drop_index("ix_orders_tenant_unique_id", table_name="orders")
    op.drop_table("orders")
    op.drop_table("sales")
