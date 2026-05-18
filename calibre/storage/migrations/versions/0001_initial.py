from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

uuid_type = sa.Uuid().with_variant(postgresql.UUID(as_uuid=True), "postgresql")
json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=True, unique=True),
        sa.Column("config", json_type, nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
    )
    op.create_table(
        "conformal_state",
        sa.Column("run_id", uuid_type, nullable=False),
        sa.Column("partition", sa.String(), nullable=False),
        sa.Column("state", json_type, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("run_id", "partition"),
    )
    op.create_table(
        "forecast_pointers",
        sa.Column("run_id", uuid_type, nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("uri", sa.String(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("run_id", "kind"),
    )


def downgrade() -> None:
    op.drop_table("forecast_pointers")
    op.drop_table("conformal_state")
    op.drop_table("runs")
