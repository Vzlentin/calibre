from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_session_keyed_state"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

uuid_type = sa.Uuid().with_variant(postgresql.UUID(as_uuid=True), "postgresql")
json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "conformal_state_v2",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("partition", sa.String(), nullable=False),
        sa.Column("run_id", uuid_type, nullable=False),
        sa.Column("state", json_type, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("session_id", "partition"),
    )
    op.execute(
        """
        INSERT INTO conformal_state_v2 (session_id, partition, run_id, state, updated_at)
        SELECT 'legacy-' || substr(replace(CAST(run_id AS TEXT), '-', ''), 1, 57),
               partition,
               run_id,
               state,
               updated_at
        FROM conformal_state
        """
    )
    op.drop_table("conformal_state")
    op.rename_table("conformal_state_v2", "conformal_state")
    op.create_table(
        "pending_observations",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("uid", sa.String(), nullable=False),
        sa.Column("origin", sa.DateTime(timezone=True), nullable=False),
        sa.Column("h", sa.Integer(), nullable=False),
        sa.Column("lo", sa.Float(), nullable=True),
        sa.Column("hi", sa.Float(), nullable=True),
        sa.Column("y_hat", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("session_id", "uid", "origin", "h"),
    )


def downgrade() -> None:
    op.create_table(
        "conformal_state_v1",
        sa.Column("run_id", uuid_type, nullable=False),
        sa.Column("partition", sa.String(), nullable=False),
        sa.Column("state", json_type, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("run_id", "partition"),
    )
    op.execute(
        """
        INSERT INTO conformal_state_v1 (run_id, partition, state, updated_at)
        SELECT run_id, partition, state, updated_at
        FROM conformal_state
        """
    )
    op.drop_table("pending_observations")
    op.drop_table("conformal_state")
    op.rename_table("conformal_state_v1", "conformal_state")
