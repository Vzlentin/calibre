from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_pending_obs_metadata"
down_revision = "0002_session_keyed_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pending_observations_v2",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("uid", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("origin", sa.DateTime(timezone=True), nullable=False),
        sa.Column("h", sa.Integer(), nullable=False),
        sa.Column("ds", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lo", sa.Float(), nullable=True),
        sa.Column("hi", sa.Float(), nullable=True),
        sa.Column("y_hat", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("session_id", "uid", "model_name", "origin", "h"),
    )
    op.execute(
        """
        INSERT INTO pending_observations_v2
            (session_id, uid, model_name, origin, h, ds, lo, hi, y_hat)
        SELECT session_id,
               uid,
               '__unknown__',
               origin,
               h,
               origin,
               lo,
               hi,
               y_hat
        FROM pending_observations
        """
    )
    op.drop_table("pending_observations")
    op.rename_table("pending_observations_v2", "pending_observations")


def downgrade() -> None:
    op.create_table(
        "pending_observations_v1",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("uid", sa.String(), nullable=False),
        sa.Column("origin", sa.DateTime(timezone=True), nullable=False),
        sa.Column("h", sa.Integer(), nullable=False),
        sa.Column("lo", sa.Float(), nullable=True),
        sa.Column("hi", sa.Float(), nullable=True),
        sa.Column("y_hat", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("session_id", "uid", "origin", "h"),
    )
    op.execute(
        """
        INSERT INTO pending_observations_v1 (session_id, uid, origin, h, lo, hi, y_hat)
        SELECT session_id, uid, origin, h, lo, hi, y_hat
        FROM pending_observations
        """
    )
    op.drop_table("pending_observations")
    op.rename_table("pending_observations_v1", "pending_observations")
