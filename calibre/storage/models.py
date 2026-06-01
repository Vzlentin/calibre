from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JsonDict = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    idempotency_key: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    config: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    row_count: Mapped[int | None] = mapped_column(nullable=True)


class ConformalState(Base):
    __tablename__ = "conformal_state"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    partition: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), nullable=False)
    state: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )


class PendingObservation(Base):
    __tablename__ = "pending_observations"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    uid: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    origin: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    h: Mapped[int] = mapped_column(Integer, primary_key=True)
    ds: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lo: Mapped[float | None] = mapped_column(Float, nullable=True)
    hi: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_hat: Mapped[float | None] = mapped_column(Float, nullable=True)


class TuningRun(Base):
    __tablename__ = "tuning_runs"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    unique_id: Mapped[str] = mapped_column(String, primary_key=True)
    config_signature: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    candidate: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )


class GlobalTuningRun(Base):
    """Single global/panel HPO result per session.

    Global HPO tunes one config shared across all series, so its natural key is
    the session, not the unique_id. ``config_signature`` captures the tuning
    inputs not already encoded in ``session_id`` (search space, objective,
    n_trials, horizon, origins) so a cached result is only reused when those
    inputs match.
    """

    __tablename__ = "global_tuning_runs"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config_signature: Mapped[str] = mapped_column(String, nullable=False)
    candidate: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )


class ForecastPointer(Base):
    __tablename__ = "forecast_pointers"

    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    kind: Mapped[str] = mapped_column(String, primary_key=True)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    byte_size: Mapped[int] = mapped_column(nullable=False)


class LifecycleFitRecord(Base):
    """Fit lifecycle metadata. Data-plane frames live as parquet under
    ``frame_uris`` (object store), never inline on this row."""

    __tablename__ = "lifecycle_fit_records"

    fit_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sku_set: Mapped[list] = mapped_column(JsonDict, nullable=False)
    forecaster_config: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    freq: Mapped[str] = mapped_column(String, nullable=False)
    conformal_config: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    artifact_urls: Mapped[dict] = mapped_column(JsonDict, nullable=False, default=dict)
    frame_uris: Mapped[dict] = mapped_column(JsonDict, nullable=False, default=dict)
    # Insertion order — the canonical "first fit" for a session (a session can
    # hold several fits, since session_id is derived from config). Postgres has
    # sub-second resolution; fit_id breaks any tie deterministically.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LifecycleConformalState(Base):
    """Session-owned conformal state, keyed by ``(session_id, partition)`` and
    referenced by fits rather than copied onto each fit row."""

    __tablename__ = "lifecycle_conformal_state"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    partition: Mapped[str] = mapped_column(String, primary_key=True)
    state: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )


class LifecycleTuneRecord(Base):
    __tablename__ = "lifecycle_tune_records"

    study_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    sku_set: Mapped[list] = mapped_column(JsonDict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    best_candidates: Mapped[dict] = mapped_column(JsonDict, nullable=False, default=dict)


class Sales(Base):
    """Project-owned sales history, read by ``SqlSalesAdapter`` at fit/tune time.

    One row per ``(unique_id, ds)``; ``as_of`` marks when the figure was
    recorded so the adapter can answer point-in-time queries (``as_of <= origin``)
    without leaking future revisions into a backtest."""

    __tablename__ = "sales"

    unique_id: Mapped[str] = mapped_column(String, primary_key=True)
    ds: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Order(Base):
    """Persistent order ledger written by ``/order`` and read by ``/sessions``.

    Natural key is the decision tuple ``(session_id, unique_id, forecast_origin,
    model_name)``; ``detail`` carries the full JSON-safe order row, while
    ``order_qty`` and the key columns are denormalized for querying."""

    __tablename__ = "orders"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    unique_id: Mapped[str] = mapped_column(String, primary_key=True)
    forecast_origin: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    model_name: Mapped[str] = mapped_column(String, primary_key=True, default="")
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    order_qty: Mapped[float] = mapped_column(Float, nullable=False)
    detail: Mapped[dict] = mapped_column(JsonDict, nullable=False, default=dict)
    placed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("ix_orders_tenant_unique_id", "tenant", "unique_id"),)
