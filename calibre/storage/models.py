from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, func
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


class FitRecordRow(Base):
    __tablename__ = "fit_records"

    fit_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    sku_set: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    forecaster_config: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    freq: Mapped[str] = mapped_column(String, nullable=False)
    history: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    future_x: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    conformal_config: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    artifact_urls: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    last_forecast: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    last_calibrated: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    last_orders: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    conformal_state: Mapped[dict | None] = mapped_column(JsonDict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class TuneRecordRow(Base):
    __tablename__ = "tune_records"

    study_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    sku_set: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    best_candidates: Mapped[dict] = mapped_column(JsonDict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
