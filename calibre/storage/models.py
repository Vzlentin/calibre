from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    idempotency_key: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)


class ConformalState(Base):
    __tablename__ = "conformal_state"

    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    partition: Mapped[str] = mapped_column(String, primary_key=True)
    state: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
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
