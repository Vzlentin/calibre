from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, Y
from calibre.execution.io import read_parquet
from calibre.ordering.simulation.state import ProductState, make_pipeline
from calibre.storage.models import InventorySnapshot, OrderRecord, SalesRecord
from calibre.storage.postgres import session_scope


class SqlInventoryAdapter:
    """InventoryAdapter backed by the project SQL store."""

    def __init__(self, factory: sessionmaker[Session], *, tenant: str) -> None:
        self.factory = factory
        self.tenant = tenant

    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState:
        with session_scope(self.factory) as session:
            row = session.scalar(
                select(InventorySnapshot)
                .where(InventorySnapshot.tenant == self.tenant)
                .where(InventorySnapshot.unique_id == str(unique_id))
                .where(InventorySnapshot.as_of <= pd.Timestamp(at_origin).to_pydatetime())
                .order_by(InventorySnapshot.as_of.desc())
                .limit(1)
            )
            if row is None:
                raise KeyError(
                    f"No inventory snapshot for tenant={self.tenant!r}, "
                    f"unique_id={unique_id!r} at or before {at_origin}"
                )
            return ProductState(
                unique_id=str(row.unique_id),
                end_inventory=float(row.end_inventory),
                pipeline=make_pipeline(
                    [float(value) for value in row.pipeline],
                    row.lead_time_depth,
                ),
                cumulative_costs={
                    str(key): float(value) for key, value in dict(row.cumulative_costs).items()
                },
            )

    def load_lead_times(self) -> dict[str, int]:
        with session_scope(self.factory) as session:
            rows = session.scalars(
                select(InventorySnapshot)
                .where(InventorySnapshot.tenant == self.tenant)
                .order_by(InventorySnapshot.unique_id, InventorySnapshot.as_of)
            )
            latest: dict[str, int] = {}
            for row in rows:
                latest[str(row.unique_id)] = int(row.lead_time_depth)
            return latest


class SqlSalesAdapter:
    """Thin sales history loader for SQL rows or parquet snapshots."""

    def __init__(
        self,
        factory: sessionmaker[Session] | None = None,
        *,
        tenant: str | None = None,
        source: str | Path | None = None,
    ) -> None:
        self.factory = factory
        self.tenant = tenant
        self.source = source

    def load_history(self, source: str | Path | None = None) -> pd.DataFrame:
        uri = source if source is not None else self.source
        if uri is not None:
            return _normalize_sales_frame(read_parquet(uri))
        if self.factory is None or self.tenant is None:
            raise ValueError("SQL sales loading requires both factory and tenant")
        with session_scope(self.factory) as session:
            rows = session.scalars(
                select(SalesRecord)
                .where(SalesRecord.tenant == self.tenant)
                .order_by(SalesRecord.unique_id, SalesRecord.ds)
            )
            frame = pd.DataFrame(
                [
                    {
                        UNIQUE_ID: row.unique_id,
                        DS: pd.Timestamp(row.ds),
                        Y: float(row.y),
                        **dict(row.payload),
                    }
                    for row in rows
                ]
            )
        return _normalize_sales_frame(frame)


class OrderRepo:
    """Persistence helper for placed order frames."""

    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    def append_frame(self, *, tenant: str, session_id: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        with session_scope(self.factory) as session:
            for payload in _records_from_frame(frame):
                unique_id = str(payload.get(UNIQUE_ID, ""))
                if not unique_id:
                    raise ValueError(f"order frame row missing {UNIQUE_ID!r}")
                order_qty = payload.get("order_qty")
                if order_qty is None:
                    raise ValueError("order frame row missing 'order_qty'")
                session.add(
                    OrderRecord(
                        tenant=tenant,
                        session_id=session_id,
                        unique_id=unique_id,
                        forecast_origin=_optional_timestamp(payload.get(FORECAST_ORIGIN)),
                        ds=_optional_timestamp(payload.get(DS)),
                        order_qty=float(order_qty),
                        payload=payload,
                    )
                )

    def list_for_session(
        self,
        session_id: str,
        *,
        tenant: str | None = None,
        unique_id: str | None = None,
    ) -> pd.DataFrame:
        with session_scope(self.factory) as session:
            statement = select(OrderRecord).where(OrderRecord.session_id == session_id)
            if tenant is not None:
                statement = statement.where(OrderRecord.tenant == tenant)
            if unique_id is not None:
                statement = statement.where(OrderRecord.unique_id == unique_id)
            rows = session.scalars(statement.order_by(OrderRecord.placed_at, OrderRecord.order_id))
            records = [dict(row.payload) for row in rows]
        frame = pd.DataFrame(records)
        for column in (DS, FORECAST_ORIGIN):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        return frame


def _normalize_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {UNIQUE_ID, DS, Y} - set(frame.columns)
    if missing:
        raise ValueError(f"sales history missing columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized[UNIQUE_ID] = normalized[UNIQUE_ID].astype(str)
    normalized[DS] = pd.to_datetime(normalized[DS])
    normalized[Y] = normalized[Y].astype(float)
    return normalized.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.copy()
    for column in clean.columns:
        clean[column] = clean[column].map(_json_safe)
    clean = clean.astype(object)
    clean[pd.isna(clean)] = None
    return cast(list[dict[str, Any]], clean.to_dict(orient="records"))


def _json_safe(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _optional_timestamp(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).to_pydatetime()
