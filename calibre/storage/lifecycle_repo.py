"""SQL repository for fit/tune lifecycle records and session conformal state.

Holds the SqlAlchemy row mapping that ``calibre/api/lifecycle.py`` used to
own inline. The API module keeps the lifecycle protocol, in-memory store,
and dataclass records near the request handlers; the persistence layer
lives here.

Conformal state is keyed by ``(session_id, partition)`` in its own table
rather than duplicated on every fit row (PR #38 review, FIX.md item 1).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from calibre.api.lifecycle import FitRecord, LifecycleStore, TuneRecord
from calibre.conformal.runtime import to_json_safe_state
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN
from calibre.core.run_status import RunStatus
from calibre.storage.models import (
    LifecycleConformalState,
    LifecycleFitRecord,
    LifecycleTuneRecord,
)
from calibre.storage.postgres import session_scope


def _records_from_frame(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    if frame is None:
        return None
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            clean[col] = clean[col].map(
                lambda value: value.isoformat() if isinstance(value, pd.Timestamp) else value
            )
    clean = clean.astype(object)
    clean[pd.isna(clean)] = None
    return cast(list[dict[str, Any]], clean.to_dict(orient="records"))


def _frame_from_records(records: list[dict[str, Any]] | None) -> pd.DataFrame | None:
    if records is None:
        return None
    frame = pd.DataFrame(records)
    for col in (DS, FORECAST_ORIGIN):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def _required_frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _frame_from_records(records)
    if frame is None:
        return pd.DataFrame()
    return frame


def _status_value(status: RunStatus | str) -> str:
    return RunStatus(status).value


def _status_value_from_object(value: object) -> str:
    if not isinstance(value, RunStatus | str):
        raise TypeError("status must be a RunStatus or str")
    return _status_value(value)


class SqlLifecycleStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    @staticmethod
    def new_fit_id() -> str:
        return LifecycleStore.new_fit_id()

    @staticmethod
    def new_study_id() -> str:
        return LifecycleStore.new_study_id()

    def put_study(self, record: TuneRecord) -> None:
        with session_scope(self.factory) as session:
            existing = session.get(LifecycleTuneRecord, record.study_id)
            values = _tune_row_values(record)
            if existing is None:
                session.add(LifecycleTuneRecord(**values))
            else:
                _apply_values(existing, values)

    def get_study(self, study_id: str) -> TuneRecord | None:
        with session_scope(self.factory) as session:
            row = session.get(LifecycleTuneRecord, study_id)
            return _tune_from_row(row) if row is not None else None

    def update_study(self, study_id: str, **fields: object) -> TuneRecord:
        with session_scope(self.factory) as session:
            row = session.get(LifecycleTuneRecord, study_id)
            if row is None:
                raise KeyError(f"Unknown study_id: {study_id}")
            for key, value in fields.items():
                _set_tune_field(row, key, value)
            session.flush()
            return _tune_from_row(row)

    def put_fit(self, record: FitRecord) -> None:
        with session_scope(self.factory) as session:
            existing = session.get(LifecycleFitRecord, record.fit_id)
            values = _fit_row_values(record)
            if existing is None:
                session.add(LifecycleFitRecord(**values))
            else:
                _apply_values(existing, values)

    def get_fit(self, fit_id: str) -> FitRecord | None:
        with session_scope(self.factory) as session:
            row = session.get(LifecycleFitRecord, fit_id)
            return _fit_from_row(row) if row is not None else None

    def update_fit(self, fit_id: str, **fields: object) -> FitRecord:
        with session_scope(self.factory) as session:
            row = session.get(LifecycleFitRecord, fit_id)
            if row is None:
                raise KeyError(f"Unknown fit_id: {fit_id}")
            for key, value in fields.items():
                _set_fit_field(row, key, value)
            session.flush()
            return _fit_from_row(row)

    def fits_for_session(self, session_id: str) -> list[FitRecord]:
        with session_scope(self.factory) as session:
            rows = _fits_for_session(session, session_id)
            return [_fit_from_row(row) for row in rows]

    def first_fit_for_session(self, session_id: str) -> FitRecord | None:
        with session_scope(self.factory) as session:
            row = session.scalar(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.session_id == session_id)
                .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
                .limit(1)
            )
            return _fit_from_row(row) if row is not None else None

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]:
        with session_scope(self.factory) as session:
            rows = session.scalars(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.tenant == tenant)
                .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
            )
            return [_fit_from_row(row) for row in rows if uid in row.sku_set]

    def get_conformal_state(self, session_id: str) -> dict[str, dict]:
        with session_scope(self.factory) as session:
            rows = session.scalars(
                select(LifecycleConformalState).where(
                    LifecycleConformalState.session_id == session_id
                )
            )
            return {row.partition: dict(row.state) for row in rows}

    def upsert_conformal_state(
        self,
        session_id: str,
        partition_states: dict[str, dict],
    ) -> None:
        if not partition_states:
            return
        with session_scope(self.factory) as session:
            for partition, partition_state in partition_states.items():
                partition_key = str(partition)
                row = session.get(LifecycleConformalState, (session_id, partition_key))
                state = to_json_safe_state(dict(partition_state))
                if row is None:
                    session.add(
                        LifecycleConformalState(
                            session_id=session_id,
                            partition=partition_key,
                            state=state,
                        )
                    )
                else:
                    row.state = state

    def delete_conformal_state(self, session_id: str) -> None:
        """Drop all partition state rows for ``session_id`` (test-only utility)."""
        with session_scope(self.factory) as session:
            session.execute(
                delete(LifecycleConformalState).where(
                    LifecycleConformalState.session_id == session_id
                )
            )


def _apply_values(row: object, values: dict[str, object]) -> None:
    for key, value in values.items():
        setattr(row, key, value)


def _fits_for_session(session: Session, session_id: str) -> list[LifecycleFitRecord]:
    return list(
        session.scalars(
            select(LifecycleFitRecord)
            .where(LifecycleFitRecord.session_id == session_id)
            .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
        )
    )


def _fit_row_values(record: FitRecord) -> dict[str, object]:
    return {
        "fit_id": record.fit_id,
        "session_id": record.session_id,
        "tenant": record.tenant,
        "sku_set": list(record.sku_set),
        "forecaster_config": dict(record.forecaster_config),
        "horizon": int(record.horizon),
        "freq": record.freq,
        "history": _records_from_frame(record.history) or [],
        "future_x": _records_from_frame(record.future_x),
        "conformal_config": dict(record.conformal_config) if record.conformal_config else None,
        "status": _status_value(record.status),
        "error": record.error,
        "artifact_urls": dict(record.artifact_urls),
        "last_forecast": _records_from_frame(record.last_forecast),
        "last_calibrated": _records_from_frame(record.last_calibrated),
        "last_orders": _records_from_frame(record.last_orders),
    }


def _fit_from_row(row: LifecycleFitRecord) -> FitRecord:
    return FitRecord(
        fit_id=row.fit_id,
        session_id=row.session_id,
        tenant=row.tenant,
        sku_set=[str(uid) for uid in row.sku_set],
        forecaster_config=dict(row.forecaster_config),
        horizon=int(row.horizon),
        freq=row.freq,
        history=_required_frame_from_records(row.history),
        future_x=_frame_from_records(row.future_x),
        conformal_config=dict(row.conformal_config) if row.conformal_config else None,
        status=RunStatus(row.status),
        error=row.error,
        artifact_urls={str(key): str(value) for key, value in dict(row.artifact_urls).items()},
        last_forecast=_frame_from_records(row.last_forecast),
        last_calibrated=_frame_from_records(row.last_calibrated),
        last_orders=_frame_from_records(row.last_orders),
    )


def _set_fit_field(row: LifecycleFitRecord, key: str, value: object) -> None:
    if key in {"history", "future_x", "last_forecast", "last_calibrated", "last_orders"}:
        if value is not None and not isinstance(value, pd.DataFrame):
            raise TypeError(f"{key} must be a DataFrame or None")
        setattr(row, key, _records_from_frame(value))
        return
    if key == "status":
        row.status = _status_value_from_object(value)
        return
    if key in {"forecaster_config", "artifact_urls"}:
        setattr(row, key, dict(_expect_mapping(key, value)))
        return
    if key == "conformal_config":
        row.conformal_config = dict(_expect_mapping(key, value)) if value is not None else None
        return
    if key == "sku_set":
        row.sku_set = [str(uid) for uid in _expect_iterable(key, value)]
        return
    if not hasattr(row, key):
        raise AttributeError(f"Unknown FitRecord field: {key}")
    setattr(row, key, value)


def _tune_row_values(record: TuneRecord) -> dict[str, object]:
    return {
        "study_id": record.study_id,
        "session_id": record.session_id,
        "tenant": record.tenant,
        "sku_set": list(record.sku_set),
        "status": _status_value(record.status),
        "error": record.error,
        "best_candidates": dict(record.best_candidates),
        "oracle_cost": record.oracle_cost,
    }


def _tune_from_row(row: LifecycleTuneRecord) -> TuneRecord:
    return TuneRecord(
        study_id=row.study_id,
        session_id=row.session_id,
        tenant=row.tenant,
        sku_set=[str(uid) for uid in row.sku_set],
        status=RunStatus(row.status),
        error=row.error,
        best_candidates=dict(row.best_candidates),
        oracle_cost=row.oracle_cost,
    )


def _set_tune_field(row: LifecycleTuneRecord, key: str, value: object) -> None:
    if key == "status":
        row.status = _status_value_from_object(value)
        return
    if key == "best_candidates":
        row.best_candidates = dict(_expect_mapping(key, value))
        return
    if key == "sku_set":
        row.sku_set = [str(uid) for uid in _expect_iterable(key, value)]
        return
    if not hasattr(row, key):
        raise AttributeError(f"Unknown TuneRecord field: {key}")
    setattr(row, key, value)


def _expect_mapping(key: str, value: object) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dict")
    return value


def _expect_iterable(key: str, value: object) -> Iterable[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise TypeError(f"{key} must be an iterable")
    return value


__all__ = ["SqlLifecycleStore"]
