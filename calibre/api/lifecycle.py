from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from uuid import uuid4

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from calibre.conformal.runtime import to_json_safe_state
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN
from calibre.core.run_status import RunStatus
from calibre.storage.models import LifecycleFitRecord, LifecycleTuneRecord
from calibre.storage.postgres import session_scope


@dataclass
class FitRecord:
    fit_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    forecaster_config: dict
    horizon: int
    freq: str
    history: pd.DataFrame
    future_x: pd.DataFrame | None
    conformal_config: dict | None
    status: RunStatus = RunStatus.QUEUED
    error: str | None = None
    artifact_urls: dict[str, str] = field(default_factory=dict)
    last_forecast: pd.DataFrame | None = None
    last_calibrated: pd.DataFrame | None = None
    last_orders: pd.DataFrame | None = None


@dataclass
class TuneRecord:
    study_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    status: RunStatus = RunStatus.QUEUED
    error: str | None = None
    best_candidates: dict[str, dict[str, dict]] = field(default_factory=dict)


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


class LifecycleStore(Protocol):
    @staticmethod
    def new_fit_id() -> str:
        return uuid4().hex

    @staticmethod
    def new_study_id() -> str:
        return uuid4().hex

    def put_study(self, record: TuneRecord) -> None: ...

    def get_study(self, study_id: str) -> TuneRecord | None: ...

    def update_study(self, study_id: str, **fields: object) -> TuneRecord: ...

    def put_fit(self, record: FitRecord) -> None: ...

    def get_fit(self, fit_id: str) -> FitRecord | None: ...

    def update_fit(self, fit_id: str, **fields: object) -> FitRecord: ...

    def fits_for_session(self, session_id: str) -> list[FitRecord]: ...

    def first_fit_for_session(self, session_id: str) -> FitRecord | None: ...

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]: ...

    def get_conformal_state(self, session_id: str) -> dict[str, dict]: ...

    def upsert_conformal_state(
        self,
        session_id: str,
        partition_states: dict[str, dict],
    ) -> None: ...


class MemoryLifecycleStore:
    def __init__(self) -> None:
        self._fits: dict[str, FitRecord] = {}
        self._conformal_state: dict[str, dict[str, dict]] = {}
        self._studies: dict[str, TuneRecord] = {}

    @staticmethod
    def new_fit_id() -> str:
        return LifecycleStore.new_fit_id()

    @staticmethod
    def new_study_id() -> str:
        return LifecycleStore.new_study_id()

    def put_study(self, record: TuneRecord) -> None:
        self._studies[record.study_id] = record

    def get_study(self, study_id: str) -> TuneRecord | None:
        return self._studies.get(study_id)

    def update_study(self, study_id: str, **fields: object) -> TuneRecord:
        record = self._studies[study_id]
        for key, value in fields.items():
            setattr(record, key, value)
        return record

    def put_fit(self, record: FitRecord) -> None:
        self._fits[record.fit_id] = record

    def get_fit(self, fit_id: str) -> FitRecord | None:
        return self._fits.get(fit_id)

    def update_fit(self, fit_id: str, **fields: object) -> FitRecord:
        record = self._fits[fit_id]
        for key, value in fields.items():
            setattr(record, key, value)
        return record

    def fits_for_session(self, session_id: str) -> list[FitRecord]:
        return [r for r in self._fits.values() if r.session_id == session_id]

    def first_fit_for_session(self, session_id: str) -> FitRecord | None:
        for record in self._fits.values():
            if record.session_id == session_id:
                return record
        return None

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]:
        return [
            record
            for record in self._fits.values()
            if record.tenant == tenant and uid in record.sku_set
        ]

    def get_conformal_state(self, session_id: str) -> dict[str, dict]:
        return dict(self._conformal_state.get(session_id, {}))

    def upsert_conformal_state(
        self,
        session_id: str,
        partition_states: dict[str, dict],
    ) -> None:
        if not partition_states:
            return
        store = self._conformal_state.setdefault(session_id, {})
        for partition, state in partition_states.items():
            store[str(partition)] = to_json_safe_state(dict(state))


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
            values = _fit_row_values(record, conformal_state={})
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
            row = session.scalar(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.session_id == session_id)
                .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
                .limit(1)
            )
            if row is None:
                return {}
            return dict(row.conformal_state or {})

    def upsert_conformal_state(
        self,
        session_id: str,
        partition_states: dict[str, dict],
    ) -> None:
        if not partition_states:
            return
        with session_scope(self.factory) as session:
            rows = _fits_for_session(session, session_id)
            for row in rows:
                state = dict(row.conformal_state or {})
                for partition, partition_state in partition_states.items():
                    state[str(partition)] = to_json_safe_state(dict(partition_state))
                row.conformal_state = state


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


def _fit_row_values(record: FitRecord, *, conformal_state: dict[str, dict]) -> dict[str, object]:
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
        "conformal_state": dict(conformal_state),
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
