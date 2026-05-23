from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from calibre.conformal.runtime import to_json_safe_state
from calibre.core.run_status import RunStatus
from calibre.storage.models import Base, FitRecordRow, TuneRecordRow


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


def _df_to_json(df: pd.DataFrame | None) -> list[dict] | None:
    if df is None:
        return None
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _json_to_df(data: list[dict] | dict | None) -> pd.DataFrame | None:
    import io

    if data is None:
        return None
    if not data:
        return pd.DataFrame()
    records = list(data.values()) if isinstance(data, dict) else data
    df = pd.read_json(io.StringIO(json.dumps(records)), orient="records")
    for col in ("ds", "forecast_origin"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _coerce_df(data: list[dict] | dict | None) -> pd.DataFrame:
    result = _json_to_df(data)
    return result if result is not None else pd.DataFrame()


def _row_to_fit_record(row: FitRecordRow) -> FitRecord:
    return FitRecord(
        fit_id=row.fit_id,
        session_id=row.session_id,
        tenant=row.tenant,
        sku_set=list(row.sku_set),
        forecaster_config=dict(row.forecaster_config),
        horizon=int(row.horizon),
        freq=str(row.freq),
        history=_coerce_df(row.history),
        future_x=_json_to_df(row.future_x),
        conformal_config=dict(row.conformal_config) if row.conformal_config else None,
        status=RunStatus(row.status),
        error=row.error,
        artifact_urls=dict(row.artifact_urls) if row.artifact_urls else {},
        last_forecast=_json_to_df(row.last_forecast),
        last_calibrated=_json_to_df(row.last_calibrated),
        last_orders=_json_to_df(row.last_orders),
    )


def _row_to_tune_record(row: TuneRecordRow) -> TuneRecord:
    return TuneRecord(
        study_id=row.study_id,
        session_id=row.session_id,
        tenant=row.tenant,
        sku_set=list(row.sku_set),
        status=RunStatus(row.status),
        error=row.error,
        best_candidates=dict(row.best_candidates) if row.best_candidates else {},
    )


@runtime_checkable
class AnyLifecycleStore(Protocol):
    @staticmethod
    def new_fit_id() -> str: ...

    @staticmethod
    def new_study_id() -> str: ...

    def put_fit(self, record: FitRecord) -> None: ...

    def get_fit(self, fit_id: str) -> FitRecord | None: ...

    def update_fit(self, fit_id: str, **fields: Any) -> FitRecord: ...

    def fits_for_session(self, session_id: str) -> list[FitRecord]: ...

    def first_fit_for_session(self, session_id: str) -> FitRecord | None: ...

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]: ...

    def put_study(self, record: TuneRecord) -> None: ...

    def get_study(self, study_id: str) -> TuneRecord | None: ...

    def update_study(self, study_id: str, **fields: Any) -> TuneRecord: ...

    def get_conformal_state(self, session_id: str) -> dict[str, dict]: ...

    def upsert_conformal_state(
        self, session_id: str, partition_states: dict[str, dict]
    ) -> None: ...


class MemoryLifecycleStore:
    def __init__(self) -> None:
        self._fits: dict[str, FitRecord] = {}
        self._conformal_state: dict[str, dict[str, dict]] = {}
        self._studies: dict[str, TuneRecord] = {}

    @staticmethod
    def new_fit_id() -> str:
        return uuid4().hex

    @staticmethod
    def new_study_id() -> str:
        return uuid4().hex

    def put_study(self, record: TuneRecord) -> None:
        self._studies[record.study_id] = record

    def get_study(self, study_id: str) -> TuneRecord | None:
        return self._studies.get(study_id)

    def update_study(self, study_id: str, **fields: Any) -> TuneRecord:
        record = self._studies[study_id]
        for key, value in fields.items():
            setattr(record, key, value)
        return record

    def put_fit(self, record: FitRecord) -> None:
        self._fits[record.fit_id] = record

    def get_fit(self, fit_id: str) -> FitRecord | None:
        return self._fits.get(fit_id)

    def update_fit(self, fit_id: str, **fields: Any) -> FitRecord:
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
            store[str(partition)] = dict(state)


class SqlLifecycleStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    @staticmethod
    def new_fit_id() -> str:
        return uuid4().hex

    @staticmethod
    def new_study_id() -> str:
        return uuid4().hex

    def _session(self) -> Session:
        return self._factory()

    def put_fit(self, record: FitRecord) -> None:
        with self._session() as session:
            row = FitRecordRow(
                fit_id=record.fit_id,
                session_id=record.session_id,
                tenant=record.tenant,
                sku_set=list(record.sku_set),
                forecaster_config=dict(record.forecaster_config),
                horizon=int(record.horizon),
                freq=record.freq,
                history=_df_to_json(record.history),
                future_x=_df_to_json(record.future_x),
                conformal_config=dict(record.conformal_config) if record.conformal_config else None,
                status=record.status.value,
                error=record.error,
                artifact_urls=dict(record.artifact_urls),
                last_forecast=_df_to_json(record.last_forecast),
                last_calibrated=_df_to_json(record.last_calibrated),
                last_orders=_df_to_json(record.last_orders),
                conformal_state=None,
            )
            session.add(row)
            session.commit()

    def get_fit(self, fit_id: str) -> FitRecord | None:
        with self._session() as session:
            row = session.get(FitRecordRow, fit_id)
            if row is None:
                return None
            return _row_to_fit_record(row)

    def update_fit(self, fit_id: str, **fields: Any) -> FitRecord:
        with self._session() as session:
            row = session.get(FitRecordRow, fit_id)
            if row is None:
                raise KeyError(f"Unknown fit_id: {fit_id}")
            df_fields = {"history", "future_x", "last_forecast", "last_calibrated", "last_orders"}
            for key, value in fields.items():
                if key in df_fields:
                    setattr(row, key, _df_to_json(value))
                elif key == "status" and isinstance(value, RunStatus):
                    row.status = value.value
                else:
                    setattr(row, key, value)
            session.commit()
            return _row_to_fit_record(row)

    def fits_for_session(self, session_id: str) -> list[FitRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(FitRecordRow).where(FitRecordRow.session_id == session_id)
            )
            return [_row_to_fit_record(r) for r in rows]

    def first_fit_for_session(self, session_id: str) -> FitRecord | None:
        with self._session() as session:
            row = session.scalar(
                select(FitRecordRow).where(FitRecordRow.session_id == session_id).limit(1)
            )
            if row is None:
                return None
            return _row_to_fit_record(row)

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]:
        with self._session() as session:
            rows = session.scalars(select(FitRecordRow).where(FitRecordRow.tenant == tenant))
            return [_row_to_fit_record(r) for r in rows if uid in list(r.sku_set)]

    def put_study(self, record: TuneRecord) -> None:
        with self._session() as session:
            row = TuneRecordRow(
                study_id=record.study_id,
                session_id=record.session_id,
                tenant=record.tenant,
                sku_set=list(record.sku_set),
                status=record.status.value,
                error=record.error,
                best_candidates=dict(record.best_candidates),
            )
            session.add(row)
            session.commit()

    def get_study(self, study_id: str) -> TuneRecord | None:
        with self._session() as session:
            row = session.get(TuneRecordRow, study_id)
            if row is None:
                return None
            return _row_to_tune_record(row)

    def update_study(self, study_id: str, **fields: Any) -> TuneRecord:
        with self._session() as session:
            row = session.get(TuneRecordRow, study_id)
            if row is None:
                raise KeyError(f"Unknown study_id: {study_id}")
            for key, value in fields.items():
                if key == "status" and isinstance(value, RunStatus):
                    row.status = value.value
                else:
                    setattr(row, key, value)
            session.commit()
            return _row_to_tune_record(row)

    def get_conformal_state(self, session_id: str) -> dict[str, dict]:
        with self._session() as session:
            row = session.scalar(
                select(FitRecordRow).where(FitRecordRow.session_id == session_id).limit(1)
            )
            if row is None or row.conformal_state is None:
                return {}
            return dict(row.conformal_state)

    def upsert_conformal_state(
        self,
        session_id: str,
        partition_states: dict[str, dict],
    ) -> None:
        if not partition_states:
            return
        with self._session() as session:
            row = session.scalar(
                select(FitRecordRow).where(FitRecordRow.session_id == session_id).limit(1)
            )
            if row is None:
                return
            existing = dict(row.conformal_state) if row.conformal_state else {}
            for partition, state in partition_states.items():
                existing[str(partition)] = to_json_safe_state(dict(state))
            row.conformal_state = existing
            session.commit()


def LifecycleStore() -> MemoryLifecycleStore | SqlLifecycleStore:
    """Factory: returns MemoryLifecycleStore (default) or SqlLifecycleStore.

    Controlled by LIFECYCLE_STORE env var: 'memory' (default) | 'sql'.
    When 'sql', reads CALIBRE_DATABASE_URL (falls back to sqlite:///:memory:).
    SQLite in-memory databases use StaticPool so the same DB is shared
    across sessions (required for :memory: to persist between transactions).
    """
    if os.environ.get("LIFECYCLE_STORE") == "sql":
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool

        from calibre.storage.postgres import make_session_factory

        url = os.environ.get("CALIBRE_DATABASE_URL", "sqlite:///:memory:")
        if ":memory:" in url:
            engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            from calibre.storage.postgres import make_engine

            engine = make_engine(url)
        Base.metadata.create_all(engine)
        factory = make_session_factory(engine)
        return SqlLifecycleStore(factory)
    return MemoryLifecycleStore()
