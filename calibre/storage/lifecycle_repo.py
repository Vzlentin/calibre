"""SQL-backed lifecycle store (roadmap P0.2).

Persists fit/tune records and session-owned conformal state so the API survives
restarts and multi-worker deployments. Per FIX #1, data-plane frames
(history, future_x, last_forecast/calibrated/orders) are written as parquet to
the object store and referenced by URI — never stored inline on the fit row.
Per FIX #2, all SQL row mapping lives here in the storage layer; the typed
records and the store contract stay at the API boundary
(``calibre.api.lifecycle``).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from calibre.api.lifecycle import FitRecord, TuneRecord
from calibre.conformal.runtime import to_json_safe_state
from calibre.core.run_status import RunStatus
from calibre.execution.io import join_uri, read_parquet, write_parquet
from calibre.storage.models import (
    LifecycleConformalState,
    LifecycleFitRecord,
    LifecycleTuneRecord,
)
from calibre.storage.objstore import artifact_base_uri
from calibre.storage.postgres import session_scope

# Frame-bearing attributes on FitRecord, persisted as parquet by reference.
_FRAME_ATTRS = ("history", "future_x", "last_forecast", "last_calibrated", "last_orders")
_FIT_SCALAR_FIELDS = ("status", "error", "artifact_urls")
_STUDY_SCALAR_FIELDS = ("status", "error", "best_candidates")


class SqlLifecycleStore:
    """SQLAlchemy + object-store implementation of ``LifecycleStoreProtocol``."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        artifact_base: str | None = None,
    ) -> None:
        self._factory = session_factory
        self._artifact_base = artifact_base or artifact_base_uri()

    @staticmethod
    def new_fit_id() -> str:
        return uuid4().hex

    @staticmethod
    def new_study_id() -> str:
        return uuid4().hex

    # --- frame persistence -------------------------------------------------

    def _frame_uri(self, fit_id: str, kind: str) -> str:
        return join_uri(self._artifact_base, fit_id, f"{kind}.parquet")

    def _write_frames(self, fit_id: str, frames: dict[str, pd.DataFrame | None]) -> dict[str, str]:
        written: dict[str, str] = {}
        for kind, frame in frames.items():
            if frame is None or frame.empty:
                continue
            uri = self._frame_uri(fit_id, kind)
            write_parquet(frame, uri)
            written[kind] = uri
        return written

    def _read_frames(self, frame_uris: dict[str, str]) -> dict[str, pd.DataFrame]:
        return {kind: read_parquet(uri) for kind, uri in frame_uris.items()}

    # --- fit ---------------------------------------------------------------

    def put_fit(self, record: FitRecord) -> None:
        frame_uris = self._write_frames(
            record.fit_id, {attr: getattr(record, attr) for attr in _FRAME_ATTRS}
        )
        with session_scope(self._factory) as session:
            session.add(
                LifecycleFitRecord(
                    fit_id=record.fit_id,
                    session_id=record.session_id,
                    tenant=record.tenant,
                    sku_set=list(record.sku_set),
                    forecaster_config=dict(record.forecaster_config),
                    horizon=int(record.horizon),
                    freq=record.freq,
                    conformal_config=dict(record.conformal_config)
                    if record.conformal_config
                    else None,
                    status=RunStatus(record.status).value,
                    error=record.error,
                    artifact_urls=dict(record.artifact_urls),
                    frame_uris=frame_uris,
                )
            )

    def get_fit(self, fit_id: str) -> FitRecord | None:
        with session_scope(self._factory) as session:
            row = session.get(LifecycleFitRecord, fit_id)
            if row is None:
                return None
            return self._row_to_fit(row)

    def update_fit(self, fit_id: str, **fields: object) -> FitRecord:
        frame_updates = {attr: fields.pop(attr) for attr in _FRAME_ATTRS if attr in fields}
        written = self._write_frames(fit_id, frame_updates)  # type: ignore[arg-type]
        unknown = set(fields) - set(_FIT_SCALAR_FIELDS)
        if unknown:
            raise ValueError(f"update_fit got unsupported fields: {sorted(unknown)}")
        with session_scope(self._factory) as session:
            row = session.get(LifecycleFitRecord, fit_id)
            if row is None:
                raise KeyError(fit_id)
            if "status" in fields:
                row.status = RunStatus(cast("Any", fields["status"])).value
            if "error" in fields:
                row.error = cast("str | None", fields["error"])
            if "artifact_urls" in fields:
                row.artifact_urls = dict(cast("dict[str, Any]", fields["artifact_urls"]))
            if written:
                row.frame_uris = {**row.frame_uris, **written}
            session.flush()
            return self._row_to_fit(row)

    def fits_for_session(self, session_id: str) -> list[FitRecord]:
        with session_scope(self._factory) as session:
            rows = session.scalars(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.session_id == session_id)
                .order_by(LifecycleFitRecord.fit_id)
            ).all()
            return [self._row_to_fit(row) for row in rows]

    def first_fit_for_session(self, session_id: str) -> FitRecord | None:
        with session_scope(self._factory) as session:
            row = session.scalars(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.session_id == session_id)
                .order_by(LifecycleFitRecord.fit_id)
                .limit(1)
            ).first()
            return self._row_to_fit(row) if row is not None else None

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]:
        with session_scope(self._factory) as session:
            rows = session.scalars(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.tenant == tenant)
                .order_by(LifecycleFitRecord.fit_id)
            ).all()
            return [self._row_to_fit(row) for row in rows if uid in row.sku_set]

    def _row_to_fit(self, row: LifecycleFitRecord) -> FitRecord:
        frames = self._read_frames(dict(row.frame_uris))
        return FitRecord(
            fit_id=row.fit_id,
            session_id=row.session_id,
            tenant=row.tenant,
            sku_set=list(row.sku_set),
            forecaster_config=dict(row.forecaster_config),
            horizon=int(row.horizon),
            freq=row.freq,
            history=frames.get("history", pd.DataFrame()),
            future_x=frames.get("future_x"),
            conformal_config=dict(row.conformal_config) if row.conformal_config else None,
            status=RunStatus(row.status),
            error=row.error,
            artifact_urls=dict(row.artifact_urls),
            last_forecast=frames.get("last_forecast"),
            last_calibrated=frames.get("last_calibrated"),
            last_orders=frames.get("last_orders"),
        )

    # --- conformal state (session-owned) -----------------------------------

    def get_conformal_state(self, session_id: str) -> dict[str, dict]:
        with session_scope(self._factory) as session:
            rows = session.scalars(
                select(LifecycleConformalState).where(
                    LifecycleConformalState.session_id == session_id
                )
            ).all()
            return {row.partition: dict(row.state) for row in rows}

    def upsert_conformal_state(self, session_id: str, partition_states: dict[str, dict]) -> None:
        if not partition_states:
            return
        with session_scope(self._factory) as session:
            for partition, state in partition_states.items():
                # Conformal partition state carries numpy arrays/scalars; coerce
                # to JSON-safe values before it hits the JSONB column.
                safe_state = to_json_safe_state(dict(state))
                key = (session_id, str(partition))
                row = session.get(LifecycleConformalState, key)
                if row is None:
                    session.add(
                        LifecycleConformalState(
                            session_id=session_id,
                            partition=str(partition),
                            state=safe_state,
                        )
                    )
                else:
                    row.state = safe_state

    # --- tune --------------------------------------------------------------

    def put_study(self, record: TuneRecord) -> None:
        with session_scope(self._factory) as session:
            session.add(
                LifecycleTuneRecord(
                    study_id=record.study_id,
                    session_id=record.session_id,
                    tenant=record.tenant,
                    sku_set=list(record.sku_set),
                    status=RunStatus(record.status).value,
                    error=record.error,
                    best_candidates=dict(record.best_candidates),
                )
            )

    def get_study(self, study_id: str) -> TuneRecord | None:
        with session_scope(self._factory) as session:
            row = session.get(LifecycleTuneRecord, study_id)
            return self._row_to_study(row) if row is not None else None

    def update_study(self, study_id: str, **fields: object) -> TuneRecord:
        unknown = set(fields) - set(_STUDY_SCALAR_FIELDS)
        if unknown:
            raise ValueError(f"update_study got unsupported fields: {sorted(unknown)}")
        with session_scope(self._factory) as session:
            row = session.get(LifecycleTuneRecord, study_id)
            if row is None:
                raise KeyError(study_id)
            if "status" in fields:
                row.status = RunStatus(cast("Any", fields["status"])).value
            if "error" in fields:
                row.error = cast("str | None", fields["error"])
            if "best_candidates" in fields:
                row.best_candidates = dict(cast("dict[str, Any]", fields["best_candidates"]))
            session.flush()
            return self._row_to_study(row)

    def _row_to_study(self, row: LifecycleTuneRecord) -> TuneRecord:
        return TuneRecord(
            study_id=row.study_id,
            session_id=row.session_id,
            tenant=row.tenant,
            sku_set=list(row.sku_set),
            status=RunStatus(row.status),
            error=row.error,
            best_candidates=dict(row.best_candidates),
        )
