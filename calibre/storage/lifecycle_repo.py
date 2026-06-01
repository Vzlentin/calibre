"""SQL-backed lifecycle store (roadmap P0.2).

Persists fit/tune records and session-owned conformal state so the API survives
restarts and multi-worker deployments. Per FIX #1, data-plane frames
(history, future_x, last_forecast/calibrated) are written as parquet to the
object store and referenced by URI — never stored inline on the fit row. Orders
are durable rows in the ``orders`` table (issue #61), not a parquet frame. Per
FIX #2, all SQL row mapping lives here in the storage layer; the typed records
and the store contract stay at the API boundary (``calibre.api.lifecycle``).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pandas as pd
from sqlalchemy import cast as sa_cast
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, sessionmaker

from calibre.api.lifecycle import FitRecord, TuneRecord, order_pk
from calibre.api.serialization import frame_from_records, json_safe_records
from calibre.conformal.runtime import to_json_safe_state
from calibre.core.run_status import RunStatus
from calibre.execution.io import join_uri, read_parquet, write_parquet
from calibre.storage.models import (
    LifecycleConformalState,
    LifecycleFitRecord,
    LifecycleTuneRecord,
    Order,
)
from calibre.storage.objstore import artifact_base_uri
from calibre.storage.postgres import session_scope

# Frame-bearing attributes on FitRecord, persisted as parquet by reference.
_FRAME_ATTRS = ("history", "future_x", "last_forecast", "last_calibrated")
_FIT_SCALAR_FIELDS = ("status", "error", "artifact_urls")
_STUDY_SCALAR_FIELDS = ("status", "error", "best_candidates")
_ORDER_PK = ("session_id", "unique_id", "forecast_origin", "model_name")


def _conformal_upsert(dialect: str, values: dict[str, Any]) -> Any:
    """Dialect-aware INSERT ... ON CONFLICT DO UPDATE for conformal state.

    Avoids the get-then-add race on the ``(session_id, partition)`` PK under
    concurrent /observe. Only the engines this project runs on are supported.
    """
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_stmt = pg_insert(LifecycleConformalState).values(**values)
        return pg_stmt.on_conflict_do_update(
            index_elements=["session_id", "partition"],
            set_={"state": pg_stmt.excluded.state, "updated_at": func.now()},
        )
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        lite_stmt = sqlite_insert(LifecycleConformalState).values(**values)
        return lite_stmt.on_conflict_do_update(
            index_elements=["session_id", "partition"],
            set_={"state": lite_stmt.excluded.state, "updated_at": func.now()},
        )
    raise NotImplementedError(f"conformal upsert unsupported on dialect {dialect!r}")


def _order_upsert(dialect: str, values: dict[str, Any]) -> Any:
    """Dialect-aware INSERT ... ON CONFLICT DO UPDATE for an order row.

    Re-placing an order for the same decision tuple overwrites the prior row
    (qty/detail/placed_at) rather than racing get-then-add into a PK error."""
    set_columns = ("tenant", "order_qty", "detail")
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_stmt = pg_insert(Order).values(**values)
        return pg_stmt.on_conflict_do_update(
            index_elements=list(_ORDER_PK),
            set_={
                **{col: getattr(pg_stmt.excluded, col) for col in set_columns},
                "placed_at": func.now(),
            },
        )
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        lite_stmt = sqlite_insert(Order).values(**values)
        return lite_stmt.on_conflict_do_update(
            index_elements=list(_ORDER_PK),
            set_={
                **{col: getattr(lite_stmt.excluded, col) for col in set_columns},
                "placed_at": func.now(),
            },
        )
    raise NotImplementedError(f"order upsert unsupported on dialect {dialect!r}")


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
        unknown = set(fields) - set(_FIT_SCALAR_FIELDS)
        if unknown:
            raise ValueError(f"update_fit got unsupported fields: {sorted(unknown)}")
        with session_scope(self._factory) as session:
            # Lock the row so concurrent updates to the same fit (e.g. /predict
            # and /order on different workers) can't lose each other's
            # frame_uris in the read-modify-write below.
            row = session.get(LifecycleFitRecord, fit_id, with_for_update=True)
            if row is None:
                raise KeyError(fit_id)
            # Write frames only after the row is confirmed, so a failed update
            # leaves no orphan parquet.
            written = self._write_frames(
                fit_id, cast("dict[str, pd.DataFrame | None]", frame_updates)
            )
            if "status" in fields:
                row.status = RunStatus(cast("Any", fields["status"])).value
            if "error" in fields:
                row.error = cast("str | None", fields["error"])
            if "artifact_urls" in fields:
                row.artifact_urls = dict(cast("dict[str, Any]", fields["artifact_urls"]))
            if written:
                row.frame_uris = {**row.frame_uris, **written}
            session.flush()
            # Callers discard the return; skip re-reading the frames we didn't change.
            return self._row_to_fit(row, load_frames=False)

    def fits_for_session(self, session_id: str) -> list[FitRecord]:
        with session_scope(self._factory) as session:
            rows = session.scalars(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.session_id == session_id)
                .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
            ).all()
            return [self._row_to_fit(row) for row in rows]

    def first_fit_for_session(self, session_id: str) -> FitRecord | None:
        with session_scope(self._factory) as session:
            # Earliest-created fit is the canonical "first" for a session, to
            # match the in-memory store's insertion order (a session can hold
            # several fits). fit_id breaks ties deterministically.
            row = session.scalars(
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.session_id == session_id)
                .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
                .limit(1)
            ).first()
            return self._row_to_fit(row) if row is not None else None

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]:
        with session_scope(self._factory) as session:
            stmt = (
                select(LifecycleFitRecord)
                .where(LifecycleFitRecord.tenant == tenant)
                .order_by(LifecycleFitRecord.created_at, LifecycleFitRecord.fit_id)
            )
            # REVIEW #14: push the sku-set membership into SQL on Postgres
            # (``sku_set @> '["uid"]'``) so this is a predicate, not a full
            # tenant scan filtered in Python. SQLite's JSON has no containment
            # operator, so keep the post-query filter there.
            if session.get_bind().dialect.name == "postgresql":
                stmt = stmt.where(sa_cast(LifecycleFitRecord.sku_set, JSONB).contains([uid]))
                rows = session.scalars(stmt).all()
            else:
                rows = [row for row in session.scalars(stmt).all() if uid in row.sku_set]
            # Metadata only: the caller picks one fit, then loads its frames via
            # get_fit — so we don't pull every matching fit's parquet here.
            return [self._row_to_fit(row, load_frames=False) for row in rows]

    def _row_to_fit(self, row: LifecycleFitRecord, *, load_frames: bool = True) -> FitRecord:
        frames = self._read_frames(dict(row.frame_uris)) if load_frames else {}
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
        )

    # --- orders (durable ledger) -------------------------------------------

    def put_orders(self, tenant: str, session_id: str, orders: pd.DataFrame) -> None:
        records = json_safe_records(orders)
        if not records:
            return
        with session_scope(self._factory) as session:
            dialect = session.get_bind().dialect.name
            for record in records:
                _, unique_id, forecast_origin, model_name = order_pk(session_id, record)
                values = {
                    "session_id": session_id,
                    "unique_id": unique_id,
                    "forecast_origin": pd.Timestamp(forecast_origin).to_pydatetime(),
                    "model_name": model_name,
                    "tenant": tenant,
                    "order_qty": float(record.get("order_qty") or 0.0),
                    "detail": record,
                }
                session.execute(_order_upsert(dialect, values))

    def open_orders_for_tenant_uid(self, tenant: str, uid: str) -> pd.DataFrame:
        with session_scope(self._factory) as session:
            rows = session.scalars(
                select(Order)
                .where(Order.tenant == tenant, Order.unique_id == uid)
                .order_by(Order.forecast_origin, Order.model_name)
            ).all()
            return frame_from_records([dict(row.detail) for row in rows])

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
            dialect = session.get_bind().dialect.name
            for partition, state in partition_states.items():
                # Conformal partition state carries numpy arrays/scalars; coerce
                # to JSON-safe values before it hits the JSONB column.
                values = {
                    "session_id": session_id,
                    "partition": str(partition),
                    "state": to_json_safe_state(dict(state)),
                }
                # Atomic upsert so concurrent /observe jobs for the same
                # (session, partition) don't race get-then-add into a PK error.
                stmt = _conformal_upsert(dialect, values)
                session.execute(stmt)

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
