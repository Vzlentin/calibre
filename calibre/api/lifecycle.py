from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol
from uuid import uuid4

import pandas as pd

from calibre.core.forecast_frame import FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID
from calibre.core.run_status import RunStatus
from calibre.core.serialization import frame_from_records, json_safe_records


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


class OrderKey(NamedTuple):
    """Decision-tuple primary key for an order row, shared by both stores.

    Mirrors the ``orders`` table PK ``(session_id, unique_id, forecast_origin,
    model_name)``; ``model_name`` is optional (defaults to ``""``)."""

    session_id: str
    unique_id: str
    forecast_origin: str
    model_name: str


def order_pk(session_id: str, record: dict) -> OrderKey:
    """Build the ``OrderKey`` for a JSON order row.

    Fails fast when ``unique_id`` or ``forecast_origin`` is missing/empty so the
    in-memory and SQL stores reject the same malformed input identically rather
    than silently keying a row on ``""``. ``model_name`` stays optional."""
    unique_id = str(record.get(UNIQUE_ID, ""))
    forecast_origin = str(record.get(FORECAST_ORIGIN, ""))
    if not unique_id or not forecast_origin:
        raise ValueError("order row must include unique_id and forecast_origin")
    return OrderKey(
        session_id=session_id,
        unique_id=unique_id,
        forecast_origin=forecast_origin,
        model_name=str(record.get(MODEL_NAME, "") or ""),
    )


@dataclass
class TuneRecord:
    study_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    status: RunStatus = RunStatus.QUEUED
    error: str | None = None
    best_candidates: dict[str, dict[str, dict]] = field(default_factory=dict)


class LifecycleStore:
    def __init__(self) -> None:
        self._fits: dict[str, FitRecord] = {}
        self._conformal_state: dict[str, dict[str, dict]] = {}
        self._studies: dict[str, TuneRecord] = {}
        # Order rows keyed by the OrderKey decision tuple (session_id,
        # unique_id, forecast_origin, model_name); value carries the owning
        # tenant + the JSON-safe row.
        self._orders: dict[OrderKey, dict] = {}

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

    def last_fit_for_session(self, session_id: str) -> FitRecord | None:
        """The session's most recent fit — the canonical selection everywhere.

        Last by insertion order (dicts preserve it), matching the SQL store's
        ``created_at desc, fit_id desc``. Calibration writes ``last_calibrated``
        onto this fit and ``/sessions`` reads it, so write and read agree; a
        re-fit creates a fresh last fit whose ``last_calibrated`` is None until
        the client calibrates it.
        """
        latest: FitRecord | None = None
        for record in self._fits.values():
            if record.session_id == session_id:
                latest = record
        return latest

    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]:
        return [
            record
            for record in self._fits.values()
            if record.tenant == tenant and uid in record.sku_set
        ]

    def put_orders(self, tenant: str, session_id: str, orders: pd.DataFrame) -> None:
        for record in json_safe_records(orders):
            self._orders[order_pk(session_id, record)] = {"tenant": tenant, "detail": record}

    def orders_for_tenant_uid(self, tenant: str, uid: str) -> pd.DataFrame:
        rows = [
            entry["detail"]
            for entry in self._orders.values()
            if entry["tenant"] == tenant and str(entry["detail"].get(UNIQUE_ID, "")) == uid
        ]
        return frame_from_records(rows)

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


class LifecycleStoreProtocol(Protocol):
    """Contract shared by the in-memory and SQL-backed lifecycle stores.

    The store owns fit/tune records and session-owned conformal state. The
    concrete ``LifecycleStore`` (above) is the in-memory implementation;
    ``calibre.storage.lifecycle_repo.SqlLifecycleStore`` is the persistent one
    selected via ``LIFECYCLE_STORE=sql``.
    """

    def put_fit(self, record: FitRecord) -> None: ...
    def get_fit(self, fit_id: str) -> FitRecord | None: ...
    def update_fit(self, fit_id: str, **fields: object) -> FitRecord: ...
    def fits_for_session(self, session_id: str) -> list[FitRecord]: ...
    def last_fit_for_session(self, session_id: str) -> FitRecord | None: ...
    def fits_for_tenant_uid(self, tenant: str, uid: str) -> list[FitRecord]: ...
    def put_orders(self, tenant: str, session_id: str, orders: pd.DataFrame) -> None: ...
    def orders_for_tenant_uid(self, tenant: str, uid: str) -> pd.DataFrame: ...
    def get_conformal_state(self, session_id: str) -> dict[str, dict]: ...
    def upsert_conformal_state(
        self, session_id: str, partition_states: dict[str, dict]
    ) -> None: ...
    def put_study(self, record: TuneRecord) -> None: ...
    def get_study(self, study_id: str) -> TuneRecord | None: ...
    def update_study(self, study_id: str, **fields: object) -> TuneRecord: ...
