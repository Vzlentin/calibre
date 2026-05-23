"""Lifecycle API boundary: typed records, the store protocol, and the in-memory store.

The SQL repository lives in :mod:`calibre.storage.lifecycle_repo` and is
re-exported here for backwards compatibility. Tests and downstream code
should prefer importing from the owning module going forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, cast
from uuid import uuid4

import pandas as pd

from calibre.conformal.runtime import to_json_safe_state
from calibre.core.run_status import RunStatus

FitFrameKind = Literal["history", "future_x", "last_forecast", "last_calibrated", "last_orders"]


@dataclass
class FitRecord:
    fit_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    forecaster_config: dict
    horizon: int
    freq: str
    conformal_config: dict | None
    history_ref: str | None = None
    future_x_ref: str | None = None
    last_forecast_ref: str | None = None
    last_calibrated_ref: str | None = None
    last_orders_ref: str | None = None
    status: RunStatus = RunStatus.QUEUED
    error: str | None = None
    artifact_urls: dict[str, str] = field(default_factory=dict)


@dataclass
class TuneRecord:
    study_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    status: RunStatus = RunStatus.QUEUED
    error: str | None = None
    best_candidates: dict[str, dict[str, dict]] = field(default_factory=dict)
    oracle_cost: float | None = None


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

    def put_fit_frame(
        self,
        fit_id: str,
        kind: FitFrameKind,
        frame: pd.DataFrame,
    ) -> str: ...

    def get_fit_frame(self, fit_id: str, kind: FitFrameKind) -> pd.DataFrame | None: ...

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
        self._fit_frames: dict[tuple[str, FitFrameKind], pd.DataFrame] = {}
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
            if key in _FIT_FRAME_REF_FIELDS:
                kind = cast(FitFrameKind, key)
                if value is None:
                    setattr(record, _FIT_FRAME_REF_FIELDS[kind], None)
                    self._fit_frames.pop((fit_id, kind), None)
                    continue
                if not isinstance(value, pd.DataFrame):
                    raise TypeError(f"{key} must be a DataFrame or None")
                self.put_fit_frame(fit_id, kind, value)
                continue
            setattr(record, key, value)
        return record

    def put_fit_frame(
        self,
        fit_id: str,
        kind: FitFrameKind,
        frame: pd.DataFrame,
    ) -> str:
        if fit_id not in self._fits:
            raise KeyError(f"Unknown fit_id: {fit_id}")
        ref = _fit_frame_ref(fit_id, kind)
        self._fit_frames[(fit_id, kind)] = frame.copy()
        setattr(self._fits[fit_id], _FIT_FRAME_REF_FIELDS[kind], ref)
        return ref

    def get_fit_frame(self, fit_id: str, kind: FitFrameKind) -> pd.DataFrame | None:
        frame = self._fit_frames.get((fit_id, kind))
        return frame.copy() if frame is not None else None

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


def __getattr__(name: str) -> object:
    if name == "SqlLifecycleStore":
        from calibre.storage.lifecycle_repo import SqlLifecycleStore

        return SqlLifecycleStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FitFrameKind",
    "FitRecord",
    "LifecycleStore",
    "MemoryLifecycleStore",
    "TuneRecord",
]


_FIT_FRAME_REF_FIELDS: dict[FitFrameKind, str] = {
    "history": "history_ref",
    "future_x": "future_x_ref",
    "last_forecast": "last_forecast_ref",
    "last_calibrated": "last_calibrated_ref",
    "last_orders": "last_orders_ref",
}


def _fit_frame_ref(fit_id: str, kind: FitFrameKind) -> str:
    return f"lifecycle://fits/{fit_id}/frames/{kind}"
