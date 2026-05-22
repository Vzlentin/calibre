from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pandas as pd

from calibre.core.run_status import RunStatus


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


class LifecycleStore:
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
            store[str(partition)] = dict(state)
