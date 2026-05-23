from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import app
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H
from calibre.core.forecast_task import ForecastTask


class _StubAdapter:
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}
        self.fit_called = False

    def fit(self, task: ForecastTask) -> None:
        self.fit_called = True
        self._task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        origin = pd.Timestamp(task.forecast_origin)
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id] * task.horizon,
                DS: [origin + pd.Timedelta(weeks=h) for h in range(1, task.horizon + 1)],
                Y_HAT: [10.0 + h for h in range(1, task.horizon + 1)],
                H: list(range(1, task.horizon + 1)),
            }
        )


_last_stub: _StubAdapter | None = None


def _tracking_factory(cfg: dict) -> _StubAdapter:
    global _last_stub
    _last_stub = _StubAdapter(cfg)
    return _last_stub


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    global _last_stub
    _last_stub = None
    store = LifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", store)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", _tracking_factory)
    return store


@pytest.fixture
def client():
    return TestClient(app)


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(i)} for i, ds in enumerate(dates)
    ]


def _fit_payload(backend: str = "stub") -> dict:
    return {
        "tenant": "t",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records("A"),
        "forecaster_config": {"backend": backend, "model": "m"},
    }


def test_fit_actually_trains_model(client) -> None:
    resp = client.post("/fit", json=_fit_payload())
    assert resp.status_code == 202, resp.text
    fit_id = resp.json()["fit_id"]

    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "succeeded"
    assert _last_stub is not None and _last_stub.fit_called


def test_fit_fails_on_missing_backend_key(client) -> None:
    payload = {
        "tenant": "t",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records("A"),
        "forecaster_config": {},
    }
    resp = client.post("/fit", json=payload)
    assert resp.status_code == 202, resp.text
    fit_id = resp.json()["fit_id"]

    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "failed"
    assert status["error"] is not None
    assert "backend" in status["error"]


def test_fit_fails_on_invalid_horizon(client) -> None:
    payload = {**_fit_payload(), "horizon": 0}
    resp = client.post("/fit", json=payload)
    assert resp.status_code == 202, resp.text
    fit_id = resp.json()["fit_id"]
    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "failed"


def test_fit_artifact_stored_after_fit(client) -> None:
    resp = client.post("/fit", json=_fit_payload())
    assert resp.status_code == 202, resp.text
    fit_id = resp.json()["fit_id"]

    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "succeeded"
    assert status["artifact_urls"], "expected artifact_urls to be populated after fit"
