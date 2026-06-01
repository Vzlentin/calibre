from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import app
from calibre.core.forecast_frame import (
    DS,
    UNIQUE_ID,
    Y_HAT,
    H,
    interval_column_names,
)
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import ModelAdapter


class _StubAdapter(ModelAdapter):
    fit_calls = 0
    load_calls = 0

    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}
        self._level = 10.0

    def fit(self, task: ForecastTask) -> None:
        type(self).fit_calls += 1
        self._task = task
        self._level = float(len(task.history))

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        horizon = int(task.horizon)
        origin = pd.Timestamp(task.forecast_origin)
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id] * horizon,
                DS: [origin + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)],
                Y_HAT: [self._level + h for h in range(1, horizon + 1)],
                H: list(range(1, horizon + 1)),
            }
        )

    def dump_state(self) -> bytes:
        return json.dumps({"level": self._level}).encode()

    def load_state(self, blob: bytes) -> None:
        type(self).load_calls += 1
        self._level = float(json.loads(blob.decode())["level"])


@pytest.fixture(autouse=True)
def _reset_lifecycle_store(monkeypatch, tmp_path):
    fresh = LifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", fresh)
    monkeypatch.setenv("CALIBRE_ARTIFACT_URI", str(tmp_path / "artifacts"))
    _StubAdapter.fit_calls = 0
    _StubAdapter.load_calls = 0
    return fresh


@pytest.fixture
def stub_adapter(monkeypatch):
    monkeypatch.setattr("calibre.api.main.resolve_adapter", lambda _: _StubAdapter(), raising=False)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda cfg: _StubAdapter(cfg))
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(idx)}
        for idx, ds in enumerate(dates)
    ]


@pytest.fixture
def sales_uri(tmp_path):
    path = tmp_path / "sales.parquet"
    pd.DataFrame(_history_records("A")).to_parquet(path)
    return str(path)


def _fit_payload(sales_uri: str) -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "sales_uri": sales_uri,
        "forecaster_config": {"backend": "stub", "model": "stub_model"},
        "conformal_config": {
            "method": "aci",
            "coverage": 0.9,
            "calibration_window": 4,
            "gamma": 0.05,
        },
    }


def test_fit_predict_calibrate_order_observe_roundtrip(client, stub_adapter, sales_uri):
    fit_resp = client.post("/fit", json=_fit_payload(sales_uri))
    assert fit_resp.status_code == 202, fit_resp.text
    handle = fit_resp.json()
    fit_id = handle["fit_id"]
    session_id = handle["session_id"]

    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "succeeded"

    predict_resp = client.post(
        "/predict",
        json={"fit_id": fit_id, "origin": "2024-02-04"},
    )
    assert predict_resp.status_code == 200, predict_resp.text
    forecast = predict_resp.json()["forecast"]
    assert len(forecast) == 2
    assert {row[UNIQUE_ID] for row in forecast} == {"A"}

    calibrate_resp = client.post(
        "/calibrate",
        json={"session_id": session_id, "forecast": forecast},
    )
    assert calibrate_resp.status_code == 200, calibrate_resp.text
    calibrated = calibrate_resp.json()["calibrated"]
    lower_col, upper_col = interval_column_names(0.9)
    assert all(lower_col in row and upper_col in row for row in calibrated)

    order_resp = client.post(
        "/order",
        json={
            "calibrated": calibrated,
            "session_id": session_id,
            "ordering": {
                "policy": "rs",
                "coverage": 0.9,
                "params": [
                    {
                        "unique_id": "A",
                        "inventory_position": 5.0,
                        "lead_time": 1,
                        "review_period": 1,
                    }
                ],
            },
        },
    )
    assert order_resp.status_code == 200, order_resp.text
    orders = order_resp.json()["orders"]
    assert len(orders) == 1
    assert orders[0][UNIQUE_ID] == "A"
    assert "order_qty" in orders[0]

    observe_resp = client.post(
        "/observe",
        json={
            "session_id": session_id,
            "actuals": [
                {UNIQUE_ID: "A", "ds": "2024-02-11", "y": 9.0},
                {UNIQUE_ID: "A", "ds": "2024-02-18", "y": 11.0},
            ],
        },
    )
    assert observe_resp.status_code == 202, observe_resp.text
    assert observe_resp.json()["session_id"] == session_id

    state = api_main._LIFECYCLE_STORE.get_conformal_state(session_id)
    assert state, "observe should persist conformal state"


def test_session_state_get(client, stub_adapter, sales_uri):
    fit_resp = client.post("/fit", json=_fit_payload(sales_uri))
    fit_id = fit_resp.json()["fit_id"]
    session_id = fit_resp.json()["session_id"]

    predict = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"}).json()
    calibrated = client.post(
        "/calibrate",
        json={"session_id": session_id, "forecast": predict["forecast"]},
    ).json()["calibrated"]
    client.post(
        "/order",
        json={
            "calibrated": calibrated,
            "session_id": session_id,
            "ordering": {
                "policy": "rs",
                "coverage": 0.9,
                "params": [
                    {
                        "unique_id": "A",
                        "inventory_position": 5.0,
                        "lead_time": 1,
                        "review_period": 1,
                    }
                ],
            },
        },
    )

    state_resp = client.get("/sessions/acme/A")
    assert state_resp.status_code == 200, state_resp.text
    body = state_resp.json()
    assert body["session_id"] == session_id
    assert body["tenant"] == "acme"
    assert body["unique_id"] == "A"
    assert body["last_forecast"] is not None and len(body["last_forecast"]) == 2
    assert body["open_orders"] is not None and len(body["open_orders"]) == 1


def test_session_state_404_when_missing(client):
    response = client.get("/sessions/acme/UNKNOWN")
    assert response.status_code == 404


def test_predict_requires_succeeded_fit(client, stub_adapter, monkeypatch, sales_uri):
    monkeypatch.setattr(api_main, "_run_fit_job", lambda fit_id: None)
    fit_resp = client.post("/fit", json=_fit_payload(sales_uri))
    fit_id = fit_resp.json()["fit_id"]

    predict_resp = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"})
    assert predict_resp.status_code == 409


def test_predict_reuses_fit_time_artifact_for_canonical_origin(client, stub_adapter):
    fit_resp = client.post("/fit", json=_fit_payload())
    assert fit_resp.status_code == 202, fit_resp.text
    fit_id = fit_resp.json()["fit_id"]

    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "succeeded"
    assert "model_artifact" in status["artifact_urls"]
    assert _StubAdapter.fit_calls == 1

    predict_resp = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-03-03"})

    assert predict_resp.status_code == 200, predict_resp.text
    assert _StubAdapter.fit_calls == 1
    assert _StubAdapter.load_calls == 1
