from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import create_app
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

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del collect_fitted_values
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
def _artifact_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CALIBRE_ARTIFACT_URI", str(tmp_path / "artifacts"))
    _StubAdapter.fit_calls = 0
    _StubAdapter.load_calls = 0


@pytest.fixture
def store():
    """The in-memory lifecycle store the test app is constructed with."""
    return LifecycleStore()


@pytest.fixture
def stub_adapter(monkeypatch):
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter", lambda cfg: _StubAdapter(cfg)
    )
    yield


@pytest.fixture
def client(store):
    return TestClient(create_app(lifecycle_store=store))


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(idx)}
        for idx, ds in enumerate(dates)
    ]


def _partition_scores(state: dict, partition: str) -> list[float]:
    """Nonconformity scores the calibrator recorded for one partition."""
    history = state[partition]["calibrator"]["score_history"][partition]
    return [float(score) for score in history]


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


def test_fit_predict_calibrate_order_observe_roundtrip(client, store, stub_adapter, sales_uri):
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
    # Origin 2024-02-04 leaves 4 history rows before it, so the stub fits
    # level=4 and predicts level + h: a known 5.0, 6.0 over the two horizons.
    assert [row[UNIQUE_ID] for row in forecast] == ["A", "A"]
    assert [row[H] for row in forecast] == [1, 2]
    assert [row[Y_HAT] for row in forecast] == [5.0, 6.0]

    calibrate_resp = client.post(
        "/calibrate",
        json={"session_id": session_id, "forecast": forecast},
    )
    assert calibrate_resp.status_code == 200, calibrate_resp.text
    calibrated = calibrate_resp.json()["calibrated"]
    lower_col, upper_col = interval_column_names(0.9)
    # ACI cold-start (no observations yet) -> zero radius -> bounds collapse onto y_hat.
    assert [row[Y_HAT] for row in calibrated] == [5.0, 6.0]
    for row in calibrated:
        assert row[lower_col] == row[Y_HAT]
        assert row[upper_col] == row[Y_HAT]

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
    order_row = orders[0]
    assert order_row[UNIQUE_ID] == "A"
    # R,S order-up-to over the protection window (lead 1 + review 1 = 2):
    # target = sum of upper bounds = 5 + 6 = 11; order = max(target - inventory(5), 0) = 6.
    assert order_row["target_stock_level"] == 11.0
    assert order_row["order_qty"] == 6.0

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

    state = store.get_conformal_state(session_id)
    # observe records one calibrator partition per resolved (model, horizon),
    # each carrying the absolute error of actual vs calibrated forecast:
    # |9 - 5| = 4 at h1, |11 - 6| = 5 at h2.
    assert set(state) == {"stub_model:h1:__global__", "stub_model:h2:__global__"}
    assert _partition_scores(state, "stub_model:h1:__global__") == [4.0]
    assert _partition_scores(state, "stub_model:h2:__global__") == [5.0]


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
    # Session state surfaces the persisted forecast and the order it produced,
    # with the same known values the round-trip computes (level=4 -> 5.0, 6.0;
    # target 11 - inventory 5 -> order 6).
    assert [row[Y_HAT] for row in body["last_forecast"]] == [5.0, 6.0]
    assert len(body["open_orders"]) == 1
    assert body["open_orders"][0]["order_qty"] == 6.0


def test_session_state_404_when_missing(client):
    response = client.get("/sessions/acme/UNKNOWN")
    assert response.status_code == 404


def test_predict_requires_succeeded_fit(client, stub_adapter, monkeypatch, sales_uri):
    monkeypatch.setattr(api_main, "_run_fit_job", lambda fit_id, *, store: None)
    fit_resp = client.post("/fit", json=_fit_payload(sales_uri))
    fit_id = fit_resp.json()["fit_id"]

    predict_resp = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"})
    assert predict_resp.status_code == 409


def test_multi_fit_selection_is_first_for_writes_last_for_reads(
    client, store, stub_adapter, sales_uri
):
    """Characterization pin (R5): with several fits in a session, /calibrate and
    /predict-driven /sessions diverge on which fit they select today. /calibrate
    writes to the FIRST (insertion-order) fit, while /sessions reads the LAST.
    This is exactly the split U4 collapses to LAST everywhere; pin today's
    behavior before the flip.

    Two identical /fit payloads share a derived session_id but mint distinct
    fit_ids -> one session, two fits in insertion order.
    """
    first_fit_id = client.post("/fit", json=_fit_payload(sales_uri)).json()["fit_id"]
    second = client.post("/fit", json=_fit_payload(sales_uri)).json()
    last_fit_id = second["fit_id"]
    session_id = second["session_id"]
    assert first_fit_id != last_fit_id

    fits = store.fits_for_session(session_id)
    assert [r.fit_id for r in fits] == [first_fit_id, last_fit_id], "insertion order"

    # Predict on each fit so both carry a distinct last_forecast frame.
    first_forecast = client.post(
        "/predict", json={"fit_id": first_fit_id, "origin": "2024-02-04"}
    ).json()["forecast"]

    calibrate = client.post(
        "/calibrate", json={"session_id": session_id, "forecast": first_forecast}
    )
    assert calibrate.status_code == 200, calibrate.text

    # /calibrate selects the FIRST fit today: its last_calibrated is populated,
    # the last fit's stays None.
    assert store.get_fit(first_fit_id).last_calibrated is not None
    assert store.get_fit(last_fit_id).last_calibrated is None

    # /sessions reads the LAST fit today; it has no forecast yet (never
    # predicted), so last_forecast is absent even though the first fit has one.
    state = client.get("/sessions/acme/A").json()
    assert state["last_forecast"] is None
    assert store.get_fit(first_fit_id).last_forecast is not None

    observe = client.post(
        "/observe",
        json={
            "session_id": session_id,
            "actuals": [
                {UNIQUE_ID: "A", "ds": "2024-02-11", "y": 9.0},
                {UNIQUE_ID: "A", "ds": "2024-02-18", "y": 11.0},
            ],
        },
    )
    assert observe.status_code == 202, observe.text
    # The observe job ran against the FIRST fit's calibrated frame -> conformal
    # state is recorded for the session.
    assert store.get_conformal_state(session_id)


def test_predict_reuses_fit_time_artifact_for_canonical_origin(client, stub_adapter, sales_uri):
    fit_resp = client.post("/fit", json=_fit_payload(sales_uri))
    assert fit_resp.status_code == 202, fit_resp.text
    fit_id = fit_resp.json()["fit_id"]

    status = client.get(f"/fits/{fit_id}").json()
    assert status["status"] == "succeeded"
    assert "model_artifact" in status["artifact_urls"]
    assert _StubAdapter.fit_calls == 1

    predict_resp = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-03-03"})

    assert predict_resp.status_code == 200, predict_resp.text
    # Canonical origin: the fit-time artifact is restored (no refit), so the
    # forecast reflects the cached level=8 (all 8 history rows): 9.0, 10.0.
    assert _StubAdapter.fit_calls == 1
    assert _StubAdapter.load_calls == 1
    forecast = predict_resp.json()["forecast"]
    assert [row[Y_HAT] for row in forecast] == [9.0, 10.0]
