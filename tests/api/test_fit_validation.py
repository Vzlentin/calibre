"""/fit eagerly validates config compatibility (roadmap P0.3).

Before this, ``_run_fit_job`` flipped to SUCCEEDED without fitting, so a config
incompatible with the data only failed lazily at /predict. These assert the
production path: a compatible config succeeds, an incompatible one FAILS with a
descriptive error at /fit, and /predict still works after a valid fit.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import app
from calibre.core.forecast_frame import UNIQUE_ID


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", LifecycleStore())


@pytest.fixture
def client():
    return TestClient(app)


def _history(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": d.strftime("%Y-%m-%d"), "y": float(i + 1)}
        for i, d in enumerate(dates)
    ]


def _payload(**overrides) -> dict:
    base = {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history("A"),
        "forecaster_config": {"backend": "statsforecast", "model": "Naive"},
    }
    base.update(overrides)
    return base


def _status(client: TestClient, fit_id: str) -> dict:
    return client.get(f"/fits/{fit_id}").json()


def test_fit_succeeds_for_compatible_config(client):
    fit_id = client.post("/fit", json=_payload()).json()["fit_id"]
    assert _status(client, fit_id)["status"] == "succeeded"


def test_fit_fails_for_unknown_model(client):
    payload = _payload(forecaster_config={"backend": "statsforecast", "model": "NotARealModel"})
    fit_id = client.post("/fit", json=payload).json()["fit_id"]
    body = _status(client, fit_id)
    assert body["status"] == "failed"
    assert "NotARealModel" in (body["error"] or "")


def test_fit_fails_when_a_sku_has_no_history(client):
    # sku_set names B, but history only contains A.
    fit_id = client.post("/fit", json=_payload(sku_set=["A", "B"])).json()["fit_id"]
    body = _status(client, fit_id)
    assert body["status"] == "failed"
    assert "B" in (body["error"] or "")


def test_predict_works_after_valid_fit(client):
    fit_id = client.post("/fit", json=_payload()).json()["fit_id"]
    assert _status(client, fit_id)["status"] == "succeeded"
    predict = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"})
    assert predict.status_code == 200, predict.text
    assert len(predict.json()["forecast"]) == 2
