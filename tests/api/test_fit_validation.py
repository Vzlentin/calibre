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
from calibre.core.forecast_frame import UNIQUE_ID, Y_HAT, H


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", LifecycleStore())
    monkeypatch.setenv("CALIBRE_ARTIFACT_URI", str(tmp_path / "artifacts"))


@pytest.fixture
def client():
    return TestClient(app)


def _history(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": d.strftime("%Y-%m-%d"), "y": float(i + 1)}
        for i, d in enumerate(dates)
    ]


@pytest.fixture
def sales_uri(tmp_path):
    # Only SKU A has sales history; B is intentionally absent.
    path = tmp_path / "sales.parquet"
    pd.DataFrame(_history("A")).to_parquet(path)
    return str(path)


def _payload(sales_uri: str, **overrides) -> dict:
    base = {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "sales_uri": sales_uri,
        "forecaster_config": {"backend": "statsforecast", "model": "Naive"},
    }
    base.update(overrides)
    return base


def _status(client: TestClient, fit_id: str) -> dict:
    return client.get(f"/fits/{fit_id}").json()


def test_fit_succeeds_for_compatible_config(client, sales_uri):
    fit_id = client.post("/fit", json=_payload(sales_uri)).json()["fit_id"]
    assert _status(client, fit_id)["status"] == "succeeded"


def test_fit_fails_for_unknown_model(client, sales_uri):
    payload = _payload(
        sales_uri, forecaster_config={"backend": "statsforecast", "model": "NotARealModel"}
    )
    fit_id = client.post("/fit", json=payload).json()["fit_id"]
    body = _status(client, fit_id)
    assert body["status"] == "failed"
    assert "NotARealModel" in (body["error"] or "")


def test_fit_fails_when_a_sku_has_no_history(client, sales_uri):
    # sku_set names B, but sales history only contains A.
    fit_id = client.post("/fit", json=_payload(sales_uri, sku_set=["A", "B"])).json()["fit_id"]
    body = _status(client, fit_id)
    assert body["status"] == "failed"
    assert "B" in (body["error"] or "")


def test_predict_works_after_valid_fit(client, sales_uri):
    fit_id = client.post("/fit", json=_payload(sales_uri)).json()["fit_id"]
    assert _status(client, fit_id)["status"] == "succeeded"
    predict = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"})
    assert predict.status_code == 200, predict.text
    forecast = predict.json()["forecast"]
    # Naive carries the last observed value (y=4 at 2024-01-28) across both horizons.
    assert [row[H] for row in forecast] == [1, 2]
    assert [row[Y_HAT] for row in forecast] == [4.0, 4.0]
