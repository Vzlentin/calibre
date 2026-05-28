"""Deploy smoke gate.

Proves the service is *deployable*: the schema applies via ``alembic upgrade
head`` (not ``create_all()``), the app boots wired to that migrated database,
and the core lifecycle — ``/fit`` → ``/predict`` → ``/calibrate`` → ``/order``
→ ``/observe`` — serves end to end with a real forecasting model.

PR #38 was certified "all green" by a suite that never ran the migration path
and never booted the app against a real database. This gate closes that hole;
as the SQL-backed lifecycle store lands, the same flow automatically exercises
the migrated lifecycle tables.
"""

from __future__ import annotations

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import app
from calibre.api.run_store import SqlRunStore
from calibre.core.forecast_frame import UNIQUE_ID, interval_column_names


@pytest.fixture
def migrated_client(fresh_db_url, monkeypatch):
    """App booted against a freshly migrated database, with clean stores."""
    monkeypatch.setenv("CALIBRE_DATABASE_URL", fresh_db_url)
    command.upgrade(Config("alembic.ini"), "head")

    # Reset module globals so the app re-resolves the SQL store / lifecycle
    # store against this test's database rather than leaking prior state.
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", LifecycleStore())
    monkeypatch.setattr(api_main, "_DB_FACTORY", None)
    monkeypatch.setattr(api_main, "_DB_URL", None)
    monkeypatch.setattr(api_main, "_SQL_STORE", None)
    return TestClient(app)


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(idx + 1)}
        for idx, ds in enumerate(dates)
    ]


def _fit_payload() -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records("A"),
        "forecaster_config": {"backend": "statsforecast", "model": "Naive"},
        "conformal_config": {
            "method": "aci",
            "coverage": 0.9,
            "calibration_window": 4,
            "gamma": 0.05,
        },
    }


def test_app_wired_to_migrated_db(migrated_client) -> None:
    """With CALIBRE_DATABASE_URL set, the app selects the SQL-backed store."""
    assert migrated_client.get("/healthz").json() == {"status": "ok"}
    assert isinstance(api_main._run_store(), SqlRunStore)


def test_lifecycle_roundtrip_on_migrated_db(migrated_client) -> None:
    client = migrated_client

    fit = client.post("/fit", json=_fit_payload())
    assert fit.status_code == 202, fit.text
    fit_id = fit.json()["fit_id"]
    session_id = fit.json()["session_id"]
    assert client.get(f"/fits/{fit_id}").json()["status"] == "succeeded"

    predict = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"})
    assert predict.status_code == 200, predict.text
    forecast = predict.json()["forecast"]
    assert len(forecast) == 2

    calibrate = client.post("/calibrate", json={"session_id": session_id, "forecast": forecast})
    assert calibrate.status_code == 200, calibrate.text
    calibrated = calibrate.json()["calibrated"]
    lower_col, upper_col = interval_column_names(0.9)
    assert all(lower_col in row and upper_col in row for row in calibrated)

    order = client.post(
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
    assert order.status_code == 200, order.text

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
    assert api_main._LIFECYCLE_STORE.get_conformal_state(session_id), (
        "observe should persist conformal state"
    )
