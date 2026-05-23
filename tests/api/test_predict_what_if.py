from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import MemoryLifecycleStore
from calibre.api.main import app
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H
from calibre.core.forecast_task import ForecastTask


class _FutureXAdapter:
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}

    def fit(self, task: ForecastTask) -> None:
        self._task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        origin = pd.Timestamp(task.forecast_origin)
        rows: list[dict] = []
        future_x = task.future_x if task.future_x is not None else pd.DataFrame()
        for h in range(1, int(task.horizon) + 1):
            ds = origin + pd.Timedelta(weeks=h)
            lift = 0.0
            if not future_x.empty and "promo_lift" in future_x.columns:
                matches = future_x[(future_x[UNIQUE_ID] == task.unique_id) & (future_x[DS] == ds)]
                if not matches.empty and pd.notna(matches.iloc[-1]["promo_lift"]):
                    lift = float(matches.iloc[-1]["promo_lift"])
            rows.append(
                {
                    UNIQUE_ID: task.unique_id,
                    DS: ds,
                    Y_HAT: 10.0 + h + lift,
                    H: h,
                }
            )
        return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _reset_lifecycle_store(monkeypatch):
    fresh = MemoryLifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", fresh)
    monkeypatch.setattr(
        "calibre.execution.backend.resolve_adapter",
        lambda cfg: _FutureXAdapter(cfg),
    )
    return fresh


@pytest.fixture
def client():
    return TestClient(app)


def _history_records() -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: "A", "ds": ds.strftime("%Y-%m-%d"), "y": float(idx)}
        for idx, ds in enumerate(dates)
    ]


def _fit_payload(*, future_x: list[dict] | None = None) -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records(),
        "future_x": future_x,
        "forecaster_config": {"backend": "stub", "model": "stub_model"},
    }


def _fit_id(client: TestClient, payload: dict) -> str:
    response = client.post("/fit", json=payload)
    assert response.status_code == 202, response.text
    return str(response.json()["fit_id"])


def _predict_yhat(client: TestClient, fit_id: str, **payload: object) -> list[float]:
    response = client.post(
        "/predict",
        json={"fit_id": fit_id, "origin": "2024-02-04", **payload},
    )
    assert response.status_code == 200, response.text
    return [row[Y_HAT] for row in response.json()["forecast"]]


def test_future_x_override_changes_forecast(client) -> None:
    fit_id = _fit_id(
        client,
        _fit_payload(
            future_x=[
                {UNIQUE_ID: "A", "ds": "2024-02-11", "price": 1.0},
                {UNIQUE_ID: "A", "ds": "2024-02-18", "price": 1.0},
            ]
        ),
    )

    baseline = _predict_yhat(client, fit_id)
    what_if = _predict_yhat(
        client,
        fit_id,
        future_x_override={
            "A": [
                {"ds": "2024-02-11", "promo_lift": 7.0},
                {"ds": "2024-02-18", "promo_lift": 3.0},
            ]
        },
    )

    assert baseline == [11.0, 12.0]
    assert what_if == [18.0, 15.0]


def test_override_does_not_persist_across_calls(client) -> None:
    fit_id = _fit_id(
        client,
        _fit_payload(
            future_x=[
                {UNIQUE_ID: "A", "ds": "2024-02-11", "promo_lift": 0.0},
                {UNIQUE_ID: "A", "ds": "2024-02-18", "promo_lift": 0.0},
            ]
        ),
    )

    what_if = _predict_yhat(
        client,
        fit_id,
        future_x_override={
            "A": [
                {"ds": "2024-02-11", "promo_lift": 5.0},
                {"ds": "2024-02-18", "promo_lift": 2.0},
            ]
        },
    )
    baseline_after = _predict_yhat(client, fit_id)

    assert what_if == [16.0, 14.0]
    assert baseline_after == [11.0, 12.0]
