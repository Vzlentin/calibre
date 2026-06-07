from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import app
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import ModelAdapter


class _FutureXAdapter(ModelAdapter):
    """Stub whose forecast bends with a ``promo_lift`` future_x column.

    The point of these tests is the override-merge wiring, so the adapter
    records the ``future_x`` frame it was actually handed; assertions inspect
    that captured payload rather than re-deriving the toy ``10 + h + lift``.
    """

    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}
        self.received: list[pd.DataFrame | None] = []

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del collect_fitted_values
        self._task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        self.received.append(None if task.future_x is None else task.future_x.copy())
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

    def dump_state(self) -> bytes:
        return b"{}"

    def load_state(self, blob: bytes) -> None:
        return None


@pytest.fixture
def received_future_x() -> list[pd.DataFrame | None]:
    """Collects the future_x each adapter instance received in ``predict``."""
    return []


@pytest.fixture(autouse=True)
def _reset_lifecycle_store(monkeypatch, tmp_path, received_future_x):
    fresh = LifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", fresh)
    monkeypatch.setenv("CALIBRE_ARTIFACT_URI", str(tmp_path / "artifacts"))

    def _make_adapter(cfg):
        adapter = _FutureXAdapter(cfg)
        adapter.received = received_future_x
        return adapter

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", _make_adapter)
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


@pytest.fixture
def sales_uri(tmp_path):
    path = tmp_path / "sales.parquet"
    pd.DataFrame(_history_records()).to_parquet(path)
    return str(path)


def _write_future_x(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "future_x.parquet"
    frame = pd.DataFrame(records)
    frame[DS] = pd.to_datetime(frame[DS])
    frame.to_parquet(path)
    return str(path)


def _fit_payload(sales_uri: str, *, future_x_uri: str | None = None) -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "sales_uri": sales_uri,
        "future_x_uri": future_x_uri,
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


def _lift_at(frame: pd.DataFrame | None, uid: str, ds: str) -> float | None:
    """The ``promo_lift`` the adapter saw for ``(uid, ds)``, or None if absent."""
    if frame is None or "promo_lift" not in frame.columns:
        return None
    match = frame[(frame[UNIQUE_ID] == uid) & (frame[DS] == pd.Timestamp(ds))]
    if match.empty or pd.isna(match.iloc[-1]["promo_lift"]):
        return None
    return float(match.iloc[-1]["promo_lift"])


def test_future_x_override_changes_forecast(client, sales_uri, tmp_path, received_future_x) -> None:
    future_x_uri = _write_future_x(
        tmp_path,
        [
            {UNIQUE_ID: "A", "ds": "2024-02-11", "price": 1.0},
            {UNIQUE_ID: "A", "ds": "2024-02-18", "price": 1.0},
        ],
    )
    fit_id = _fit_id(client, _fit_payload(sales_uri, future_x_uri=future_x_uri))
    # /fit eagerly validates by predicting once; drop that capture so only the
    # two /predict calls below remain.
    received_future_x.clear()

    baseline = _predict_yhat(client, fit_id)
    baseline_x = received_future_x[-1]
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
    what_if_x = received_future_x[-1]

    # The baseline request carries no override, so the adapter never sees promo_lift.
    assert _lift_at(baseline_x, "A", "2024-02-11") is None
    assert _lift_at(baseline_x, "A", "2024-02-18") is None
    # The override payload is merged onto the right (uid, ds) and reaches the adapter.
    assert _lift_at(what_if_x, "A", "2024-02-11") == 7.0
    assert _lift_at(what_if_x, "A", "2024-02-18") == 3.0
    # Having reached the model, the override actually moves the forecast.
    assert what_if != baseline


def test_override_does_not_persist_across_calls(
    client, sales_uri, tmp_path, received_future_x
) -> None:
    future_x_uri = _write_future_x(
        tmp_path,
        [
            {UNIQUE_ID: "A", "ds": "2024-02-11", "promo_lift": 0.0},
            {UNIQUE_ID: "A", "ds": "2024-02-18", "promo_lift": 0.0},
        ],
    )
    fit_id = _fit_id(client, _fit_payload(sales_uri, future_x_uri=future_x_uri))
    # /fit eagerly validates by predicting once; drop that capture so only the
    # two /predict calls below remain.
    received_future_x.clear()

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
    what_if_x = received_future_x[-1]
    baseline_after = _predict_yhat(client, fit_id)
    baseline_after_x = received_future_x[-1]

    # The override reaches the adapter on the what-if call...
    assert _lift_at(what_if_x, "A", "2024-02-11") == 5.0
    assert _lift_at(what_if_x, "A", "2024-02-18") == 2.0
    # ...but the next call sees only the persisted base future_x (lift 0.0): the
    # override is request-scoped and does not mutate stored state.
    assert _lift_at(baseline_after_x, "A", "2024-02-11") == 0.0
    assert _lift_at(baseline_after_x, "A", "2024-02-18") == 0.0
    # And the forecast reverts once the transient override is gone.
    assert what_if != baseline_after
