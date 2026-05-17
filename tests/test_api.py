from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from calibre.api.main import app
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter


class _ApiDatasetAdapter:
    def name(self) -> str:
        return "unit_api"

    def load(self, path: str, **kwargs) -> DatasetBundle:
        del path, kwargs
        dates = pd.date_range("2024-01-07", periods=8, freq="W")
        return DatasetBundle(
            history=pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]}),
            future_x=None,
            costs=CostStruct(),
            hierarchy=None,
            censoring=None,
        )


class _StubAdapter:
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}

    def fit(self, task: ForecastTask) -> None:
        self.task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id],
                DS: [task.forecast_origin + pd.Timedelta(weeks=1)],
                Y_HAT: [10.0],
                H: [1],
            }
        )


register_dataset_adapter("unit_api")(_ApiDatasetAdapter)


def _payload() -> dict:
    return {
        "config": {
            "config_schema": "1.0",
            "dataset": {"adapter": "unit_api", "path": "ignored"},
            "tasks": [
                {
                    "model": "stub_model",
                    "horizon": 1,
                    "config": {"backend": "stub"},
                }
            ],
            "origins": {"start": "2024-02-04", "end": "2024-02-04", "freq": "W-SUN"},
            "output": {},
            "execution": {"engine": None, "seed": 123},
        }
    }


def test_forecasts_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())
    client = TestClient(app)

    response = client.post("/forecasts", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 1
    assert body["forecasts"][0][UNIQUE_ID] == "A"


def test_backtests_endpoint_records_status(monkeypatch) -> None:
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())
    client = TestClient(app)

    response = client.post("/backtests", json=_payload(), headers={"Idempotency-Key": "abc"})

    assert response.status_code == 202
    run_id = response.json()["id"]
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "succeeded"
    assert status["artifact_urls"]["rows"] == "1"
