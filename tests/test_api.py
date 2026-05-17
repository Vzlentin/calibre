from __future__ import annotations

import copy
from pathlib import Path
from uuid import UUID

import pandas as pd
from fastapi.testclient import TestClient

from calibre.api.main import app
from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter
from calibre.execution.ledger import resolved_ledger_uri
from calibre.storage.models import Base
from calibre.storage.postgres import (
    ForecastPointerRepo,
    RunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)


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


class _ManyApiDatasetAdapter:
    count = 31

    def name(self) -> str:
        return "unit_api_many"

    def load(self, path: str, **kwargs) -> DatasetBundle:
        del path, kwargs
        dates = pd.date_range("2024-01-07", periods=8, freq="W")
        rows = [
            {UNIQUE_ID: f"SKU_{idx:02d}", DS: ds, Y: float(step)}
            for idx in range(self.count)
            for step, ds in enumerate(dates)
        ]
        return DatasetBundle(
            history=pd.DataFrame(rows),
            future_x=None,
            costs=CostStruct(),
            hierarchy=None,
            censoring=None,
        )


class _ThirtyApiDatasetAdapter(_ManyApiDatasetAdapter):
    count = 30

    def name(self) -> str:
        return "unit_api_thirty"


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
register_dataset_adapter("unit_api_many")(_ManyApiDatasetAdapter)
register_dataset_adapter("unit_api_thirty")(_ThirtyApiDatasetAdapter)


def _payload(adapter: str = "unit_api") -> dict:
    return {
        "config": {
            "config_schema": "1.0",
            "dataset": {"adapter": adapter, "path": "ignored"},
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


def test_metrics_endpoint_exposes_prometheus_payload() -> None:
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "calibre_forecast_duration_seconds" in response.text


def test_forecasts_endpoint_rejects_more_than_30_skus() -> None:
    client = TestClient(app)

    response = client.post("/forecasts", json=_payload(adapter="unit_api_many"))

    assert response.status_code == 400
    assert "maximum allowed is 30" in response.json()["detail"]


def test_forecasts_endpoint_accepts_30_skus_and_returns_intervals(monkeypatch) -> None:
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())
    payload = _payload(adapter="unit_api_thirty")
    payload["config"]["conformal"] = {
        "method": "aci",
        "coverage": 0.9,
        "calibration_window": 4,
        "gamma": 0.05,
    }
    client = TestClient(app)

    response = client.post("/forecasts", json=payload)

    assert response.status_code == 200
    body = response.json()
    lower_col, upper_col = interval_column_names(0.9)
    assert body["rows"] == 30
    assert {row[UNIQUE_ID] for row in body["forecasts"]} == {f"SKU_{idx:02d}" for idx in range(30)}
    assert all(lower_col in row and upper_col in row for row in body["forecasts"])


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


def test_backtests_endpoint_persists_runs_and_pointers(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    db_url = f"sqlite+pysqlite:///{(tmp_path / 'runs.db').as_posix()}"
    db = make_engine(db_url)
    Base.metadata.create_all(db)
    factory = make_session_factory(db)
    monkeypatch.setenv("CALIBRE_DATABASE_URL", db_url)

    payload = _payload()
    ledger_path = tmp_path / "ledger.parquet"
    payload["config"]["output"] = {
        "ledger_path": ledger_path.as_posix(),
        "streaming": False,
    }
    client = TestClient(app)
    expected_payload = _payload()
    expected_result = run_config(load_config_from_mapping(expected_payload["config"]))
    expected_ledger = expected_result.ledger.to_df().reset_index(drop=True)

    response = client.post("/backtests", json=payload, headers={"Idempotency-Key": "db-key"})

    assert response.status_code == 202
    run_id = response.json()["id"]
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "succeeded"
    assert status["artifact_urls"]["rows"] == "1"
    assert "ledger" in status["artifact_urls"]
    assert Path(status["artifact_urls"]["ledger"]) == ledger_path
    assert ledger_path.exists()
    actual_ledger = pd.read_parquet(ledger_path).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_ledger, expected_ledger)

    with session_scope(factory) as session:
        run = RunRepo(session).get(UUID(run_id))
        assert run is not None
        assert run.status == "succeeded"
        pointer = ForecastPointerRepo(session).get(UUID(run_id), "ledger")
        assert pointer is not None
        assert pointer.byte_size > 0


def test_backtests_endpoint_retries_failed_db_run_from_streaming_pointer(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {"count": 0, "fail": False}

    class _FlakyStubAdapter(_StubAdapter):
        def predict(self, task: ForecastTask) -> pd.DataFrame:
            calls["count"] += 1
            if calls["fail"] and calls["count"] == 2:
                raise RuntimeError("interrupted between origins")
            return super().predict(task)

    monkeypatch.setattr(
        "calibre.execution.task_builder.get_adapter_cls",
        lambda _: _FlakyStubAdapter,
    )
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _FlakyStubAdapter())

    db_url = f"sqlite+pysqlite:///{(tmp_path / 'retry.db').as_posix()}"
    db = make_engine(db_url)
    Base.metadata.create_all(db)
    monkeypatch.setenv("CALIBRE_DATABASE_URL", db_url)

    payload = _payload()
    payload["config"]["origins"] = {
        "start": "2024-02-04",
        "end": "2024-02-18",
        "freq": "W-SUN",
    }
    ledger_path = tmp_path / "streaming-ledger.parquet"
    payload["config"]["output"] = {
        "ledger_path": ledger_path.as_posix(),
        "streaming": True,
    }

    expected_payload = copy.deepcopy(payload)
    expected_payload["config"]["output"] = {}
    expected = run_config(load_config_from_mapping(expected_payload["config"])).ledger.to_df()

    client = TestClient(app)
    calls["count"] = 0
    calls["fail"] = True
    first = client.post("/backtests", json=payload, headers={"Idempotency-Key": "retry-key"})

    assert first.status_code == 202
    run_id = first.json()["id"]
    failed = client.get(f"/runs/{run_id}").json()
    assert failed["status"] == "failed"
    assert ledger_path.exists()

    calls["count"] = 0
    calls["fail"] = False
    retry = client.post("/backtests", json=payload, headers={"Idempotency-Key": "retry-key"})

    assert retry.status_code == 202
    assert retry.json()["id"] == run_id
    assert retry.json()["status"] == "queued"
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "succeeded"
    assert Path(status["artifact_urls"]["ledger"]) == Path(resolved_ledger_uri(ledger_path))

    actual = pd.read_parquet(status["artifact_urls"]["ledger"])
    sort_cols = [UNIQUE_ID, FORECAST_ORIGIN, H]
    pd.testing.assert_frame_equal(
        actual.sort_values(sort_cols).reset_index(drop=True),
        expected.sort_values(sort_cols).reset_index(drop=True),
    )
