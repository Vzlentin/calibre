from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import MemoryLifecycleStore
from calibre.api.main import app
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import ModelAdapter


class _CountingAdapter(ModelAdapter):
    fit_calls = 0

    def __init__(self, model_config: dict) -> None:
        self.model_config = model_config
        self._mean: float | None = None

    def fit(self, task: ForecastTask) -> None:
        type(self).fit_calls += 1
        self._mean = float(task.history[Y].mean())

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        assert self._mean is not None
        origin = pd.Timestamp(task.forecast_origin)
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id] * int(task.horizon),
                DS: [origin + pd.Timedelta(weeks=h) for h in range(1, int(task.horizon) + 1)],
                Y_HAT: [self._mean] * int(task.horizon),
                H: list(range(1, int(task.horizon) + 1)),
            }
        )


@pytest.fixture(autouse=True)
def _reset_lifecycle_store(monkeypatch, tmp_path):
    _CountingAdapter.fit_calls = 0
    monkeypatch.setenv("CALIBRE_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    monkeypatch.setattr(api_main, "_MODEL_CACHE", None)
    monkeypatch.setattr(api_main, "_MODEL_CACHE_URI", None)
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", MemoryLifecycleStore())
    yield
    _CountingAdapter.fit_calls = 0


@pytest.fixture
def counting_adapter(monkeypatch):
    monkeypatch.setattr(
        "calibre.execution.backend.resolve_adapter",
        lambda cfg: _CountingAdapter(cfg),
    )


@pytest.fixture
def client():
    return TestClient(app)


def _history_records() -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=4, freq="W-SUN")
    return [
        {UNIQUE_ID: "A", "ds": ds.strftime("%Y-%m-%d"), "y": float(idx + 1)}
        for idx, ds in enumerate(dates)
    ]


def _fit_payload(*, forecaster_config: dict | None = None) -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records(),
        "forecaster_config": forecaster_config or {"backend": "stub", "model": "mean"},
    }


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    assert parsed.scheme == "file"
    return Path(url2pathname(unquote(parsed.path)))


def test_fit_actually_trains_model(client, counting_adapter) -> None:
    response = client.post("/fit", json=_fit_payload())

    assert response.status_code == 202, response.text
    status = client.get(f"/fits/{response.json()['fit_id']}").json()
    assert status["status"] == "succeeded"
    assert _CountingAdapter.fit_calls == 1


def test_fit_fails_on_invalid_config(client) -> None:
    response = client.post(
        "/fit",
        json=_fit_payload(forecaster_config={"backend": "does-not-exist", "model": "mean"}),
    )

    assert response.status_code == 202, response.text
    status = client.get(f"/fits/{response.json()['fit_id']}").json()
    assert status["status"] == "failed"
    assert "Unknown backend" in status["error"]


def test_fit_artifact_stored_and_loadable(client, counting_adapter) -> None:
    fit = client.post("/fit", json=_fit_payload()).json()
    status = client.get(f"/fits/{fit['fit_id']}").json()
    artifact_urls = status["artifact_urls"]

    assert artifact_urls
    assert all(_path_from_file_uri(uri).exists() for uri in artifact_urls.values())
    assert _CountingAdapter.fit_calls == 1

    predict = client.post(
        "/predict",
        json={"fit_id": fit["fit_id"], "origin": "2024-02-04"},
    )

    assert predict.status_code == 200, predict.text
    assert _CountingAdapter.fit_calls == 1
    assert [row[Y_HAT] for row in predict.json()["forecast"]] == [2.5, 2.5]
