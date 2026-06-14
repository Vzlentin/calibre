"""Tests for NeuralForecast native persistence round-trips."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting import neuralforecast_adapter as adapter_module


class _FakeNeuralForecast:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "state.txt").write_text(str(self.value), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> _FakeNeuralForecast:
        value = float((Path(path) / "state.txt").read_text(encoding="utf-8"))
        return _FakeNeuralForecast(value)

    def predict(self, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                UNIQUE_ID: ["SKU_001", "SKU_001"],
                DS: pd.date_range("2024-07-07", periods=2, freq="W"),
                "NHITS": [self.value, self.value + 1.0],
            }
        )


def test_neuralforecast_native_state_round_trip_with_mock(monkeypatch) -> None:
    # Construct through the real adapter (availability gate + constructor) and
    # exercise the real dump_state/load_state/predict surface. Only the fitted
    # model object is faked: a genuine fit would train a neuralforecast net, so
    # the fake stands in for trained weights that the persistence layer ferries.
    monkeypatch.setattr(adapter_module, "NeuralForecast", _FakeNeuralForecast)
    monkeypatch.setattr(adapter_module, "_NEURALFORECAST_AVAILABLE", True)
    task = ForecastTask(
        history=pd.DataFrame(
            {
                UNIQUE_ID: ["SKU_001"] * 4,
                DS: pd.date_range("2024-06-02", periods=4, freq="W"),
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        horizon=2,
        model_config={"backend": "neuralforecast", "model": "NHITS", "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = adapter_module.NeuralForecastAdapter(task.model_config)
    adapter._nf = _FakeNeuralForecast(12.0)

    blob = adapter.dump_state()
    restored = adapter_module.NeuralForecastAdapter(task.model_config)
    restored.load_state(blob)

    result = restored.predict(task)
    assert list(result.columns) == [UNIQUE_ID, DS, Y_HAT, H]
    assert result[Y_HAT].tolist() == [12.0, 13.0]
