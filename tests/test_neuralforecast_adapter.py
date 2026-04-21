from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from calibre.models.neuralforecast import NeuralForecastAdapter
from calibre.tasks.forecast_task import ForecastTask


def _nf_predict_return(uid: str, n: int) -> pd.DataFrame:
    """Minimal Nixtla-format predict output accepted by _build_predict_frame."""
    return pd.DataFrame(
        {
            "unique_id": [uid] * n,
            "ds": pd.date_range("2024-07-07", periods=n, freq="W"),
            "NHITS": [10.0] * n,
        }
    )


@pytest.fixture
def repeating_history():
    """24 periods of repeating [10, 20, 30, 40] pattern."""
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 6
    return pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})


@pytest.fixture
def nhits_task(repeating_history):
    return ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={"backend": "neuralforecast", "model": "NHITS", "freq": "W", "max_steps": 5},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


@pytest.fixture
def tide_task(repeating_history):
    return ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={"backend": "neuralforecast", "model": "TiDE", "freq": "W", "max_steps": 5},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


@pytest.fixture
def patchtst_task(repeating_history):
    return ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "neuralforecast",
            "model": "PatchTST",
            "freq": "W",
            "max_steps": 5,
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


def test_nhits_fit_predict_columns(nhits_task):
    adapter = NeuralForecastAdapter(nhits_task.model_config)
    adapter.fit(nhits_task)
    result = adapter.predict(nhits_task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_tide_fit_predict_columns(tide_task):
    adapter = NeuralForecastAdapter(tide_task.model_config)
    adapter.fit(tide_task)
    result = adapter.predict(tide_task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_patchtst_fit_predict_columns(patchtst_task):
    adapter = NeuralForecastAdapter(patchtst_task.model_config)
    adapter.fit(patchtst_task)
    result = adapter.predict(patchtst_task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_predict_before_fit_raises(nhits_task):
    adapter = NeuralForecastAdapter(nhits_task.model_config)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict(nhits_task)


def test_y_hat_dtype_is_float64(nhits_task):
    adapter = NeuralForecastAdapter(nhits_task.model_config)
    adapter.fit(nhits_task)
    result = adapter.predict(nhits_task)
    assert result["y_hat"].dtype == np.float64


def test_fit_preserves_exogenous_columns(monkeypatch, repeating_history):
    history = repeating_history.copy()
    history["promo"] = [0.0, 1.0] * 12
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={
            "backend": "neuralforecast",
            "model": "NHITS",
            "freq": "W",
            "max_steps": 5,
            "futr_exog_list": ["promo"],
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _nf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.models.neuralforecast.NeuralForecast", MagicMock(return_value=mock_instance)
    )

    adapter = NeuralForecastAdapter(task.model_config)
    adapter.fit(task)

    _, fit_kwargs = mock_instance.fit.call_args
    fit_df = fit_kwargs["df"]
    assert "promo" in fit_df.columns


def test_predict_forwards_future_x_as_futr_df(monkeypatch, repeating_history):
    history = repeating_history.copy()
    history["promo"] = [0.0, 1.0] * 12
    future_x = pd.DataFrame(
        {
            "unique_id": ["SKU_001"] * 4,
            "ds": pd.date_range("2024-07-07", periods=4, freq="W"),
            "promo": [1.0, 0.0, 1.0, 0.0],
        }
    )
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={
            "backend": "neuralforecast",
            "model": "NHITS",
            "freq": "W",
            "max_steps": 5,
            "futr_exog_list": ["promo"],
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
        future_x=future_x,
    )
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _nf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.models.neuralforecast.NeuralForecast", MagicMock(return_value=mock_instance)
    )

    adapter = NeuralForecastAdapter(task.model_config)
    adapter.fit(task)
    adapter.predict(task)

    _, predict_kwargs = mock_instance.predict.call_args
    assert "futr_df" in predict_kwargs
    pd.testing.assert_frame_equal(predict_kwargs["futr_df"], future_x)


def test_predict_without_future_x_omits_futr_df(monkeypatch, nhits_task):
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _nf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.models.neuralforecast.NeuralForecast", MagicMock(return_value=mock_instance)
    )

    adapter = NeuralForecastAdapter(nhits_task.model_config)
    adapter.fit(nhits_task)
    adapter.predict(nhits_task)

    _, predict_kwargs = mock_instance.predict.call_args
    assert "futr_df" not in predict_kwargs
