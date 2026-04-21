from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from calibre.models.statsforecast import StatsForecastAdapter
from calibre.tasks.forecast_task import ForecastTask


def _sf_predict_return(uid: str, n: int) -> pd.DataFrame:
    """Minimal Nixtla-format predict output accepted by _build_predict_frame."""
    return pd.DataFrame(
        {
            "unique_id": [uid] * n,
            "ds": pd.date_range("2024-07-07", periods=n, freq="W"),
            "AutoARIMA": [10.0] * n,
        }
    )


@pytest.fixture
def repeating_history():
    """24 periods of repeating [10, 20, 30, 40] pattern."""
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 6
    return pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})


@pytest.fixture
def sn_task(repeating_history):
    return ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "season_length": 4,
            "freq": "W",
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


def test_fit_predict_returns_correct_columns(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_seasonal_naive_repeats_pattern(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)

    np.testing.assert_array_almost_equal(result["y_hat"].values, [10.0, 20.0, 30.0, 40.0])


def test_predict_before_fit_raises(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict(sn_task)


def test_y_hat_dtype_is_float64(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)
    assert result["y_hat"].dtype == np.float64


def test_auto_ets_fit_predict(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "statsforecast",
            "model": "AutoETS",
            "season_length": 4,
            "freq": "W",
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = StatsForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_fit_preserves_exogenous_columns(monkeypatch, repeating_history):
    history = repeating_history.copy()
    history["promo"] = [0.0, 1.0] * 12
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={
            "backend": "statsforecast",
            "model": "AutoARIMA",
            "freq": "W",
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _sf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.models.statsforecast.StatsForecast", MagicMock(return_value=mock_instance)
    )

    adapter = StatsForecastAdapter(task.model_config)
    adapter.fit(task)

    fit_df = mock_instance.fit.call_args[0][0]
    assert "promo" in fit_df.columns


def test_predict_forwards_future_x_as_X_df(monkeypatch, repeating_history):
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
            "backend": "statsforecast",
            "model": "AutoARIMA",
            "freq": "W",
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
        future_x=future_x,
    )
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _sf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.models.statsforecast.StatsForecast", MagicMock(return_value=mock_instance)
    )

    adapter = StatsForecastAdapter(task.model_config)
    adapter.fit(task)
    adapter.predict(task)

    _, predict_kwargs = mock_instance.predict.call_args
    assert "X_df" in predict_kwargs
    pd.testing.assert_frame_equal(predict_kwargs["X_df"], future_x)


def test_predict_without_future_x_omits_X_df(monkeypatch, sn_task):
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _sf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.models.statsforecast.StatsForecast", MagicMock(return_value=mock_instance)
    )

    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    adapter.predict(sn_task)

    _, predict_kwargs = mock_instance.predict.call_args
    assert "X_df" not in predict_kwargs
