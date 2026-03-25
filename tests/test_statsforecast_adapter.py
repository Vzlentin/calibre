import numpy as np
import pandas as pd
import pytest

from calibre.models.statsforecast import StatsForecastAdapter
from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def repeating_history():
    """24 periods of repeating [10, 20, 30, 40] pattern."""
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 6
    return pd.DataFrame({"ds": dates, "y": pattern})


@pytest.fixture
def sn_task(repeating_history):
    return ForecastTask(
        unique_id="SKU_001",
        history=repeating_history,
        horizon=4,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4, "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


def test_fit_predict_returns_correct_columns(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)

    assert list(result.columns) == ["ds", "y_hat", "h"]
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
        unique_id="SKU_001",
        history=repeating_history,
        horizon=4,
        model_config={"backend": "statsforecast", "model": "AutoETS", "season_length": 4, "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = StatsForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert list(result.columns) == ["ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]
