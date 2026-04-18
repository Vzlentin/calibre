import numpy as np
import pandas as pd
import pytest

from calibre.models.mlforecast import MLForecastAdapter
from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def repeating_history():
    """24 periods of repeating [10, 20, 30, 40] pattern."""
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 6
    return pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})


@pytest.fixture
def lgbm_task(repeating_history):
    return ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


@pytest.fixture
def xgb_task(repeating_history):
    return ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={"backend": "mlforecast", "model": "xgboost.XGBRegressor", "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


def test_lightgbm_fit_predict_columns(lgbm_task):
    adapter = MLForecastAdapter(lgbm_task.model_config)
    adapter.fit(lgbm_task)
    result = adapter.predict(lgbm_task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_xgboost_fit_predict_columns(xgb_task):
    adapter = MLForecastAdapter(xgb_task.model_config)
    adapter.fit(xgb_task)
    result = adapter.predict(xgb_task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_predict_before_fit_raises(lgbm_task):
    adapter = MLForecastAdapter(lgbm_task.model_config)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict(lgbm_task)


def test_y_hat_dtype_is_float64(lgbm_task):
    adapter = MLForecastAdapter(lgbm_task.model_config)
    adapter.fit(lgbm_task)
    result = adapter.predict(lgbm_task)
    assert result["y_hat"].dtype == np.float64


def test_custom_lags_produces_valid_output(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "lags": [1, 2],
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert len(result) == 4


def test_quantile_models_produce_quantile_columns(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "objective": "quantile",
            "quantiles": [0.5, 0.833],
            "verbosity": -1,
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert "q_0p5" in result.columns
    assert "q_0p833" in result.columns
    assert "y_hat" in result.columns
    # y_hat should equal the median quantile when 0.5 is requested
    assert (result["y_hat"] == result["q_0p5"]).all()
    assert result["h"].tolist() == [1, 2, 3]


def test_direct_strategy_runs(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "strategy": "direct",
            "verbosity": -1,
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert result["h"].tolist() == [1, 2, 3]


def test_quantile_plus_direct_strategy(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "objective": "quantile",
            "quantiles": [0.52],
            "strategy": "direct",
            "verbosity": -1,
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert "q_0p52" in result.columns
    # With a single quantile, y_hat == that quantile
    assert (result["y_hat"] == result["q_0p52"]).all()


def test_invalid_strategy_raises(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "strategy": "fancy",
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    with pytest.raises(ValueError, match="strategy"):
        adapter.fit(task)


def test_invalid_quantile_value_raises(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "quantiles": [1.5],
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    with pytest.raises(ValueError, match="quantile"):
        adapter.fit(task)
