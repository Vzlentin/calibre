from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mlforecast.lag_transforms import RollingMean, RollingStd

from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.mlforecast_adapter import MLForecastAdapter


def _mlf_predict_return(uid: str, n: int) -> pd.DataFrame:
    """Minimal Nixtla-format predict output accepted by build_predict_frame."""
    return pd.DataFrame(
        {
            "unique_id": [uid] * n,
            "ds": pd.date_range("2024-07-07", periods=n, freq="W"),
            "LGBMRegressor": [10.0] * n,
        }
    )


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


@pytest.mark.parametrize("model", ["lightgbm.LGBMRegressor", "xgboost.XGBRegressor"])
def test_fit_predict_columns(repeating_history, model):
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={"backend": "mlforecast", "model": model, "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)
    adapter.fit(task)
    result = adapter.predict(task)

    assert list(result.columns) == ["unique_id", "ds", "y_hat", "h"]
    assert result["h"].tolist() == [1, 2, 3, 4]
    # Tree ensembles predict averages of leaf targets, so forecasts stay within
    # the observed target range [10, 40] of the repeating pattern (no extrapolation).
    assert result["y_hat"].between(10.0 - 1e-9, 40.0 + 1e-9).all()


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


def test_lag_transform_specs_are_resolved_for_yaml_configs(monkeypatch, repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "lags": [1, 2],
            "lag_transforms": {
                1: [
                    {"name": "RollingMean", "window_size": 4},
                    {"transform": "mlforecast.lag_transforms.RollingStd", "window_size": 4},
                ]
            },
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    mock_instance = MagicMock()
    mock_mlf = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast",
        mock_mlf,
    )

    MLForecastAdapter(task.model_config).fit(task)

    constructor_kwargs = mock_mlf.call_args.kwargs
    transforms = constructor_kwargs["lag_transforms"][1]
    assert isinstance(transforms[0], RollingMean)
    assert isinstance(transforms[1], RollingStd)


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


def test_native_state_round_trip_preserves_quantile_mapping(repeating_history, monkeypatch):
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
    expected = adapter.predict(task)

    blob = adapter.dump_state()
    restored = MLForecastAdapter(task.model_config)
    restored.load_state(blob)

    def fail_fit(_task):
        raise AssertionError("restored adapter should not refit")

    monkeypatch.setattr(restored, "fit", fail_fit)
    assert restored._name_to_quantile == adapter._name_to_quantile
    pd.testing.assert_frame_equal(restored.predict(task), expected)


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


def test_fit_preserves_exogenous_columns(monkeypatch, repeating_history):
    history = repeating_history.copy()
    history["promo"] = [0.0, 1.0] * 12
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _mlf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast", MagicMock(return_value=mock_instance)
    )

    adapter = MLForecastAdapter(task.model_config)
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
        model_config={"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
        future_x=future_x,
    )
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _mlf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast", MagicMock(return_value=mock_instance)
    )

    adapter = MLForecastAdapter(task.model_config)
    adapter.fit(task)
    adapter.predict(task)

    _, predict_kwargs = mock_instance.predict.call_args
    assert "X_df" in predict_kwargs
    pd.testing.assert_frame_equal(predict_kwargs["X_df"], future_x)


def test_predict_without_future_x_omits_X_df(monkeypatch, lgbm_task):
    mock_instance = MagicMock()
    mock_instance.predict.return_value = _mlf_predict_return("SKU_001", 4)
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast", MagicMock(return_value=mock_instance)
    )

    adapter = MLForecastAdapter(lgbm_task.model_config)
    adapter.fit(lgbm_task)
    adapter.predict(lgbm_task)

    _, predict_kwargs = mock_instance.predict.call_args
    assert "X_df" not in predict_kwargs
