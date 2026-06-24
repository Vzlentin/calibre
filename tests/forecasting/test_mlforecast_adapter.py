"""Tests for the MLForecast adapter."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mlforecast.lag_transforms import RollingMean, RollingStd

from calibre.core.forecast_frame import DS, FITTED_Y_HAT, IN_STOCK, MODEL_NAME, UNIQUE_ID, H, Y
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


def test_fit_forwards_static_features_to_mlforecast(monkeypatch, repeating_history):
    """``static_features`` from the config reaches ``MLForecast.fit`` verbatim.

    An empty list declares every exogenous column dynamic, which is what a
    time-varying ``future_x`` feature (e.g. ``promo``) requires.
    """
    history = repeating_history.copy()
    history["promo"] = [0.0, 1.0] * 12
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "static_features": [],
        },
    )
    mock_instance = MagicMock()
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast", MagicMock(return_value=mock_instance)
    )

    MLForecastAdapter(task.model_config).fit(task)

    _, fit_kwargs = mock_instance.fit.call_args
    assert fit_kwargs["static_features"] == []


def test_fit_omits_static_features_when_unset(monkeypatch, lgbm_task):
    """Absent ``static_features`` is not forwarded (mlforecast keeps its default)."""
    mock_instance = MagicMock()
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast", MagicMock(return_value=mock_instance)
    )

    MLForecastAdapter(lgbm_task.model_config).fit(lgbm_task)

    _, fit_kwargs = mock_instance.fit.call_args
    assert "static_features" not in fit_kwargs


def test_fitted_values_normalize_to_sidecar_contract(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "name": "global_lgbm",
            "freq": "W",
            "lags": [1, 2],
            "verbosity": -1,
            "n_estimators": 5,
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)

    adapter.fit(task, collect_fitted_values=True)
    fitted = adapter.fitted_values(task)

    assert list(fitted.columns) == ["unique_id", "ds", "y", "model_name", "fitted_y_hat"]
    assert set(fitted[MODEL_NAME]) == {"global_lgbm"}
    assert fitted[FITTED_Y_HAT].dtype == np.float64
    assert fitted[Y].dtype == np.float64
    assert fitted[FITTED_Y_HAT].notna().all()


def test_quantile_fitted_values_use_point_quantile(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "name": "quantile_lgbm",
            "freq": "W",
            "objective": "quantile",
            "quantiles": [0.52],
            "strategy": "direct",
            "verbosity": -1,
            "n_estimators": 5,
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    adapter = MLForecastAdapter(task.model_config)

    adapter.fit(task, collect_fitted_values=True)
    fitted = adapter.fitted_values(task)

    assert set(fitted[MODEL_NAME]) == {"quantile_lgbm"}
    assert fitted[FITTED_Y_HAT].notna().all()


def test_direct_fitted_values_drop_backend_horizon_metadata(repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=2,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "name": "direct_lgbm",
            "freq": "W",
            "strategy": "direct",
        },
        forecast_origin=pd.Timestamp("2024-06-23"),
    )
    raw = pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["SKU_001"] * 4, dtype="object"),
            DS: pd.to_datetime(["2024-01-14", "2024-01-14", "2024-01-21", "2024-01-21"]),
            Y: np.array([20.0, 20.0, 30.0, 30.0], dtype=np.float64),
            H: np.array([1, 2, 1, 2], dtype=np.int64),
            "LGBMRegressor": np.array([19.0, 18.0, 31.0, 29.0], dtype=np.float64),
        }
    )
    adapter = MLForecastAdapter(task.model_config)
    adapter._mlf = MagicMock()
    adapter._mlf.forecast_fitted_values.return_value = raw

    fitted = adapter.fitted_values(task)

    assert H not in fitted.columns
    assert fitted.duplicated([UNIQUE_ID, DS, MODEL_NAME]).sum() == 0
    assert fitted[FITTED_Y_HAT].tolist() == [19.0, 31.0]
    assert set(fitted[MODEL_NAME]) == {"direct_lgbm"}


def _capture_fit_df(monkeypatch, task) -> pd.DataFrame:
    """Spy on the underlying mlforecast ``.fit`` and return the frame it received."""
    mock_instance = MagicMock()
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast", MagicMock(return_value=mock_instance)
    )
    MLForecastAdapter(task.model_config).fit(task)
    return mock_instance.fit.call_args[0][0]


def _censoring_frame(history: pd.DataFrame, oos_index: int) -> pd.DataFrame:
    """Long in-stock frame marking exactly one week out of stock for SKU_001."""
    in_stock = [True] * len(history)
    in_stock[oos_index] = False
    return pd.DataFrame(
        {
            UNIQUE_ID: history[UNIQUE_ID].to_numpy(),
            DS: history[DS].to_numpy(),
            IN_STOCK: in_stock,
        }
    )


def test_gate_off_uses_observed_target_even_with_censoring(monkeypatch, repeating_history):
    censoring = _censoring_frame(repeating_history, oos_index=10)
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "freq": "W"},
        censoring=censoring,
    )

    fit_df = _capture_fit_df(monkeypatch, task)

    expected = repeating_history[Y].astype("float32").to_numpy()
    assert np.array_equal(fit_df[Y].to_numpy(), expected)


def test_gate_on_with_censoring_none_falls_back_to_observed(monkeypatch, repeating_history):
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "censoring_fit": True,
        },
        censoring=None,
    )

    fit_df = _capture_fit_df(monkeypatch, task)

    expected = repeating_history[Y].astype("float32").to_numpy()
    assert np.array_equal(fit_df[Y].to_numpy(), expected)


def test_gate_on_with_censoring_uses_imputed_target(monkeypatch, repeating_history):
    # Force an out-of-stock week whose imputed demand exceeds the censored
    # observation: at index 8 (a value-10 week) the expanding median of prior
    # in-stock sales (~25) lifts the target above the observed 10.
    oos_index = 8
    censoring = _censoring_frame(repeating_history, oos_index=oos_index)
    task = ForecastTask(
        history=repeating_history,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "model": "lightgbm.LGBMRegressor",
            "freq": "W",
            "censoring_fit": True,
        },
        censoring=censoring,
    )

    fit_df = _capture_fit_df(monkeypatch, task)

    # Exactly one Y column — guards the rename-then-select duplicate landmine.
    assert list(fit_df.columns).count(Y) == 1
    # The OOS week's target is lifted above the censored observation.
    observed = float(repeating_history[Y].iloc[oos_index])
    assert float(fit_df[Y].iloc[oos_index]) > observed
