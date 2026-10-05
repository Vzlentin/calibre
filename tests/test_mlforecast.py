import numpy as np
import pandas as pd
import pytest
from mlforecast.lag_transforms import RollingMean
from sklearn.linear_model import LinearRegression

from calibre.forecast import Covariate
from calibre.forecast.mlforecast import MLForecast
from calibre.forecast.origins import forecast_origins
from calibre.hierarchy import Hierarchy
from calibre.panel import Panel
from calibre.reconcile import BottomUp

ORIGINS = np.array([10, 15])


def panel_of(values: np.ndarray) -> tuple[Panel, Hierarchy]:
    series = np.array([f"s{i}" for i in range(len(values))])
    periods = pd.date_range("2024-01-01", periods=values.shape[1], freq="D")
    return Panel(series, periods, values, "D"), Hierarchy.flat(series)


def trend() -> tuple[Panel, Hierarchy]:
    """a = t and b = 10 + t: one global rule, y = lag1 + 1, fits both exactly."""
    return panel_of(np.arange(20, dtype=np.float32) + np.array([[0], [10]], dtype=np.float32))


def test_global_lag_model_continues_the_trend_recursively():
    panel, hierarchy = trend()
    model = MLForecast(LinearRegression(), lags=[1])
    run = forecast_origins(panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 3)
    np.testing.assert_allclose(run.points[0], [[11, 12, 13], [21, 22, 23]], atol=1e-4)
    np.testing.assert_allclose(run.points[1], [[16, 17, 18], [26, 27, 28]], atol=1e-4)


def test_short_fit_and_predict_tails_give_the_same_points():
    panel, hierarchy = trend()
    full = MLForecast(LinearRegression(), lags=[1, 2])
    tail = MLForecast(LinearRegression(), lags=[1, 2], fit_periods=6, lookback=2)
    expected = forecast_origins(panel, hierarchy, full, BottomUp(hierarchy), ORIGINS, 3)
    got = forecast_origins(panel, hierarchy, tail, BottomUp(hierarchy), ORIGINS, 3)
    np.testing.assert_allclose(got.points, expected.points, atol=1e-4)


def test_known_ahead_and_static_features():
    # y = 3 * promo + 10 * size. The promo plan is known 3 days ahead.
    rng = np.random.default_rng(0)
    promo = rng.integers(0, 5, size=(2, 23)).astype(np.float32)
    size = np.array([[1.0], [2.0]], dtype=np.float32)
    panel, hierarchy = panel_of(3 * promo[:, :20] + 10 * size)
    covariates = {
        "promo": Covariate(promo, known_ahead=True, aggregate="sum"),
        "size": Covariate(size, known_ahead=False, aggregate="sum"),
    }
    model = MLForecast(LinearRegression(), features=["promo", "size"])
    run = forecast_origins(
        panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 3, covariates=covariates
    )
    for index, origin in enumerate(ORIGINS):
        future = promo[:, origin + 1 : origin + 4]
        np.testing.assert_allclose(run.points[index], 3 * future + 10 * size, atol=1e-3)


def test_unknown_dynamic_features_and_missing_names_are_rejected():
    panel, hierarchy = trend()
    weather = {"weather": Covariate(np.ones((2, 20)), known_ahead=False, aggregate="mean")}
    model = MLForecast(LinearRegression(), features=["weather"])
    with pytest.raises(ValueError, match="not known ahead"):
        forecast_origins(panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 3, None, weather)
    model = MLForecast(LinearRegression(), features=["price"])
    with pytest.raises(ValueError, match="not a covariate"):
        forecast_origins(panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 3, None, weather)


def test_refit_returns_a_new_model_and_keeps_the_forecaster_unchanged():
    panel, hierarchy = trend()
    regressor = LinearRegression()
    model = MLForecast(regressor, lags=[1])
    forecast_origins(panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 3, refit_every=1)
    assert not hasattr(regressor, "coef_")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "at least one of"),
        ({"lags": [7], "lookback": 3}, "shorter than the longest lag"),
        ({"lags": [1], "fit_periods": 0}, "fit_periods"),
    ],
)
def test_mlforecast_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MLForecast(LinearRegression(), **kwargs)


@pytest.mark.filterwarnings("ignore:Found null values")
def test_a_lookback_shorter_than_a_lag_transform_window_fails():
    panel, hierarchy = trend()
    model = MLForecast(LinearRegression(), lag_transforms={1: [RollingMean(window_size=4)]})
    short = MLForecast(
        LinearRegression(), lag_transforms={1: [RollingMean(window_size=4)]}, lookback=2
    )
    forecast_origins(panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 3)
    with pytest.raises(ValueError, match="lookback is shorter"):
        forecast_origins(panel, hierarchy, short, BottomUp(hierarchy), ORIGINS, 3)
