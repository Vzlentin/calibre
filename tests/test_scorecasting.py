import subprocess
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from statsforecast.models import SeasonalNaive as SfSeasonalNaive
from statsforecast.models import Theta

from calibre import AR1, ConformalPID, ForecastScorecaster, SeasonalNaive
from calibre.forecast.models.mlforecast import MLForecast
from calibre.forecast.models.statsforecast import StatsForecastModel


def histories(values):
    history = np.asarray(values, dtype=np.float64)
    if history.ndim == 1:
        history = history[None, None, :]
    return history, np.ones(history.shape[1], dtype=np.int64)


def test_forecast_scorecaster_batches_unequal_histories_without_filling_holes():
    history, _ = histories(
        [
            [[1, 2, 3, 4], [np.nan, np.nan, 8, 9], [np.nan, 4, 5, 6]],
            [[np.nan, 5, 6, 7], [np.nan, np.nan, np.nan, np.nan], [2, 3, 4, 5]],
        ]
    )
    caster = ForecastScorecaster(SeasonalNaive(2), min_history=3)
    ahead = np.array([3, 1, 2])
    expected = [[3, 8.5, 6], [6, 0, 5]]
    before = [array.copy() for array in (history, ahead)]
    np.testing.assert_array_equal(caster.predict(history, ahead), expected)
    for actual, original in zip((history, ahead), before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_global_forecaster_pools_nodes_and_columns_only_within_each_length():
    history = np.array(
        [
            [[1, 2, 3, 4], [np.nan, 8, 7, 6]],
            [[np.nan, 2, 4, 6], [9, 6, 3, 1]],
        ],
        dtype=np.float64,
    )
    ahead = np.array([1, 3])
    caster = ForecastScorecaster(MLForecast(LinearRegression(), lags=[1]))
    expected = np.empty((2, 2))
    for positions in [[(0, 0), (1, 1)], [(0, 1), (1, 0)]]:
        rows = [history[node, column] for node, column in positions]
        rows = [row[np.isfinite(row)] for row in rows]
        x = np.concatenate([row[:-1] for row in rows])
        y = np.concatenate([row[1:] for row in rows])
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
        for (node, column), row in zip(positions, rows, strict=True):
            last = row[-1]
            for _ in range(ahead[column]):
                last = intercept + slope * last
            expected[node, column] = last
    np.testing.assert_allclose(caster.predict(history, ahead), expected, rtol=1e-6)


@pytest.mark.parametrize("season_length", [1, 4])
def test_theta_adapter_forecasts_each_column_horizon_with_the_real_vendor_model(season_length):
    history = 10 + 0.1 * np.arange(32) + np.tile([1, 0, -1, 0], 8)
    scores = np.broadcast_to(history, (2, 2, len(history))).copy()
    scores[1] *= 2
    ahead = np.array([1, 3])
    vendor = Theta(season_length=season_length)
    caster = ForecastScorecaster(StatsForecastModel(vendor), min_history=8)
    parameters = vars(vendor).copy()
    expected = np.empty((2, 2))
    for node in range(2):
        for column, horizon in enumerate(ahead):
            expected[node, column] = Theta(season_length=season_length).forecast(
                scores[node, column], h=int(horizon)
            )["mean"][-1]
    actual = caster.predict(scores, ahead)
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    np.testing.assert_array_equal(caster.predict(scores, ahead), actual)
    assert vars(vendor) == parameters
    assert not np.allclose(actual[:, 0], actual[:, 1])


@pytest.mark.parametrize("values", [[], [7], [2, 4]])
def test_adapter_owns_the_short_history_mean_fallback(values):
    history = np.full((1, 1, 3), np.nan)
    if values:
        history[0, 0, -len(values) :] = values
    caster = ForecastScorecaster(AR1())
    np.testing.assert_array_equal(
        caster.predict(history, np.array([3])), [[np.mean(values) if values else 0]]
    )


def test_theta_short_history_uses_the_explicit_mean_fallback():
    caster = ForecastScorecaster(StatsForecastModel(Theta()), min_history=4)
    np.testing.assert_array_equal(caster.predict(*histories([1, 2, 3])), [[2]])


@pytest.mark.parametrize("value", [0.0, 1.0, 7.0])
def test_theta_pid_preserves_constant_seed_quantiles(value):
    method = ConformalPID(
        0.9,
        scorecaster=ForecastScorecaster(StatsForecastModel(Theta()), min_history=4),
        ahead=3,
    )
    state = method.seed(method.initial_state(1, 1), np.full((4, 1, 1), value))
    np.testing.assert_allclose(method._scorecast(state), [[value]], rtol=1e-7)
    np.testing.assert_array_equal(method.threshold(state), [[value]])


def test_forecast_scorecaster_does_not_hide_model_errors():
    caster = ForecastScorecaster(SeasonalNaive(5))
    with pytest.raises(ValueError, match="season is 5"):
        caster.predict(*histories([1, 2, 3]))


def test_forecast_scorecaster_rejects_nonfinite_vendor_output():
    caster = ForecastScorecaster(StatsForecastModel(SfSeasonalNaive(1)))
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="forecaster must return finite"),
    ):
        caster.predict(*histories([1e40, 2e40, 3e40]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forecaster": object()},
        {"min_history": 0},
        {"min_history": True},
        {"min_history": 1.5},
    ],
)
def test_forecast_scorecaster_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ForecastScorecaster(**({"forecaster": AR1()} | kwargs))


def test_importing_calibre_does_not_load_optional_forecasting_vendors():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import calibre; "
            "vendors = {'statsforecast', 'statsmodels', 'neuralforecast', 'mlforecast'}; "
            "assert not vendors & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
