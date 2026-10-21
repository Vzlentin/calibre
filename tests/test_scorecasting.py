import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from statsforecast.models import SeasonalNaive as SfSeasonalNaive
from statsforecast.models import Theta

from calibre import AR1, ConformalPID, ForecastScorecaster, Scorecaster, SeasonalNaive
from calibre.forecast.models.statsforecast import StatsForecastModel


def histories(values):
    arrays = np.asarray(values, dtype=np.float64)
    if arrays.ndim == 1:
        arrays = arrays[None, None, :]
    counts = np.isfinite(arrays).sum(axis=-1).astype(np.int64)
    return arrays, counts, np.ones(counts.shape), np.ones(counts.shape[1], dtype=np.int64)


def test_scorecaster_is_an_abstract_contract():
    with pytest.raises(TypeError):
        Scorecaster()


def test_forecast_scorecaster_batches_unequal_histories_without_filling_holes():
    scores, counts, scale, _ = histories(
        [
            [[1, 2, 3, 4], [np.nan, np.nan, 8, 9]],
            [[np.nan, 5, 6, 7], [np.nan, np.nan, np.nan, np.nan]],
        ]
    )
    caster = ForecastScorecaster(SeasonalNaive(2), min_history=3)
    ahead = np.array([3, 1])
    # Three-step seasonal naive repeats the penultimate score; short histories use means.
    expected = [[3, 8.5], [6, 0]]
    before = [array.copy() for array in (scores, counts, scale, ahead)]
    np.testing.assert_array_equal(caster.predict(scores, counts, scale, ahead), expected)
    for actual, original in zip((scores, counts, scale, ahead), before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("seed_scale", [1e-8, 1e-4, 1.0, 10.0])
def test_ar1_adapter_preserves_the_fixed_seed_ridge_and_its_absolute_floor(seed_scale):
    history = np.array([1, 3, 2, 4]) * 1e-4
    scores, counts, scale, _ = histories(history)
    scale[:] = seed_scale
    caster = ForecastScorecaster(AR1(), scale_floor=1e-3)
    # x=[1,3,2]*1e-4, y=[3,2,4]*1e-4; covariance=-1e-8, denominator=2e-8.
    slope = -1e-8 / (2e-8 + max(seed_scale**2 * 1e-6, 1e-12))
    equilibrium = (3e-4 - slope * 2e-4) / (1 - slope)
    expected = equilibrium + slope**3 * (4e-4 - equilibrium)
    np.testing.assert_allclose(
        caster.predict(scores, counts, scale, np.array([3])), [[expected]], rtol=1e-14
    )


@pytest.mark.parametrize("season_length", [1, 4])
def test_theta_adapter_forecasts_each_column_horizon_with_the_real_vendor_model(season_length):
    history = 10 + 0.1 * np.arange(32) + np.tile([1, 0, -1, 0], 8)
    scores = np.broadcast_to(history, (2, 2, len(history))).copy()
    scores[1] *= 2
    counts = np.full((2, 2), len(history), dtype=np.int64)
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
    actual = caster.predict(scores, counts, np.ones((2, 2)), ahead)
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    np.testing.assert_array_equal(caster.predict(scores, counts, np.ones((2, 2)), ahead), actual)
    assert vars(vendor) == parameters
    assert not np.allclose(actual[:, 0], actual[:, 1])


def test_theta_short_history_uses_the_explicit_mean_fallback():
    caster = ForecastScorecaster(StatsForecastModel(Theta()), min_history=4)
    args = histories([1, 2, 3])
    np.testing.assert_array_equal(caster.predict(*args), [[2]])


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
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite predictions"):
        caster.predict(*histories([1e40, 2e40, 3e40]))


@pytest.mark.parametrize("path", ["fallback-mean", "restored-units"])
def test_forecast_scorecaster_rejects_nonfinite_score_unit_output(path):
    caster = ForecastScorecaster(AR1(), min_history=4, scale_floor=1e-3)
    if path == "fallback-mean":
        args = histories(np.full(3, np.finfo(np.float64).max))
    else:
        scores, counts, scale, ahead = histories(np.array([1, 2, 3, 4]) * 1e307)
        scale[:] = 1e308
        ahead[:] = 100
        args = scores, counts, scale, ahead
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="score units"):
        caster.predict(*args)


def test_pid_uses_a_seasonal_scorecaster_instead_of_ar1():
    method = ConformalPID(
        0.75,
        scorecaster=ForecastScorecaster(SeasonalNaive(2), min_history=2),
        ahead=np.arange(1, 4),
    )
    scores = np.broadcast_to(np.array([1, 8, 2, 7])[:, None, None], (4, 1, 3))
    state = method.seed(method.initial_state(1, 3), scores)
    np.testing.assert_array_equal(method._scorecast(state), [[2, 7, 2]])
    np.testing.assert_array_equal(state["proportional"], [[5, 0, 5]])
    np.testing.assert_array_equal(method.threshold(state), [[7, 7, 7]])


def test_pid_clips_negative_score_predictions_not_generic_forecasts():
    method = ConformalPID(
        0.9,
        scorecaster=ForecastScorecaster(AR1(), scale_floor=1e-3),
        ahead=10,
    )
    state = method.seed(method.initial_state(1, 1), np.array([10, 9, 8, 7])[:, None, None])
    np.testing.assert_array_equal(method._scorecast(state), [[0]])


def test_pid_requires_an_explicit_scorecaster_choice():
    missing: dict[str, Any] = {"level": 0.9}
    with pytest.raises(TypeError, match="scorecaster"):
        ConformalPID(**missing)
    method = ConformalPID(0.9, scorecaster=None)
    state = method.initial_state(1, 1)
    np.testing.assert_array_equal(method._scorecast(state), [[0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forecaster": object()},
        {"min_history": 0},
        {"min_history": True},
        {"min_history": 1.5},
        {"scale_floor": 0},
        {"scale_floor": -1},
        {"scale_floor": np.nan},
        {"scale_floor": np.inf},
        {"scale_floor": [1]},
    ],
)
def test_forecast_scorecaster_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ForecastScorecaster(**({"forecaster": AR1()} | kwargs))


@pytest.mark.parametrize("bad_input", ["shape", "count", "ahead", "hole", "scale"])
def test_forecast_scorecaster_validates_its_array_boundary(bad_input):
    scores, counts, scale, ahead = histories([1, 2, 3])
    if bad_input == "shape":
        counts = np.ones(2, dtype=np.int64)
    elif bad_input == "count":
        counts[:] = 4
    elif bad_input == "ahead":
        ahead[:] = 0
    elif bad_input == "hole":
        scores[0, 0, 1] = np.nan
    else:
        scale[:] = -1
    with pytest.raises(ValueError):
        ForecastScorecaster(AR1()).predict(scores, counts, scale, ahead)


def test_pid_and_forecast_keep_their_existing_dependency_boundaries():
    root = Path(__file__).parents[1] / "src" / "calibre"
    allowed = {
        "conformal/calibrators/pid.py": {
            "calibre.conformal.calibrators.base",
            "calibre.conformal.calibrators.ranks",
            "calibre.conformal.losses",
        },
        "conformal/calibrators/base.py": {"calibre.conformal.scores"},
        "forecast/models/ar1.py": {"calibre.forecast.models.base"},
        "backtest/scorecasting.py": {
            "calibre.conformal.calibrators.base",
            "calibre.forecast.models.base",
        },
    }
    for path, modules in allowed.items():
        imports = ast.walk(ast.parse((root / path).read_text()))
        for statement in imports:
            if (
                isinstance(statement, ast.ImportFrom)
                and statement.module is not None
                and statement.module.startswith("calibre")
            ):
                assert statement.module in modules


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
