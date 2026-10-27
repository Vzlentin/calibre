import numpy as np
import pandas as pd
import pytest

from calibre import AR1, Window


def window(history, horizon=3, x=None):
    y = np.atleast_2d(np.asarray(history, dtype=np.float64))
    y.flags.writeable = False
    return Window(
        y=y,
        x={} if x is None else x,
        periods=pd.date_range("2024-01-01", periods=y.shape[1] + horizon),
        horizon=horizon,
    )


def test_ar1_multi_step_forecasts_match_an_independent_ridge_fit():
    history = np.array([2, 4, 5.5, 6.625, 7.46875])
    model = AR1(ridge=0.01)
    data = window(history)
    design = np.column_stack([np.ones(4), history[:-1]])
    penalty = np.sqrt(0.01 * np.var(history[:-1]))
    design = np.vstack([design, [0, penalty]])
    intercept, slope = np.linalg.lstsq(design, np.append(history[1:], 0), rcond=None)[0]
    expected, last = [], history[-1]
    for _ in range(3):
        last = intercept + slope * last
        expected.append(last)
    np.testing.assert_allclose(model.fit(data).predict(data), [expected], rtol=1e-14)


def test_ar1_negative_persistence_and_slope_clipping():
    model = AR1(ridge=0.1)
    # Paired means=1.5, covariance=-.5, variance=.25.
    slope = -0.5 / (0.5 + 0.1 * 0.25)
    expected = 1.5 + slope ** np.arange(1, 4) * (1 - 1.5)
    np.testing.assert_allclose(model.predict(window([1, 2, 1])), [expected], rtol=1e-14)
    clipped = AR1(ridge=1e-12, slope_limit=0.8)
    expected = 1.5 + (-0.8) ** np.arange(1, 4) * (1 - 1.5)
    np.testing.assert_allclose(clipped.predict(window([1, 2, 1])), [expected], rtol=1e-14)


@pytest.mark.parametrize("history", [[3, 3, 3], [1, 1, 4], [0, 0, 0, 0]])
def test_ar1_zero_centered_variance_predicts_the_paired_target_mean(history):
    np.testing.assert_array_equal(AR1().predict(window(history)), [[np.mean(history[1:])] * 3])


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0, 10.0])
def test_ar1_variance_relative_ridge_is_invariant_to_score_units(scale):
    history = np.array([1, 3, 2, 4], dtype=np.float64)
    # Paired means 2 and 3; covariance=-1, sum of squares=2, variance=2/3.
    slope = -1 / (2 + 1e-6 * (2 / 3))
    equilibrium = (3 - slope * 2) / (1 - slope)
    expected = equilibrium + slope ** np.arange(1, 4) * (4 - equilibrium)
    np.testing.assert_allclose(
        AR1().predict(window(history * scale)), [expected * scale], rtol=1e-14
    )


def test_ar1_horizon_does_not_change_earlier_predictions():
    history = [2, 4, 5.5, 6.625, 7.46875]
    np.testing.assert_array_equal(
        AR1().predict(window(history, horizon=3)),
        AR1().predict(window(history, horizon=7))[:, :3],
    )


def test_ar1_is_batched_read_only_and_can_forecast_negative_values():
    data = window([[10, 9, 8, 7], [1, 2, 1, 2]], horizon=10)
    before = data.y.copy()
    model = AR1()
    parameters = vars(model).copy()
    batch = model.predict(data)
    assert batch[0, -1] < 0
    for index in range(2):
        np.testing.assert_array_equal(batch[index], model.predict(window(data.y[index], 10))[0])
    np.testing.assert_array_equal(data.y, before)
    assert vars(model) == parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ridge": 0},
        {"ridge": -1},
        {"ridge": np.nan},
        {"ridge": np.inf},
        {"ridge": [1]},
        {"slope_limit": 0},
        {"slope_limit": 1},
        {"slope_limit": np.inf},
    ],
)
def test_ar1_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AR1(**kwargs)


@pytest.mark.parametrize("history", [[], [7], [2, 4]])
def test_ar1_requires_three_values_leaving_warmup_to_the_caller(history):
    with pytest.raises(ValueError, match="at least three"):
        AR1().predict(window(history))


def test_ar1_rejects_covariates():
    with pytest.raises(ValueError, match="covariates"):
        AR1().predict(window([1, 2, 3], x={"feature": np.ones((1, 3))}))
