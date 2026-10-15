import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from calibre import (
    Asymmetric,
    BottomUp,
    EnbPI,
    FitResiduals,
    Identity,
    SeasonalNaive,
    Step,
    Window,
    initial_state,
    replay,
    rolling_forecasts,
)
from calibre.backtest.forecasts import _reconciled_residuals
from calibre.forecast.models.bootstrap import BootstrapEnsemble
from tests.synthetic import make_panel


def window(history, horizon=1, x=None):
    history = np.asarray(history, dtype=np.float32).copy()
    history.flags.writeable = False
    return Window(
        history,
        {} if x is None else x,
        pd.date_range("2020-01-01", periods=history.shape[1] + horizon, freq="D"),
        horizon,
    )


def test_bootstrap_oob_means_and_extended_mean_center_have_independent_expected_values():
    history = np.array([[2, 7, 1, 12, 3, 18, 5, 23, 8, 30]], dtype=np.float32)
    estimator = DummyRegressor(strategy="mean")
    model = BootstrapEnsemble(estimator, n_lags=2, n_estimators=3, random_state=4)
    fitted = model.fit(window(history))
    y = history[0, 2:]
    # A mean regressor predicts the mean of its sampled targets, independently of X.
    bag_means = np.array([np.mean(y[bag], dtype=np.float64) for bag in fitted.bootstrap_indices])
    expected_oob = []
    for row in range(len(y)):
        excluded = np.array([row not in bag for bag in fitted.bootstrap_indices])
        np.testing.assert_array_equal(excluded, fitted.oob_mask[row])
        assert excluded.any()
        expected_oob.append(bag_means[excluded].mean())
    np.testing.assert_allclose(fitted.oob_predictions[:, 0], expected_oob)
    np.testing.assert_allclose(fitted.residuals().values[:, 0, 0], y - expected_oob, atol=1e-6)
    np.testing.assert_allclose(fitted.predict(window(history)), [[np.mean(expected_oob)]])
    assert abs(np.mean(expected_oob) - bag_means.mean()) > 1e-3
    np.testing.assert_allclose(fitted.ensemble_weights.sum(), 1)
    np.testing.assert_array_equal(fitted.residuals().origins, np.arange(1, 9))
    assert not hasattr(estimator, "constant_")
    assert np.all(np.diff(fitted.bootstrap_indices, axis=1) >= 0)


def test_bootstrap_covers_every_oob_row_using_extra_bags_and_fails_at_the_budget():
    history = window([[0, 1, 2]])
    with pytest.raises(RuntimeError, match="no OOB"):
        BootstrapEnsemble(DummyRegressor(), 1, 1, max_estimators=1).fit(history)
    fitted = BootstrapEnsemble(DummyRegressor(), 1, 1, random_state=3).fit(history)
    assert not fitted.oob_mask.all(axis=0).any()  # a bag always trains on some row
    assert fitted.oob_mask.any(axis=1).all()
    assert len(fitted.bootstrap_indices) > 1


def test_bootstrap_fits_independent_local_models_with_aligned_resamples():
    values = np.array([[0, 2, 1, 4, 3, 8, 5, 12], [5, 9, 7, 13, 11, 21, 15, 29]])
    fitted = BootstrapEnsemble(DummyRegressor(), 2, 5, random_state=3).fit(window(values))
    # The second series is exactly 2 * first + 5.
    np.testing.assert_allclose(fitted.oob_predictions[:, 1], 2 * fitted.oob_predictions[:, 0] + 5)
    points = fitted.predict(window(values))
    np.testing.assert_allclose(points[1], 2 * points[0] + 5)
    assert fitted.residuals().values.shape == (6, 2, 1)


def test_bootstrap_is_reproducible_and_prediction_does_not_refit_learners():
    history = window([[0, 2, 1, 4, 3, 8, 5, 12]])
    model = BootstrapEnsemble(DecisionTreeRegressor(max_depth=2), 2, 6, random_state=7)
    fitted, other = model.fit(history), model.fit(history)
    np.testing.assert_array_equal(fitted.bootstrap_indices, other.bootstrap_indices)
    np.testing.assert_array_equal(fitted.predict(history), other.predict(history))
    trees = [
        (tree.tree_.value.copy(), tree.tree_.threshold.copy()) for tree in fitted.estimators[0]
    ]
    fitted.predict(window([[0, 2, 1, 4, 3, 8, 5, 12, 100, -50]]))
    for tree, (values, thresholds) in zip(fitted.estimators[0], trees, strict=True):
        np.testing.assert_array_equal(tree.tree_.value, values)
        np.testing.assert_array_equal(tree.tree_.threshold, thresholds)
    for values in (
        fitted.bootstrap_indices,
        fitted.oob_mask,
        fitted.oob_predictions,
        fitted.ensemble_weights,
        fitted.residuals().values,
        fitted.residuals().origins,
    ):
        assert not values.flags.writeable


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_lags": 0}, "n_lags"),
        ({"n_lags": 2.5}, "n_lags"),
        ({"n_lags": True}, "n_lags"),
        ({"n_lags": 1, "n_estimators": 0}, "n_estimators"),
        ({"n_lags": 1, "n_estimators": 3, "max_estimators": 2}, "max_estimators"),
    ],
)
def test_bootstrap_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BootstrapEnsemble(DummyRegressor(), **kwargs)


def test_bootstrap_rejects_unsupported_or_invalid_inputs():
    model = BootstrapEnsemble(DummyRegressor(), 2)
    with pytest.raises(ValueError, match="horizon=1"):
        model.fit(window([[1, 2, 3, 4]], horizon=2))
    with pytest.raises(ValueError, match="covariates"):
        model.fit(window([[1, 2, 3, 4]], x={"unused": np.ones((1, 4))}))
    with pytest.raises(ValueError, match="at least 4"):
        model.fit(window([[1, 2, 3]]))
    with pytest.raises(ValueError, match="finite"):
        model.fit(window([[1, 2, np.nan, 4]]))
    with pytest.raises(TypeError):
        BootstrapEnsemble(SeasonalNaive(1), 1).fit(window([[1, 2, 3, 4]]))
    with pytest.raises(ValueError, match="constant"):
        BootstrapEnsemble(DummyRegressor(strategy="constant", constant=np.inf), 1).fit(
            window([[1, 2, 3, 4]])
        )


def test_fitted_bootstrap_rejects_a_past_or_incompatible_window():
    fitted = BootstrapEnsemble(DummyRegressor(), 1).fit(window([[1, 2, 3, 4]]))
    with pytest.raises(ValueError, match="at least 4"):
        fitted.predict(window([[1, 2, 3]]))
    with pytest.raises(ValueError, match="series"):
        fitted.predict(window([[1, 2, 3, 4], [1, 2, 3, 4]]))


def test_rolling_bootstrap_collects_reconciled_initial_errors_and_remains_causal():
    panel, hierarchy = make_panel(2, 50)
    model = BootstrapEnsemble(LinearRegression(), 3, 8, random_state=6)
    origins = np.arange(25, 40)
    forecasts = rolling_forecasts(
        panel, hierarchy, model, BottomUp(hierarchy), origins, 1, initial_residuals=True
    )
    fitted = model.fit(window(panel.values[:, :26]))
    expected = hierarchy.summing @ fitted.residuals().values[..., 0].T
    assert forecasts.initial_residuals is not None
    np.testing.assert_allclose(forecasts.initial_residuals.values[..., 0], expected.T)
    assert forecasts.initial_residuals.origins[-1] + 1 == origins[0]
    changed = panel.values.copy()
    changed[:, 35:] += 100
    later = type(panel)(panel.series, panel.periods, changed, panel.freq)
    other = rolling_forecasts(
        later, hierarchy, model, BottomUp(hierarchy), origins, 1, initial_residuals=True
    )
    assert other.initial_residuals is not None
    np.testing.assert_array_equal(forecasts.points[origins < 35], other.points[origins < 35])
    np.testing.assert_array_equal(
        forecasts.initial_residuals.values, other.initial_residuals.values
    )


def test_initial_residuals_require_a_supported_frozen_model():
    panel, hierarchy = make_panel(1, 12)
    with pytest.raises(ValueError, match="without refit"):
        rolling_forecasts(
            panel,
            hierarchy,
            SeasonalNaive(1),
            Identity(hierarchy),
            np.arange(3, 8),
            1,
            refit_every=2,
            initial_residuals=True,
        )
    with pytest.raises(TypeError, match="FittedResiduals"):
        rolling_forecasts(
            panel,
            hierarchy,
            SeasonalNaive(1),
            Identity(hierarchy),
            np.arange(3, 8),
            1,
            initial_residuals=True,
        )
    n_nodes = len(hierarchy.nodes)
    with pytest.raises(ValueError, match="known"):
        _reconciled_residuals(
            FitResiduals(np.array([3]), np.ones((1, n_nodes, 1))), Identity(hierarchy), 3, 1
        )
    with pytest.raises(ValueError, match="shape"):
        _reconciled_residuals(
            FitResiduals(np.array([1]), np.ones((1, n_nodes + 1, 1))), Identity(hierarchy), 3, 1
        )


def test_bootstrap_and_enbpi_work_end_to_end_through_existing_origin_loops():
    panel, hierarchy = make_panel(2, 65, seed=4)
    forecasts = rolling_forecasts(
        panel,
        hierarchy,
        BootstrapEnsemble(LinearRegression(), 3, 12, random_state=9),
        BottomUp(hierarchy),
        np.arange(30, 60),
        1,
        initial_residuals=True,
    )
    assert forecasts.initial_residuals is not None
    errors = forecasts.initial_residuals.values
    method = EnbPI(0.9, window=len(errors))
    state = initial_state(Step(), method, len(hierarchy.nodes), 1)
    state["calibrator"] = method.seed(state["calibrator"], errors)
    run = replay(
        forecasts,
        hierarchy.aggregate(panel.values),
        target=Step(),
        score=Asymmetric(),
        calibrator=method,
        state=state,
    )
    assert np.isfinite(run.lower).all() and np.isfinite(run.upper).all()
    assert (run.lower <= run.upper).all()
    assert run.point.shape == (30, len(hierarchy.nodes), 1)
    assert run.threshold.shape == (30, len(hierarchy.nodes), 1, 2)
    np.testing.assert_array_equal(run.lower, run.point + run.threshold[..., 0])
    np.testing.assert_array_equal(run.upper, run.point + run.threshold[..., 1])


@pytest.mark.parametrize(
    ("origins", "values", "match"),
    [
        (np.array([1.5]), np.ones((1, 1, 1)), "indexes"),
        (np.array([2, 1]), np.ones((2, 1, 1)), "indexes"),
        (np.array([1]), np.ones((1, 1)), "matching"),
        (np.array([1]), np.full((1, 1, 1), np.inf), "finite"),
    ],
)
def test_fit_residuals_validate_the_array_contract(origins, values, match):
    with pytest.raises(ValueError, match=match):
        FitResiduals(origins, values)
