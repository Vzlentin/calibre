import numpy as np
import pandas as pd
import pytest
from mlforecast.target_transforms import Differences
from neuralforecast.models import MLP
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor

from calibre import (
    CQR,
    BottomUp,
    Covariate,
    FittedQuantiles,
    Forecasts,
    Hierarchy,
    Identity,
    Panel,
    SplitQuantile,
    Step,
    Window,
    WlsStruct,
    replay,
    rolling_forecasts,
)
from calibre.forecast.models.base import check_quantiles
from calibre.forecast.models.mlforecast import MLForecast
from calibre.forecast.models.neuralforecast import NeuralForecast

CONFIG = {
    "input_size": 5,
    "hidden_size": 8,
    "num_layers": 1,
    "max_steps": 2,
    "val_check_steps": 2,
    "random_seed": 17,
    "accelerator": "cpu",
}


def data():
    values = np.random.default_rng(11).normal(size=(2, 40)).astype(np.float32)
    series = np.array(["a", "b"])
    periods = pd.date_range("2024-01-01", periods=40, freq="D")
    panel = Panel(series, periods, values, "D")
    return panel, Hierarchy.flat(series)


def window(panel, origin=20, horizon=3):
    seen = origin + 1
    return Window(panel.values[:, :seen], {}, panel.periods[: seen + horizon], horizon)


def ml():
    return MLForecast(
        LinearRegression(),
        lags=[1],
        quantile_models={
            0.1: QuantileRegressor(quantile=0.1, alpha=0.1),
            0.9: QuantileRegressor(quantile=0.9, alpha=0.1),
        },
    )


def neural():
    return NeuralForecast(MLP, CONFIG, quantiles=(0.1, 0.9))


@pytest.mark.parametrize(
    "levels",
    [
        (0, 0.9),
        (0.1, 1),
        (0.9, 0.1),
        (0.2, 0.2),
        (np.nan, 0.9),
        (0.1, np.inf),
        (0.1,),
        (0.1, 0.5, 0.9),
    ],
)
def test_quantile_levels_must_be_a_strict_finite_pair(levels):
    with pytest.raises(ValueError, match="quantiles"):
        check_quantiles(levels)
    with pytest.raises(ValueError, match="quantiles"):
        NeuralForecast(MLP, CONFIG, quantiles=levels)


@pytest.mark.parametrize("levels", [[], [0.1], [0.1, 0.5, 0.9], [0, 0.9], [0.1, 1]])
def test_mlforecast_rejects_invalid_quantile_model_mappings(levels):
    with pytest.raises(ValueError, match="quantiles"):
        MLForecast(
            DummyRegressor(), lags=[1], quantile_models={q: DummyRegressor() for q in levels}
        )


def test_mlforecast_direct_quantiles_match_independently_fitted_regressions():
    panel, _ = data()
    first = window(panel)
    model = ml()
    fitted = model.fit(first)
    assert isinstance(fitted, FittedQuantiles)
    got = fitted.predict_quantiles(first)
    expected = np.empty((2, 3, 2), dtype=np.float32)
    for h in range(3):
        x = panel.values[:, : 20 - h].reshape(-1, 1)
        y = panel.values[:, 1 + h : 21].reshape(-1)
        for j, q in enumerate([0.1, 0.9]):
            independent = QuantileRegressor(quantile=q, alpha=0.1).fit(x, y)
            expected[:, h, j] = independent.predict(panel.values[:, 20:21])
    np.testing.assert_allclose(got.interval, expected, atol=1e-6)
    assert got.levels == (0.1, 0.9)
    np.testing.assert_array_equal(got.point, fitted.predict(first))
    assert fitted.quantile_engine.ts.max_horizon == 3
    assert len(fitted.quantile_engine.models_["lower"]) == 3
    assert len(fitted.quantile_engine.models_["upper"]) == 3
    assert not hasattr(model.model, "coef_")
    assert all(not hasattr(estimator, "coef_") for estimator in model.quantile_models.values())


def test_mlforecast_each_horizon_has_its_own_training_targets_and_keeps_point_behavior():
    panel, _ = data()
    first = window(panel)
    model = MLForecast(
        DummyRegressor(),
        lags=[1],
        quantile_models={
            0.9: DummyRegressor(strategy="quantile", quantile=0.9),
            0.1: DummyRegressor(strategy="quantile", quantile=0.1),
        },
    )
    fitted = model.fit(first)
    assert isinstance(fitted, FittedQuantiles)
    expected = np.array(
        [np.quantile(panel.values[:, 1 + h : 21].ravel(), [0.1, 0.9]) for h in range(3)]
    )
    output = fitted.predict_quantiles(first)
    np.testing.assert_allclose(output.interval, np.broadcast_to(expected, (2, 3, 2)), atol=1e-6)
    np.testing.assert_allclose(output.point, panel.values[:, 1:21].mean(), atol=1e-6)
    np.testing.assert_array_equal(
        fitted.predict_quantiles(window(panel, 25)).interval, output.interval
    )


def test_mlforecast_quantiles_use_known_ahead_features_at_prediction():
    panel, hierarchy = data()
    promo = np.random.default_rng(12).normal(size=(2, 43)).astype(np.float32)
    values = 3 * promo[:, :40]
    panel = Panel(panel.series, panel.periods, values, "D")
    model = MLForecast(
        LinearRegression(),
        features=["promo"],
        quantile_models={q: QuantileRegressor(quantile=q, alpha=0) for q in [0.1, 0.9]},
    )
    forecasts = rolling_forecasts(
        panel,
        hierarchy,
        model,
        Identity(hierarchy),
        np.array([20, 25]),
        1,
        covariates={"promo": Covariate(promo, True, "sum")},
    )
    assert forecasts.interval is not None
    expected = np.stack([3 * promo[:, o + 1 : o + 2] for o in [20, 25]])
    np.testing.assert_allclose(forecasts.interval[..., 0], expected, atol=1e-5)
    np.testing.assert_allclose(forecasts.interval[..., 1], expected, atol=1e-5)


def test_mlforecast_native_quantiles_reject_arbitrary_target_transforms():
    with pytest.raises(ValueError, match="target_transforms"):
        MLForecast(
            DummyRegressor(),
            lags=[1],
            target_transforms=[Differences([1])],
            quantile_models={0.1: DummyRegressor(), 0.9: DummyRegressor()},
        )


@pytest.mark.parametrize(
    "levels,alias", [((0.1, 0.9), "named-network"), ((0.5, 0.9), None), ((0.10001, 0.10002), None)]
)
def test_neural_quantiles_and_median_match_vendor_outputs_without_rounded_label_collisions(
    levels, alias
):
    panel, _ = data()
    config = {**CONFIG, "alias": alias}
    model = NeuralForecast(MLP, config, quantiles=levels)
    first = window(panel)
    fitted = model.fit(first)
    assert isinstance(fitted, FittedQuantiles)
    got = fitted.predict_quantiles(first)
    qs = sorted({0.5, *levels})
    np.testing.assert_allclose(fitted.engine.models[0].loss.quantiles.detach().numpy(), qs)
    assert "loss" not in model.config
    assert got.levels == levels
    frame = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(2), 5),
            "ds": np.tile(panel.periods[16:21].to_numpy(), 2),
            "y": panel.values[:, 16:21].reshape(-1),
        }
    )
    raw = fitted.engine.predict(df=frame).sort_values(["unique_id", "ds"])
    prefix = alias if alias is not None else "MLP"
    expected = np.stack(
        [raw[prefix + f"-q{qs.index(q)}"].to_numpy().reshape(2, 3) for q in (0.5, *levels)], axis=-1
    )
    np.testing.assert_array_equal(got.point, expected[..., 0])
    np.testing.assert_array_equal(got.interval, expected[..., 1:])
    np.testing.assert_array_equal(fitted.predict(first), got.point)


@pytest.mark.parametrize("key", ["loss", "valid_loss"])
def test_neural_quantile_mode_does_not_silently_override_a_configured_loss(key):
    from neuralforecast.losses.pytorch import MAE

    with pytest.raises(ValueError, match="set by the adapter"):
        NeuralForecast(MLP, {**CONFIG, key: MAE()}, quantiles=(0.1, 0.9))


@pytest.mark.parametrize("factory", [ml, neural])
def test_native_forecasts_flow_through_cqr_and_never_use_later_training_data(factory):
    panel, hierarchy = data()
    origins = np.arange(20, 31)
    model = factory()
    forecasts = rolling_forecasts(panel, hierarchy, model, Identity(hierarchy), origins, 3)
    assert forecasts.points.shape == (11, 2, 3)
    assert forecasts.interval is not None
    assert forecasts.interval.shape == (11, 2, 3, 2)
    assert forecasts.interval.dtype == np.float32
    assert forecasts.quantile_levels == (0.1, 0.9)
    assert (forecasts.interval[..., 0] <= forecasts.interval[..., 1]).all()
    actuals = hierarchy.summing @ panel.values
    run = replay(forecasts, actuals, target=Step(), score=CQR(), calibrator=SplitQuantile(0.5))
    expected = np.maximum(
        forecasts.interval[..., 0] - run.target, run.target - forecasts.interval[..., 1]
    )
    np.testing.assert_array_equal(run.score, expected)
    assert np.isfinite(run.threshold[3:]).all()
    changed = panel.values.copy()
    changed[:, 27:] += 1000
    other = rolling_forecasts(
        Panel(panel.series, panel.periods, changed, "D"),
        hierarchy,
        factory(),
        Identity(hierarchy),
        origins,
        3,
    )
    assert other.interval is not None
    np.testing.assert_array_equal(forecasts.interval[origins < 27], other.interval[origins < 27])
    np.testing.assert_array_equal(forecasts.points[origins < 27], other.points[origins < 27])


@pytest.mark.parametrize("kwargs", [{"refit_every": 1}, {"initial_residuals": True}])
def test_native_rolling_forecasts_reject_mixing_model_generations_or_oob_seeds(kwargs):
    panel, hierarchy = data()
    with pytest.raises(ValueError, match="frozen model"):
        rolling_forecasts(
            panel, hierarchy, ml(), Identity(hierarchy), np.array([20, 25]), 2, **kwargs
        )


@pytest.mark.parametrize("reconciler", [BottomUp, WlsStruct])
def test_native_rolling_forecasts_do_not_reconcile_quantiles(reconciler):
    panel, hierarchy = data()
    with pytest.raises(ValueError, match="require Identity"):
        rolling_forecasts(panel, hierarchy, ml(), reconciler(hierarchy), np.array([20]), 2)


def test_native_quantiles_forecast_aggregate_nodes_directly_not_by_summing_quantiles():
    panel, _ = data()
    hierarchy = Hierarchy.from_attributes(
        panel.series, pd.DataFrame({"group": ["all", "all"]}, index=panel.series)
    )
    model = MLForecast(
        DummyRegressor(),
        lags=[1],
        quantile_models={q: DummyRegressor(strategy="quantile", quantile=q) for q in [0.1, 0.9]},
    )
    forecasts = rolling_forecasts(panel, hierarchy, model, Identity(hierarchy), np.array([20]), 2)
    assert forecasts.interval is not None
    actuals = hierarchy.aggregate(panel.values)
    assert actuals.shape[0] == 3
    for h in range(2):
        expected = np.quantile(actuals[:, 1 + h : 21].ravel(), [0.1, 0.9])
        np.testing.assert_allclose(
            forecasts.interval[0, :, h], np.broadcast_to(expected, (3, 2)), atol=1e-6
        )
    run = replay(forecasts, actuals, target=Step(), score=CQR(), calibrator=SplitQuantile(0.5))
    assert run.score.shape == (1, 3, 2)


def test_forecasts_orders_a_copy_of_native_intervals_and_keeps_metadata():
    raw = np.array([[[[9, -1], [2, 0]]]])
    forecasts = Forecasts(
        np.array([0]), np.zeros((1, 1, 2)), interval=raw, quantile_levels=(0.1, 0.9)
    )
    assert forecasts.interval is not None
    np.testing.assert_array_equal(forecasts.interval, [[[[-1, 9], [0, 2]]]])
    np.testing.assert_array_equal(raw, [[[[9, -1], [2, 0]]]])
    assert forecasts.interval.dtype == np.float32


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"interval": np.zeros((1, 1, 1, 2))}, "supplied together"),
        ({"quantile_levels": (0.1, 0.9)}, "supplied together"),
        ({"interval": np.zeros((1, 1, 2)), "quantile_levels": (0.1, 0.9)}, "shape"),
        ({"interval": np.full((1, 1, 1, 2), np.nan), "quantile_levels": (0.1, 0.9)}, "finite"),
        ({"interval": np.full((1, 1, 1, 2), np.inf), "quantile_levels": (0.1, 0.9)}, "finite"),
    ],
)
def test_forecasts_rejects_invalid_native_data(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Forecasts(np.array([0]), np.zeros((1, 1, 1)), **kwargs)
