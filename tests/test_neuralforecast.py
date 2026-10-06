import numpy as np
import pandas as pd
import pytest
from neuralforecast.models import MLP

from calibre.backtest import rolling_forecasts
from calibre.data import Hierarchy, Panel
from calibre.forecast import Covariate
from calibre.forecast.models.neuralforecast import NeuralForecast
from calibre.forecast.reconcile import BottomUp

ORIGINS = np.array([30, 35])
HORIZON = 3
CONFIG = {
    "input_size": 7,
    "hidden_size": 8,
    "num_layers": 1,
    "max_steps": 5,
    "val_check_steps": 5,
    "random_seed": 0,
    "accelerator": "cpu",
}


def panel() -> tuple[Panel, Hierarchy]:
    rng = np.random.default_rng(0)
    values = rng.poisson(5.0, size=(3, 40)).astype(np.float32)
    series = np.array(["a", "b", "c"])
    periods = pd.date_range("2024-01-01", periods=40, freq="D")
    return Panel(series, periods, values, "D"), Hierarchy.flat(series)


def run(model: NeuralForecast, covariates: dict | None = None) -> np.ndarray:
    data, hierarchy = panel()
    return rolling_forecasts(
        data, hierarchy, model, BottomUp(hierarchy), ORIGINS, HORIZON, covariates=covariates
    ).points


def promo(values: np.ndarray) -> dict[str, Covariate]:
    return {"promo": Covariate(values, known_ahead=True, aggregate="sum")}


def test_a_seeded_network_gives_the_same_finite_points_twice():
    first = run(NeuralForecast(MLP, CONFIG))
    assert first.shape == (2, 3, HORIZON) and np.isfinite(first).all()
    np.testing.assert_array_equal(run(NeuralForecast(MLP, CONFIG)), first)


def test_known_ahead_values_reach_the_model_only_through_the_horizon():
    values = np.random.default_rng(1).normal(size=(3, 50)).astype(np.float32)
    model = NeuralForecast(MLP, CONFIG, features=["promo"], fit_periods=20)
    base = run(model, promo(values))
    after = values.copy()
    after[:, ORIGINS[-1] + 1 + HORIZON :] = 100
    np.testing.assert_array_equal(run(model, promo(after)), base)
    inside = values.copy()
    inside[:, ORIGINS[-1] + 1] = 100
    assert not np.array_equal(run(model, promo(inside))[-1], base[-1])


def test_historical_and_static_features_are_used():
    rng = np.random.default_rng(2)
    covariates = {
        "weather": Covariate(rng.normal(size=(3, 40)), known_ahead=False, aggregate="mean"),
        "size": Covariate([[1.0], [2.0], [3.0]], known_ahead=False, aggregate="sum"),
    }
    model = NeuralForecast(MLP, CONFIG, features=["weather", "size"])
    base = run(model, covariates)
    covariates["size"] = Covariate([[1.0], [2.0], [9.0]], known_ahead=False, aggregate="sum")
    assert not np.array_equal(run(model, covariates), base)


@pytest.mark.parametrize(
    ("config", "kwargs", "match"),
    [
        ({**CONFIG, "h": 5}, {}, r"\['h'\] are set by the adapter"),
        ({"max_steps": 5}, {}, "input_size"),
        (CONFIG, {"fit_periods": 0}, "fit_periods"),
    ],
)
def test_neuralforecast_rejects_invalid_settings(config, kwargs, match):
    with pytest.raises(ValueError, match=match):
        NeuralForecast(MLP, config, **kwargs)
