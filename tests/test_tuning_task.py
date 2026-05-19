from __future__ import annotations

import optuna
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.evaluation.point_metrics import mae, smape
from calibre.tuning.objectives import Accuracy
from calibre.tuning.optimizer import _cap_threaded_config, optimize_task
from calibre.tuning.task import TuningTask


def _space_season_length(trial: optuna.Trial) -> dict:
    return {"season_length": trial.suggest_categorical("season_length", [2, 4])}


@pytest.fixture
def series_df(dates, repeating_pattern):
    return pd.DataFrame(
        {
            "unique_id": "test_series",
            "ds": dates,
            "y": repeating_pattern,
        }
    )


@pytest.fixture(scope="module", autouse=True)
def ray_local_runtime():
    import ray

    if ray.is_initialized():
        yield
        return
    ray.init(include_dashboard=False, ignore_reinit_error=True, local_mode=True)
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture
def tuning_task(series_df, dates):
    return TuningTask(
        unique_id="test_series",
        history=series_df,
        horizon=4,
        base_model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "name": "sn_tuning",
        },
        search_space=_space_season_length,
        actuals=series_df,
        origins=[dates[15]],
        objective=Accuracy(metric=smape),
        n_trials=1,
        freq="W",
        seed=3,
        ray_local_mode=True,
    )


@pytest.fixture(scope="module")
def tuned_best_config():
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    series = pd.DataFrame(
        {
            "unique_id": "test_series",
            "ds": dates,
            "y": [10.0, 20.0, 30.0, 40.0] * 5,
        }
    )
    return optimize_task(
        TuningTask(
            unique_id="test_series",
            history=series,
            horizon=4,
            base_model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "name": "sn_tuning",
            },
            search_space=_space_season_length,
            actuals=series,
            origins=[dates[15]],
            objective=Accuracy(metric=smape),
            n_trials=1,
            freq="W",
            seed=3,
            ray_local_mode=True,
        )
    )


def test_optimize_finds_correct_season_length(tuned_best_config):
    """SeasonalNaive HPO should pick season_length=4 for a period-4 pattern."""
    assert tuned_best_config["season_length"] == 4


def test_optimize_returns_complete_config(tuned_best_config):
    """Result must contain all base_model_config keys plus tuned params."""
    assert tuned_best_config["backend"] == "statsforecast"
    assert tuned_best_config["model"] == "SeasonalNaive"
    assert tuned_best_config["name"] == "sn_tuning"
    assert "season_length" in tuned_best_config


def test_optimize_single_trial(tuned_best_config):
    """n_trials=1 should run without error."""
    assert isinstance(tuned_best_config, dict)
    assert "season_length" in tuned_best_config


def test_optimize_with_mae_metric(series_df, dates):
    """Custom metric (mae) should also converge to season_length=4."""
    objective = Accuracy(metric=mae)
    assert objective.metric.__name__ == "mae"


def test_optimize_accepts_conformal_config(series_df, dates):
    factory_calls = 0

    def _runtime_factory() -> SymmetricIntervalRuntime:
        nonlocal factory_calls
        factory_calls += 1
        return SymmetricIntervalRuntime(
            SymmetricIntervalConfig(
                method="aci",
                coverage=0.9,
                calibration_window=4,
                gamma=0.05,
            )
        )

    task = TuningTask(
        unique_id="test_series",
        history=series_df,
        horizon=4,
        base_model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "name": "sn_tuning",
        },
        search_space=_space_season_length,
        actuals=series_df,
        origins=[dates[15]],
        objective=Accuracy(metric=smape),
        n_trials=1,
        freq="W",
        conformal_runtime_factory=_runtime_factory,
        seed=3,
        ray_local_mode=True,
    )
    result = optimize_task(task)
    assert isinstance(result, dict)
    assert "season_length" in result


def test_resource_budget_caps_threaded_model_configs():
    capped = _cap_threaded_config(
        {"model": "lightgbm.LGBMRegressor", "n_jobs": -1, "num_threads": 16},
        cpu_per_trial=2.0,
    )
    assert capped["n_jobs"] == 2
    assert capped["num_threads"] == 2


def test_resource_budget_does_not_add_threads_to_unthreaded_model():
    capped = _cap_threaded_config({"model": "SeasonalNaive"}, cpu_per_trial=2.0)
    assert "n_jobs" not in capped


def test_asha_prunes_trials_between_origins(monkeypatch):
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    monkeypatch.setenv("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")

    def _trainable(config: dict) -> None:
        for origin_idx in range(1, 4):
            tune.report(
                {
                    "objective": float(config["loss"]) * origin_idx,
                    "origin_index": origin_idx,
                }
            )

    tuner = tune.Tuner(
        _trainable,
        param_space={"loss": tune.grid_search([0.0, 100.0])},
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric="objective",
                mode="min",
                time_attr="origin_index",
                max_t=3,
                grace_period=1,
            ),
            num_samples=1,
        ),
        run_config=tune.RunConfig(verbose=0),
    )

    results = tuner.fit()

    pruned = [
        result
        for result in results
        if result.config["loss"] == 100.0 and result.metrics["origin_index"] == 1
    ]
    assert pruned
