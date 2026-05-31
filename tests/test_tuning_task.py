from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import optuna
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.evaluation.point_metrics import mae, pinball_linear, smape
from calibre.execution.threading import cap_threaded_config, thread_budget
from calibre.tuning import optimizer
from calibre.tuning.objectives import Accuracy
from calibre.tuning.optimizer import _resolve_tune_storage_path, optimize_task, run_optuna_study
from calibre.tuning.task import StudyConfig, TuningCandidate, TuningTask


def pinball_loss(actual, predicted):
    return pinball_linear(actual, predicted, tau=0.5)


def _space_season_length(trial: optuna.Trial) -> TuningCandidate:
    return TuningCandidate(
        model_config={"season_length": trial.suggest_categorical("season_length", [2, 4])}
    )


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
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        local_mode=True,
        _skip_env_hook=True,
    )
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
        study_config=StudyConfig(n_trials=1, freq="W", seed=3),
    )


@pytest.fixture(scope="module", params=[pinball_loss, mae], ids=["pinball_loss", "mae"])
def tuned_best_config(request):
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
            objective=Accuracy(metric=request.param),
            study_config=StudyConfig(n_trials=1, freq="W", seed=3),
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


def test_optimize_task_from_tmp_cwd_does_not_use_cwd_as_tune_uri(
    monkeypatch, tmp_path, tuning_task
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAYTUNE_RESULTS_DIR", raising=False)

    result = optimize_task(
        replace(tuning_task, study_config=replace(tuning_task.study_config, results_dir="results"))
    )

    assert "season_length" in result


def test_optimize_converges_for_metric(tuned_best_config):
    """Both configured metrics should run end-to-end and pick season_length=4."""
    assert tuned_best_config["season_length"] == 4


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
        conformal_runtime_factory=_runtime_factory,
        study_config=StudyConfig(n_trials=1, freq="W", seed=3),
    )
    result = optimize_task(task)
    assert isinstance(result, dict)
    assert "season_length" in result


def test_resource_budget_caps_threaded_model_configs():
    capped = cap_threaded_config(
        {"model": "lightgbm.LGBMRegressor", "n_jobs": -1, "num_threads": 16},
        cpu_budget=2.0,
    )
    assert capped["n_jobs"] == 2
    assert capped["num_threads"] == 2


def test_resource_budget_does_not_add_threads_to_unthreaded_model():
    capped = cap_threaded_config({"model": "SeasonalNaive"}, cpu_budget=2.0)
    assert "n_jobs" not in capped


def test_score_forecast_task_caps_native_threads_via_threadpoolctl(monkeypatch):
    """Scoring caps BLAS/OpenMP pools to the CPU budget via ``threadpool_limits``
    without mutating the process-global thread-count env vars (REVIEW #7)."""
    recorded: list[int] = []

    @contextmanager
    def _spy_limits(*, limits):
        recorded.append(limits)
        yield

    monkeypatch.setattr(optimizer, "threadpool_limits", _spy_limits)

    class _EmptyEngine:
        def iter_origins(self, tasks, actuals, origins):
            return iter(())

    thread_keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    )
    before = {key: os.environ.get(key) for key in thread_keys}

    cost = optimizer._score_forecast_task(
        engine=_EmptyEngine(),
        forecast_task=None,
        objective=None,
        actuals=None,
        origins=[],
        cpu_per_trial=2.0,
    )

    assert cost == 0.0
    assert recorded == [thread_budget(2.0)]
    assert {key: os.environ.get(key) for key in thread_keys} == before


def test_default_tune_storage_path_stays_under_results_dir_when_home_unwritable(
    monkeypatch, tmp_path, tuning_task
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAYTUNE_RESULTS_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-writable-home"))

    storage_path = Path(_resolve_tune_storage_path(tuning_task))

    assert storage_path.is_absolute()
    assert storage_path.is_dir()
    assert storage_path.is_relative_to(tmp_path / "results")


def test_run_optuna_study_rejects_grace_period_not_less_than_max_t():
    with pytest.raises(ValueError, match="asha_grace_period must be less than max_t"):
        run_optuna_study(
            space=lambda trial: None,
            trainable=lambda config, **kwargs: None,
            n_trials=1,
            max_t=2,
            seed=None,
            asha_grace_period=2,
            cpu_per_trial=1.0,
            max_concurrent_trials=1,
            ray_address=None,
            ray_local_mode=False,
            tune_storage_path="unused",
        )


def test_asha_prunes_trials_between_origins(monkeypatch, tmp_path):
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    monkeypatch.setenv("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
    monkeypatch.setenv("RAY_CHDIR_TO_TRIAL_DIR", "0")

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
            metric="objective",
            mode="min",
            scheduler=ASHAScheduler(
                time_attr="origin_index",
                max_t=3,
                grace_period=1,
            ),
            num_samples=1,
        ),
        run_config=tune.RunConfig(storage_path=str(tmp_path / "asha-tune"), verbose=0),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pickle, copy, and deepcopy support will be removed from itertools",
            category=DeprecationWarning,
            module="ray.cloudpickle.cloudpickle",
        )
        results = tuner.fit()

    pruned = [
        result
        for result in results
        if result.config["loss"] == 100.0 and result.metrics["origin_index"] == 1
    ]
    assert pruned
