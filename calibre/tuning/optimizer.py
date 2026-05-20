"""Tuning optimizer: runs Ray Tune studies for TuningTasks."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Any
import warnings

import optuna
import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID, Y_HAT, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ConformalOptions, ExecutionOptions
from calibre.execution.io import join_uri
from calibre.execution.ray_runtime import acquire_ray_runtime, prepare_ray_environment
from calibre.tuning.task import TuningTask

_OBJECTIVE_METRIC = "objective"
_ORIGIN_INDEX = "origin_index"
_DEFAULT_TUNE_RESULTS_SUBDIR = "ray_tune"


def create_tpe_sampler(seed: int | None) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(seed=seed)


def _available_cpus() -> int:
    return max(1, os.cpu_count() or 1)


def _resolved_max_concurrent_trials(task: TuningTask) -> int:
    if task.max_concurrent_trials is not None:
        return max(1, int(task.max_concurrent_trials))
    cpu_per_trial = max(float(task.cpu_per_trial), 1e-9)
    return max(1, int(_available_cpus() // cpu_per_trial))


def _thread_budget(cpu_per_trial: float) -> int:
    return max(1, int(cpu_per_trial))


def _normalize_tune_storage_path(path: str) -> str:
    if "://" in path:
        return path
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _resolve_tune_storage_path(task: TuningTask) -> str:
    if task.tune_storage_path is not None:
        return _normalize_tune_storage_path(task.tune_storage_path)
    if env_storage_path := os.environ.get("RAYTUNE_RESULTS_DIR"):
        return _normalize_tune_storage_path(env_storage_path)
    if task.results_dir is not None:
        return _normalize_tune_storage_path(
            join_uri(task.results_dir, _DEFAULT_TUNE_RESULTS_SUBDIR)
        )
    return tempfile.mkdtemp(prefix="calibre-tune-")


def _cap_threaded_config(config: dict[str, Any], cpu_per_trial: float) -> dict[str, Any]:
    """Keep library-level parallelism inside the Tune trial CPU budget."""
    capped = dict(config)
    threads = _thread_budget(cpu_per_trial)
    for key in ("n_jobs", "num_threads", "nthread"):
        if key not in capped:
            continue
        value = capped.get(key)
        if value is None or int(value) < 1 or int(value) > threads:
            capped[key] = threads
    model_name = str(capped.get("model", "")).lower()
    if "lgbm" in model_name or "lightgbm" in model_name or "xgb" in model_name:
        capped.setdefault("n_jobs", threads)
    return capped


@contextmanager
def restore_cwd():
    """Ray Tune trials chdir into a per-trial working dir and don't always restore it."""
    original = os.getcwd()
    try:
        yield
    finally:
        try:
            if os.getcwd() != original:
                os.chdir(original)
        except FileNotFoundError:
            os.chdir(original)


@contextmanager
def _trial_thread_env(cpu_per_trial: float):
    threads = str(_thread_budget(cpu_per_trial))
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = threads
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _history_with_uid(task: TuningTask) -> pd.DataFrame:
    history = task.history.copy()
    if UNIQUE_ID not in history.columns:
        history.insert(0, UNIQUE_ID, task.unique_id)
    return history


def _build_mlflow_callbacks(task: TuningTask) -> list[Any]:
    if task.mlflow_tracking_uri is None and task.mlflow_experiment_name is None:
        return []
    from ray.air.integrations.mlflow import MLflowLoggerCallback

    tags: dict[str, str] = {}
    if task.mlflow_parent_run_id is not None:
        tags["calibre_parent_run_id"] = task.mlflow_parent_run_id
    return [
        MLflowLoggerCallback(
            tracking_uri=task.mlflow_tracking_uri,
            experiment_name=task.mlflow_experiment_name,
            tags=tags or None,
            save_artifact=True,
            log_params_on_trial_end=True,
        )
    ]


def _best_result_config(results: Any) -> dict[str, Any]:
    valid_results = [
        result
        for result in results
        if result.error is None
        and result.metrics is not None
        and _OBJECTIVE_METRIC in result.metrics
        and isfinite(float(result.metrics[_OBJECTIVE_METRIC]))
    ]
    if not valid_results:
        failed = sum(1 for result in results if result.error is not None)
        raise RuntimeError(
            "Ray Tune completed without a valid objective result "
            f"({failed} failed trial(s)). Check trial logs and model/search-space settings."
        )
    best = results.get_best_result(
        metric=_OBJECTIVE_METRIC,
        mode="min",
        filter_nan_and_inf=True,
    )
    return dict(best.config)


def _validate_task(task: TuningTask) -> list[pd.Timestamp]:
    if not task.origins:
        raise ValueError("TuningTask.origins must contain at least one origin")
    if task.asha_grace_period < 1:
        raise ValueError("TuningTask.asha_grace_period must be at least 1")
    if len(task.origins) > 1 and task.asha_grace_period >= len(task.origins):
        raise ValueError("TuningTask.asha_grace_period must be less than the number of origins")
    if task.cpu_per_trial <= 0:
        raise ValueError("TuningTask.cpu_per_trial must be positive")
    return [pd.Timestamp(origin) for origin in task.origins]


def _evaluate_candidate(
    task: TuningTask,
    config: dict[str, Any],
    origins: list[pd.Timestamp],
) -> float:
    history = _history_with_uid(task)
    candidate_config = _cap_threaded_config(
        {**task.base_model_config, **config, "freq": task.freq},
        task.cpu_per_trial,
    )
    forecast_task = ForecastTask(
        history=history,
        horizon=task.horizon,
        model_config=candidate_config,
    )
    conformal_options = (
        ConformalOptions(runtime=task.conformal_runtime_factory())
        if task.conformal_runtime_factory is not None
        else ConformalOptions()
    )
    with BackendEngine(
        execution=ExecutionOptions(
            freq=task.freq,
            backend="local",
            max_concurrency=task.max_uid_concurrency,
        ),
        conformal=conformal_options,
    ) as engine:
        value = float("inf")
        with _trial_thread_env(task.cpu_per_trial):
            for result in engine.iter_origins([forecast_task], task.actuals, origins):
                resolved = result.ledger.to_df().dropna(subset=[Y, Y_HAT])
                value = (
                    float("inf")
                    if resolved.empty
                    else float(task.objective.evaluate(resolved, resolved[Y]))
                )
        return value


def _optimize_task_sequential(task: TuningTask, origins: list[pd.Timestamp]) -> dict[str, Any]:
    study = optuna.create_study(direction="minimize", sampler=create_tpe_sampler(task.seed))

    def _objective(trial: optuna.Trial) -> float:
        config = task.search_space(trial)
        trial.set_user_attr("resolved_config", dict(config))
        return _evaluate_candidate(task, config, origins)

    study.optimize(_objective, n_trials=task.n_trials, gc_after_trial=True)
    if not study.trials or study.best_trial.value is None or not isfinite(study.best_trial.value):
        raise RuntimeError("Sequential Optuna completed without a valid objective result")
    best_config = study.best_trial.user_attrs.get("resolved_config", study.best_trial.params)
    return {**task.base_model_config, **dict(best_config)}


def optimize_task(task: TuningTask) -> dict:
    """Run HPO and return the best model_config dict."""
    origins = _validate_task(task)

    if task.conformal_runtime_factory is not None and not task.ray_local_mode:
        warnings.warn(
            "conformal_runtime_factory is set: falling back to sequential Optuna "
            "instead of Ray Tune. Set ray_local_mode=True to use Ray Tune with conformal.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _optimize_task_sequential(task, origins)

    prepare_ray_environment()

    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    history = _history_with_uid(task)
    max_t = len(origins)
    max_concurrent_trials = _resolved_max_concurrent_trials(task)
    search_alg = OptunaSearch(
        space=task.search_space,
        metric=_OBJECTIVE_METRIC,
        mode="min",
        sampler=create_tpe_sampler(task.seed),
    )
    scheduler = ASHAScheduler(
        metric=_OBJECTIVE_METRIC,
        mode="min",
        time_attr=_ORIGIN_INDEX,
        max_t=max_t,
        grace_period=task.asha_grace_period,
    )

    def _trainable(config: dict[str, Any]) -> None:
        candidate_config = _cap_threaded_config(
            {**task.base_model_config, **config, "freq": task.freq},
            task.cpu_per_trial,
        )
        forecast_task = ForecastTask(
            history=history,
            horizon=task.horizon,
            model_config=candidate_config,
        )
        conformal_options = (
            ConformalOptions(runtime=task.conformal_runtime_factory())
            if task.conformal_runtime_factory is not None
            else ConformalOptions()
        )
        engine = BackendEngine(
            execution=ExecutionOptions(
                freq=task.freq,
                backend="local",
                max_concurrency=task.max_uid_concurrency,
            ),
            conformal=conformal_options,
        )
        try:
            with _trial_thread_env(task.cpu_per_trial):
                for origin_idx, result in enumerate(
                    engine.iter_origins([forecast_task], task.actuals, origins),
                    start=1,
                ):
                    resolved = result.ledger.to_df().dropna(subset=[Y, Y_HAT])
                    value = (
                        float("inf")
                        if resolved.empty
                        else float(task.objective.evaluate(resolved, resolved[Y]))
                    )
                    tune.report({_OBJECTIVE_METRIC: value, _ORIGIN_INDEX: origin_idx})
        finally:
            engine.close()

    trainable = tune.with_resources(
        _trainable,
        resources={"cpu": float(task.cpu_per_trial)},
    )
    run_config_kwargs: dict[str, Any] = {
        "callbacks": _build_mlflow_callbacks(task),
        "storage_path": _resolve_tune_storage_path(task),
    }
    if task.tune_experiment_name is not None:
        run_config_kwargs["name"] = task.tune_experiment_name

    ray_runtime = acquire_ray_runtime(
        address=task.ray_address,
        local_mode=task.ray_local_mode,
    )
    previous_auto_loggers = os.environ.get("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS")
    os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
    previous_chdir = os.environ.get("RAY_CHDIR_TO_TRIAL_DIR")
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    try:
        with restore_cwd():
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    search_alg=search_alg,
                    scheduler=scheduler,
                    num_samples=task.n_trials,
                    max_concurrent_trials=max_concurrent_trials,
                ),
                run_config=tune.RunConfig(**run_config_kwargs),
            )
            results = tuner.fit()
    finally:
        if previous_chdir is None:
            os.environ.pop("RAY_CHDIR_TO_TRIAL_DIR", None)
        else:
            os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = previous_chdir
        if previous_auto_loggers is None:
            os.environ.pop("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", None)
        else:
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = previous_auto_loggers
        ray_runtime.release()
    return {**task.base_model_config, **_best_result_config(results)}
