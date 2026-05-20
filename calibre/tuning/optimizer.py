"""Tuning optimizer: runs Ray Tune studies for TuningTasks."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID, Y_HAT, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ConformalOptions, ExecutionOptions
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


def _is_uri(path: str) -> bool:
    return "://" in path


def _join_storage_root(root: str, child: str) -> str:
    if _is_uri(root):
        return f"{root.rstrip('/')}/{child}"
    return str(Path(root) / child)


def _absolute_local_storage_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _normalize_tune_storage_path(path: str) -> str:
    if _is_uri(path):
        return path
    return _absolute_local_storage_path(path)


def _resolve_tune_storage_path(task: TuningTask) -> str:
    if task.tune_storage_path is not None:
        return _normalize_tune_storage_path(task.tune_storage_path)
    if env_storage_path := os.environ.get("RAYTUNE_RESULTS_DIR"):
        return _normalize_tune_storage_path(env_storage_path)
    if task.results_dir is not None:
        return _normalize_tune_storage_path(
            _join_storage_root(task.results_dir, _DEFAULT_TUNE_RESULTS_SUBDIR)
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
    best = results.get_best_result(metric=_OBJECTIVE_METRIC, mode="min")
    return dict(best.config)


def optimize_task(task: TuningTask) -> dict:
    """Run HPO via Ray Tune and return the best model_config dict."""
    if not task.origins:
        raise ValueError("TuningTask.origins must contain at least one origin")
    if task.asha_grace_period < 1:
        raise ValueError("TuningTask.asha_grace_period must be at least 1")
    if task.cpu_per_trial <= 0:
        raise ValueError("TuningTask.cpu_per_trial must be positive")

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    history = _history_with_uid(task)
    origins = [pd.Timestamp(origin) for origin in task.origins]
    max_t = len(origins)
    grace_period = min(task.asha_grace_period, max_t)
    max_concurrent_trials = _resolved_max_concurrent_trials(task)
    search_alg = OptunaSearch(
        space=task.search_space,
        metric=_OBJECTIVE_METRIC,
        mode="min",
        sampler=create_tpe_sampler(task.seed),
    )
    scheduler = ASHAScheduler(
        time_attr=_ORIGIN_INDEX,
        max_t=max_t,
        grace_period=grace_period,
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

    ray_was_initialized = ray.is_initialized()
    if not ray_was_initialized:
        if task.ray_address is not None:
            ray.init(address=task.ray_address, ignore_reinit_error=True)
        else:
            ray.init(
                include_dashboard=False,
                ignore_reinit_error=True,
                local_mode=task.ray_local_mode,
            )
    previous_auto_loggers = os.environ.get("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS")
    os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
    try:
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                metric=_OBJECTIVE_METRIC,
                mode="min",
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=task.n_trials,
                max_concurrent_trials=max_concurrent_trials,
            ),
            run_config=tune.RunConfig(**run_config_kwargs),
        )
        results = tuner.fit()
    finally:
        if previous_auto_loggers is None:
            os.environ.pop("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", None)
        else:
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = previous_auto_loggers
        if not ray_was_initialized and task.ray_address is None:
            ray.shutdown()
    return {**task.base_model_config, **_best_result_config(results)}
