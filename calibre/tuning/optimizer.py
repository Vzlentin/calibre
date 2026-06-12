"""Tuning optimizer: runs Ray Tune studies for LocalTuningTasks."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import optuna
import pandas as pd
from threadpoolctl import threadpool_limits

from calibre.conformal.runtime import (
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
    to_json_safe_state,
)
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.io import join_uri
from calibre.execution.backend import BackendEngine, ConformalOptions, ExecutionOptions
from calibre.execution.ray_runtime import acquire_ray_runtime, prepare_ray_environment
from calibre.execution.task_builder import partition_tasks
from calibre.execution.threading import cap_threaded_config, thread_budget
from calibre.tuning.task import GlobalTuningTask, LocalTuningTask, StudyConfig, TuningCandidate

if TYPE_CHECKING:
    from ray import ObjectRef
    from ray.tune import Callback, Result, ResultGrid

OBJECTIVE_METRIC = "objective"
ORIGIN_INDEX = "origin_index"
_DEFAULT_TUNE_RESULTS_SUBDIR = "ray_tune"
_FORECAST_KEY_COLUMNS = [UNIQUE_ID, DS, FORECAST_ORIGIN, MODEL_NAME, H]

TuningTask = LocalTuningTask | GlobalTuningTask


@dataclass(frozen=True, slots=True)
class _ConformalRuntimeSnapshot:
    config: SymmetricIntervalConfig
    state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StudyOutcome:
    best_config: dict[str, Any]
    results: ResultGrid


def create_tpe_sampler(seed: int | None) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(seed=seed)


def _available_cpus() -> int:
    return max(1, os.cpu_count() or 1)


def _resolve_max_concurrent_trials(
    max_concurrent_trials: int | None,
    cpu_per_trial: float,
) -> int:
    if max_concurrent_trials is not None:
        return max(1, int(max_concurrent_trials))
    cpu_per_trial = max(float(cpu_per_trial), 1e-9)
    return max(1, int(_available_cpus() // cpu_per_trial))


def _normalize_tune_storage_path(path: str) -> str:
    if "://" in path:
        return path
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _resolve_tune_storage_path_values(
    tune_storage_path: str | None,
    results_dir: str | None,
) -> str:
    if tune_storage_path is not None:
        return _normalize_tune_storage_path(tune_storage_path)
    if env_storage_path := os.environ.get("RAYTUNE_RESULTS_DIR"):
        return _normalize_tune_storage_path(env_storage_path)
    if results_dir is not None:
        return _normalize_tune_storage_path(join_uri(results_dir, _DEFAULT_TUNE_RESULTS_SUBDIR))
    return tempfile.mkdtemp(prefix="calibre-tune-")


def _resolve_tune_storage_path_config(config: StudyConfig) -> str:
    return _resolve_tune_storage_path_values(config.tune_storage_path, config.results_dir)


def _resolve_tune_storage_path(task: LocalTuningTask) -> str:
    return _resolve_tune_storage_path_config(task.study_config)


@contextmanager
def _restore_cwd():
    """Ray Tune trials chdir into a per-trial working dir and don't always restore it."""
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)


@contextmanager
def _ray_tune_env():
    previous_auto_loggers = os.environ.get("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS")
    os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
    previous_chdir = os.environ.get("RAY_CHDIR_TO_TRIAL_DIR")
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    try:
        yield
    finally:
        if previous_chdir is None:
            os.environ.pop("RAY_CHDIR_TO_TRIAL_DIR", None)
        else:
            os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = previous_chdir
        if previous_auto_loggers is None:
            os.environ.pop("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", None)
        else:
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = previous_auto_loggers


def _history_with_uid(task: LocalTuningTask) -> pd.DataFrame:
    history = task.history.copy()
    if UNIQUE_ID not in history.columns:
        history.insert(0, UNIQUE_ID, task.unique_id)
    return history


def _build_mlflow_callbacks(config: StudyConfig) -> list[Callback]:
    if config.mlflow_tracking_uri is None and config.mlflow_experiment_name is None:
        return []
    from ray.air.integrations.mlflow import MLflowLoggerCallback

    tags: dict[str, str] = {}
    if config.mlflow_parent_run_id is not None:
        tags["calibre_parent_run_id"] = config.mlflow_parent_run_id
    return [
        MLflowLoggerCallback(
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment_name,
            tags=tags or None,
            save_artifact=True,
            log_params_on_trial_end=True,
        )
    ]


def _snapshot_conformal_runtime(task: LocalTuningTask) -> _ConformalRuntimeSnapshot | None:
    if task.conformal_runtime_factory is None:
        return None
    seed_runtime = task.conformal_runtime_factory()
    if not isinstance(seed_runtime, SymmetricIntervalRuntime):
        raise TypeError(
            "Ray Tune conformal optimization requires conformal_runtime_factory "
            "to return SymmetricIntervalRuntime"
        )
    return _ConformalRuntimeSnapshot(
        config=seed_runtime.config,
        state=to_json_safe_state(seed_runtime.get_resume_state()),
    )


def _resolve_state_ref(
    state_ref: ObjectRef[dict[str, Any]] | dict[str, Any] | None,
) -> dict[str, Any]:
    import ray

    if state_ref is None:
        raise ValueError("state_ref is required to resolve conformal state")
    if isinstance(state_ref, ray.ObjectRef):
        return ray.get(state_ref)
    return dict(state_ref)


def _newly_resolved_frame(
    ledger_df: pd.DataFrame,
    seen_keys: set[tuple[Any, ...]],
) -> pd.DataFrame:
    resolved = ledger_df.dropna(subset=[Y, Y_HAT])
    if resolved.empty:
        return resolved

    missing = [col for col in _FORECAST_KEY_COLUMNS if col not in resolved.columns]
    if missing:
        raise ValueError(f"Resolved ledger is missing key columns: {missing}")

    keys = list(resolved[_FORECAST_KEY_COLUMNS].itertuples(index=False, name=None))
    mask = [key not in seen_keys for key in keys]
    for key, keep in zip(keys, mask, strict=True):
        if keep:
            seen_keys.add(key)
    return resolved.loc[mask].copy()


def _objective_contribution_with(
    objective: Any,
    ledger_df: pd.DataFrame,
    seen_keys: set[tuple[Any, ...]],
) -> float:
    resolved = _newly_resolved_frame(ledger_df, seen_keys)
    if resolved.empty:
        return float("inf")
    return float(objective.evaluate(resolved, resolved[Y]))


def _best_result_config(
    results: ResultGrid,
    *,
    metric: str = OBJECTIVE_METRIC,
    mode: str = "min",
) -> dict[str, Any]:
    valid_results: list[Result] = [
        result
        for result in results
        if result.error is None
        and result.metrics is not None
        and metric in result.metrics
        and isfinite(float(result.metrics[metric]))
    ]
    if not valid_results:
        failed = sum(1 for result in results if result.error is not None)
        raise RuntimeError(
            "Ray Tune completed without a valid objective result "
            f"({failed} failed trial(s)). Check trial logs and model/search-space settings."
        )
    best = results.get_best_result(
        metric=metric,
        mode=mode,
        filter_nan_and_inf=True,
    )
    best_config = best.config
    if best_config is None:
        raise RuntimeError("Ray Tune best result has no config; cannot extract best parameters.")
    return dict(best_config)


def run_optuna_study(
    *,
    space: Callable[[optuna.Trial], None],
    trainable: Callable[..., None],
    n_trials: int,
    max_t: int,
    seed: int | None,
    asha_grace_period: int,
    cpu_per_trial: float,
    max_concurrent_trials: int,
    ray_address: str | None,
    tune_storage_path: str,
    metric: str = OBJECTIVE_METRIC,
    mode: str = "min",
    time_attr: str = ORIGIN_INDEX,
    experiment_name: str | None = None,
    callbacks: list[Callback] | None = None,
    trial_state: dict[str, Any] | None = None,
    fail_fast: bool | str = False,
) -> StudyOutcome:
    """Run a Ray Tune/Optuna study for any Tune-compatible trainable.

    ``fail_fast`` is forwarded to Ray's ``FailureConfig``. Pass ``"raise"`` when
    the trainable reports recoverable failures as a finite/``inf`` objective and
    only *raises* on genuine infrastructure errors: Tune then re-raises that
    exception out of ``fit()`` instead of recording it as one errored trial among
    many (which a successful trial would otherwise mask).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if max_t < 1:
        raise ValueError("max_t must be at least 1")
    if asha_grace_period < 1:
        raise ValueError("asha_grace_period must be at least 1")
    if max_t > 1 and asha_grace_period >= max_t:
        raise ValueError("asha_grace_period must be less than max_t (number of origins)")
    if cpu_per_trial <= 0:
        raise ValueError("cpu_per_trial must be positive")
    if max_concurrent_trials < 1:
        raise ValueError("max_concurrent_trials must be at least 1")

    prepare_ray_environment()

    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    search_alg = OptunaSearch(
        space=space,
        metric=metric,
        mode=mode,
        sampler=create_tpe_sampler(seed),
    )
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        time_attr=time_attr,
        max_t=max_t,
        grace_period=asha_grace_period,
    )

    ray_runtime = acquire_ray_runtime(address=ray_address)
    try:
        import ray

        state_ref = ray.put(trial_state) if trial_state is not None else None
        trainable_fn = (
            tune.with_parameters(trainable, state_ref=state_ref)
            if state_ref is not None
            else trainable
        )
        trainable_with_resources = tune.with_resources(
            trainable_fn,
            resources={"cpu": float(cpu_per_trial)},
        )
        run_config_kwargs: dict[str, Any] = {
            "callbacks": callbacks or [],
            "storage_path": tune_storage_path,
            "failure_config": tune.FailureConfig(fail_fast=fail_fast),
        }
        if experiment_name is not None:
            run_config_kwargs["name"] = experiment_name

        with _ray_tune_env(), _restore_cwd():
            tuner = tune.Tuner(
                trainable_with_resources,
                tune_config=tune.TuneConfig(
                    search_alg=search_alg,
                    scheduler=scheduler,
                    num_samples=n_trials,
                    max_concurrent_trials=max_concurrent_trials,
                ),
                run_config=tune.RunConfig(**run_config_kwargs),
            )
            results = tuner.fit()
    finally:
        ray_runtime.release()
    return StudyOutcome(
        best_config=_best_result_config(results, metric=metric, mode=mode), results=results
    )


def _validate_origins(origins: list[pd.Timestamp], *, label: str) -> list[pd.Timestamp]:
    if not origins:
        raise ValueError(f"{label}.origins must contain at least one origin")
    return [pd.Timestamp(origin) for origin in origins]


def _validate_local_task(task: LocalTuningTask) -> list[pd.Timestamp]:
    return _validate_origins(task.origins, label="LocalTuningTask")


def _validate_global_task(task: GlobalTuningTask) -> list[pd.Timestamp]:
    if task.base_model_config.get("scope") != "global":
        raise ValueError("GlobalTuningTask.base_model_config must set scope='global'")
    return _validate_origins(task.origins, label="GlobalTuningTask")


def _resolve_candidate(value: Any) -> TuningCandidate:
    if isinstance(value, TuningCandidate):
        return value
    raise TypeError(
        f"LocalTuningTask.search_space must return a TuningCandidate; got {type(value).__name__}"
    )


class _OptunaSearchSpaceAdapter:
    """Picklable wrapper that exposes ``search_space`` to OptunaSearch.

    OptunaSearch requires the define-by-run callable to return ``None`` or a
    plain ``dict``; we discard the ``TuningCandidate`` return value while
    keeping the ``suggest_*`` calls (which are what OptunaSearch actually
    needs to record the parameter space). Defined at module scope so Ray
    Tune can pickle the searcher state.
    """

    __slots__ = ("_search_space",)

    def __init__(self, search_space: Callable[[optuna.Trial], TuningCandidate]) -> None:
        self._search_space = search_space

    def __call__(self, trial: optuna.Trial) -> None:
        _resolve_candidate(self._search_space(trial))
        return None


def _apply_conformal_overrides(
    config: SymmetricIntervalConfig, overrides: dict[str, Any]
) -> SymmetricIntervalConfig:
    if not overrides:
        return config
    return replace(config, **overrides)


def _apply_ordering_overrides(objective: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return objective
    if not is_dataclass(objective) or isinstance(objective, type):
        raise TypeError(
            "ordering_config overrides require the tuning objective to be a "
            f"dataclass instance; got {type(objective).__name__}"
        )
    return replace(objective, **overrides)


def _candidate_model_config(
    base_model_config: dict,
    candidate: TuningCandidate,
    config: StudyConfig,
) -> dict:
    return cap_threaded_config(
        {**base_model_config, **candidate.model_config, "freq": config.freq},
        config.cpu_per_trial,
    )


def _build_scoring_engine(
    config: StudyConfig,
    *,
    conformal_options: ConformalOptions | None = None,
) -> BackendEngine:
    return BackendEngine(
        execution=ExecutionOptions(
            freq=config.freq,
            backend="local",
            max_concurrency=config.max_uid_concurrency,
        ),
        conformal=conformal_options or ConformalOptions(),
    )


def _score_forecast_task(
    *,
    engine: BackendEngine,
    forecast_task: ForecastTask,
    objective: Any,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
    cpu_per_trial: float,
    report: Callable[[dict[str, float | int]], None] | None = None,
) -> float:
    total_cost = 0.0
    seen_keys: set[tuple[Any, ...]] = set()
    with threadpool_limits(limits=thread_budget(cpu_per_trial)):
        for origin_idx, result in enumerate(
            engine.iter_origins(partition_tasks([forecast_task]), actuals, origins),
            start=1,
        ):
            contribution = _objective_contribution_with(
                objective,
                result.ledger.to_df(),
                seen_keys,
            )
            if not isfinite(contribution):
                total_cost = float("inf")
                if report is not None:
                    report({OBJECTIVE_METRIC: total_cost, ORIGIN_INDEX: origin_idx})
                return total_cost
            total_cost += contribution
            if report is not None:
                report({OBJECTIVE_METRIC: total_cost, ORIGIN_INDEX: origin_idx})
    return total_cost


def _conformal_options(
    config: SymmetricIntervalConfig | None,
    candidate: TuningCandidate,
    state: dict[str, Any] | None,
) -> ConformalOptions:
    """Build conformal options for a trial from a config snapshot + resume state.

    ``config is None`` means the task carries no conformal runtime, so scoring runs
    without conformal calibration.
    """
    if config is None:
        return ConformalOptions()
    return ConformalOptions(
        runtime=SymmetricIntervalRuntime.from_state(
            _apply_conformal_overrides(config, candidate.conformal_config),
            state,
        )
    )


def _score_candidate(
    *,
    task: TuningTask,
    candidate: TuningCandidate,
    history: pd.DataFrame,
    origins: list[pd.Timestamp],
    conformal_options: ConformalOptions,
    report: Callable[[dict[str, float | int]], None] | None = None,
) -> float:
    """Build the trial's ForecastTask + objective and backtest it to an objective value.

    Shared by sequential evaluation (:func:`evaluate_candidate`) and the Ray Tune
    trainable (:func:`_make_ray_trainable`).
    """
    study_config = task.study_config
    forecast_task = ForecastTask(
        history=history,
        horizon=task.horizon,
        model_config=_candidate_model_config(task.base_model_config, candidate, study_config),
    )
    objective = _apply_ordering_overrides(task.objective, candidate.ordering_config)
    with _build_scoring_engine(study_config, conformal_options=conformal_options) as engine:
        return _score_forecast_task(
            engine=engine,
            forecast_task=forecast_task,
            objective=objective,
            actuals=task.actuals,
            origins=origins,
            cpu_per_trial=study_config.cpu_per_trial,
            report=report,
        )


def evaluate_candidate(
    task: LocalTuningTask,
    candidate: TuningCandidate,
    origins: list[pd.Timestamp],
) -> float:
    runtime_snapshot = _snapshot_conformal_runtime(task)
    conformal_options = _conformal_options(
        runtime_snapshot.config if runtime_snapshot is not None else None,
        candidate,
        runtime_snapshot.state if runtime_snapshot is not None else None,
    )
    return _score_candidate(
        task=task,
        candidate=candidate,
        history=_history_with_uid(task),
        origins=origins,
        conformal_options=conformal_options,
    )


def _optimize_task_sequential(task: LocalTuningTask, origins: list[pd.Timestamp]) -> dict[str, Any]:
    config = task.study_config
    study = optuna.create_study(direction="minimize", sampler=create_tpe_sampler(config.seed))

    def _objective(trial: optuna.Trial) -> float:
        candidate = _resolve_candidate(task.search_space(trial))
        trial.set_user_attr("resolved_config", dict(candidate.model_config))
        return evaluate_candidate(task, candidate, origins)

    study.optimize(_objective, n_trials=config.n_trials, gc_after_trial=True)
    if not study.trials or study.best_trial.value is None or not isfinite(study.best_trial.value):
        raise RuntimeError("Sequential Optuna completed without a valid objective result")
    best_config = study.best_trial.user_attrs.get("resolved_config", study.best_trial.params)
    return {**task.base_model_config, **dict(best_config)}


def _candidate_from_params(task: TuningTask, params: dict[str, Any]) -> TuningCandidate:
    """Replay ``task.search_space`` against best Optuna params to rebuild the candidate."""
    return _resolve_candidate(
        task.search_space(cast(optuna.Trial, optuna.trial.FixedTrial(dict(params))))
    )


def _merge_with_base_model(task: LocalTuningTask, candidate: TuningCandidate) -> TuningCandidate:
    return TuningCandidate(
        model_config={**task.base_model_config, **dict(candidate.model_config)},
        conformal_config=dict(candidate.conformal_config),
        ordering_config=dict(candidate.ordering_config),
    )


def _make_ray_trainable(
    *,
    task: TuningTask,
    history: pd.DataFrame,
    origins: list[pd.Timestamp],
    conformal_config: SymmetricIntervalConfig | None,
) -> Callable[..., None]:
    """Build the Ray Tune trainable for ``task``.

    The returned closure is the only Ray-facing surface: it resolves the trial's
    candidate, rehydrates conformal state from ``state_ref`` (when the task carries a
    conformal runtime), and delegates the backtest to :func:`_score_candidate`. Pass
    ``conformal_config=None`` for tasks scored without conformal calibration.
    """

    def _trainable(trial_config: dict[str, Any], *, state_ref: Any | None = None) -> None:
        from ray import tune

        candidate = _candidate_from_params(task, trial_config)
        state = _resolve_state_ref(state_ref) if conformal_config is not None else None
        _score_candidate(
            task=task,
            candidate=candidate,
            history=history,
            origins=origins,
            conformal_options=_conformal_options(conformal_config, candidate, state),
            report=tune.report,
        )

    return _trainable


def optimize_local_task_candidate(task: LocalTuningTask) -> TuningCandidate:
    """Run HPO and return the best :class:`TuningCandidate` (model + conformal + ordering)."""
    optuna_params = _run_optuna_study(task)
    return _merge_with_base_model(task, _candidate_from_params(task, optuna_params))


def optimize_local_task(task: LocalTuningTask) -> dict:
    """Run HPO and return the best model_config dict."""
    return dict(optimize_local_task_candidate(task).model_config)


def optimize_global_task_candidate(task: GlobalTuningTask) -> TuningCandidate:
    """Run global HPO and return the best :class:`TuningCandidate`."""
    origins = _validate_global_task(task)
    history = task.history.copy()
    config = task.study_config
    max_t = len(origins)
    max_concurrent_trials = _resolve_max_concurrent_trials(
        config.max_concurrent_trials,
        config.cpu_per_trial,
    )

    outcome = run_optuna_study(
        space=_OptunaSearchSpaceAdapter(task.search_space),
        trainable=_make_ray_trainable(
            task=task, history=history, origins=origins, conformal_config=None
        ),
        n_trials=config.n_trials,
        max_t=max_t,
        seed=config.seed,
        asha_grace_period=config.asha_grace_period,
        cpu_per_trial=config.cpu_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        ray_address=config.ray_address,
        tune_storage_path=_resolve_tune_storage_path_config(config),
        experiment_name=config.tune_experiment_name,
        callbacks=_build_mlflow_callbacks(config),
    )
    candidate = _resolve_candidate(
        task.search_space(cast(optuna.Trial, optuna.trial.FixedTrial(dict(outcome.best_config))))
    )
    return TuningCandidate(
        model_config={**task.base_model_config, **dict(candidate.model_config)},
        conformal_config=dict(candidate.conformal_config),
        ordering_config=dict(candidate.ordering_config),
    )


def optimize_global_task(task: GlobalTuningTask) -> dict:
    """Run global HPO and return the best model_config dict."""
    return dict(optimize_global_task_candidate(task).model_config)


def _run_optuna_study(task: LocalTuningTask) -> dict[str, Any]:
    """Run Ray Tune for ``task`` and return the best trial's Optuna params dict."""
    origins = _validate_local_task(task)
    runtime_snapshot = _snapshot_conformal_runtime(task)
    conformal_config = runtime_snapshot.config if runtime_snapshot is not None else None
    worker_task = replace(task, conformal_runtime_factory=None)
    config = worker_task.study_config

    history = _history_with_uid(worker_task)
    max_t = len(origins)
    max_concurrent_trials = _resolve_max_concurrent_trials(
        config.max_concurrent_trials,
        config.cpu_per_trial,
    )

    outcome = run_optuna_study(
        space=_OptunaSearchSpaceAdapter(worker_task.search_space),
        trainable=_make_ray_trainable(
            task=worker_task,
            history=history,
            origins=origins,
            conformal_config=conformal_config,
        ),
        n_trials=config.n_trials,
        max_t=max_t,
        seed=config.seed,
        asha_grace_period=config.asha_grace_period,
        cpu_per_trial=config.cpu_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        ray_address=config.ray_address,
        tune_storage_path=_resolve_tune_storage_path(worker_task),
        experiment_name=config.tune_experiment_name,
        callbacks=_build_mlflow_callbacks(config),
        trial_state=runtime_snapshot.state if runtime_snapshot is not None else None,
    )
    return outcome.best_config
