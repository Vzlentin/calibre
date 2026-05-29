from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import optuna
import pandas as pd

from calibre.conformal import ConformalRuntime
from calibre.tuning.objectives import TuningObjective

_UNSET = object()


@dataclass(frozen=True, slots=True)
class TuningCandidate:
    """Per-trial configuration produced by a search space.

    Splits the flat config dict that older search spaces returned into the
    three routing channels the optimizer drives separately:
    ``model_config`` is merged into the ForecastTask, ``conformal_config``
    overrides fields on the conformal runtime config snapshot, and
    ``ordering_config`` overrides fields on the tuning objective
    (``Cost`` / ``Pareto``).
    """

    model_config: dict
    conformal_config: dict = field(default_factory=dict)
    ordering_config: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyConfig:
    n_trials: int = 50
    freq: str = "W"
    seed: int | None = None
    asha_grace_period: int = 1
    cpu_per_trial: float = 1.0
    max_concurrent_trials: int | None = None
    max_uid_concurrency: int | None = None
    ray_address: str | None = None
    ray_local_mode: bool = False
    tune_storage_path: str | None = None
    results_dir: str | None = "results"
    tune_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_parent_run_id: str | None = None


def _coerce_study_config(
    study_config: StudyConfig | None,
    *,
    n_trials: object = _UNSET,
    freq: object = _UNSET,
    seed: object = _UNSET,
    asha_grace_period: object = _UNSET,
    cpu_per_trial: object = _UNSET,
    max_concurrent_trials: object = _UNSET,
    max_uid_concurrency: object = _UNSET,
    ray_address: object = _UNSET,
    ray_local_mode: object = _UNSET,
    tune_storage_path: object = _UNSET,
    results_dir: object = _UNSET,
    tune_experiment_name: object = _UNSET,
    mlflow_tracking_uri: object = _UNSET,
    mlflow_experiment_name: object = _UNSET,
    mlflow_parent_run_id: object = _UNSET,
) -> StudyConfig:
    config = study_config or StudyConfig()
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "n_trials": n_trials,
            "freq": freq,
            "seed": seed,
            "asha_grace_period": asha_grace_period,
            "cpu_per_trial": cpu_per_trial,
            "max_concurrent_trials": max_concurrent_trials,
            "max_uid_concurrency": max_uid_concurrency,
            "ray_address": ray_address,
            "ray_local_mode": ray_local_mode,
            "tune_storage_path": tune_storage_path,
            "results_dir": results_dir,
            "tune_experiment_name": tune_experiment_name,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_parent_run_id": mlflow_parent_run_id,
        }.items()
        if value is not _UNSET
    }
    return replace(config, **overrides) if overrides else config


@dataclass(frozen=True, init=False)
class TuningTask:
    """Per-series hyperparameter optimization task."""

    unique_id: str
    history: pd.DataFrame
    horizon: int
    base_model_config: dict
    search_space: Callable[[optuna.Trial], TuningCandidate]
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    objective: TuningObjective
    conformal_runtime_factory: Callable[[], ConformalRuntime] | None = None
    study_config: StudyConfig = field(default_factory=StudyConfig)

    def __init__(
        self,
        *,
        unique_id: str,
        history: pd.DataFrame,
        horizon: int,
        base_model_config: dict,
        search_space: Callable[[optuna.Trial], TuningCandidate],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
        objective: TuningObjective,
        conformal_runtime_factory: Callable[[], ConformalRuntime] | None = None,
        study_config: StudyConfig | None = None,
        n_trials: object = _UNSET,
        freq: object = _UNSET,
        seed: object = _UNSET,
        asha_grace_period: object = _UNSET,
        cpu_per_trial: object = _UNSET,
        max_concurrent_trials: object = _UNSET,
        max_uid_concurrency: object = _UNSET,
        ray_address: object = _UNSET,
        ray_local_mode: object = _UNSET,
        tune_storage_path: object = _UNSET,
        results_dir: object = _UNSET,
        tune_experiment_name: object = _UNSET,
        mlflow_tracking_uri: object = _UNSET,
        mlflow_experiment_name: object = _UNSET,
        mlflow_parent_run_id: object = _UNSET,
    ) -> None:
        object.__setattr__(self, "unique_id", unique_id)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "base_model_config", base_model_config)
        object.__setattr__(self, "search_space", search_space)
        object.__setattr__(self, "actuals", actuals)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "conformal_runtime_factory", conformal_runtime_factory)
        object.__setattr__(
            self,
            "study_config",
            _coerce_study_config(
                study_config,
                n_trials=n_trials,
                freq=freq,
                seed=seed,
                asha_grace_period=asha_grace_period,
                cpu_per_trial=cpu_per_trial,
                max_concurrent_trials=max_concurrent_trials,
                max_uid_concurrency=max_uid_concurrency,
                ray_address=ray_address,
                ray_local_mode=ray_local_mode,
                tune_storage_path=tune_storage_path,
                results_dir=results_dir,
                tune_experiment_name=tune_experiment_name,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                mlflow_parent_run_id=mlflow_parent_run_id,
            ),
        )

    @property
    def n_trials(self) -> int:
        return self.study_config.n_trials

    @property
    def freq(self) -> str:
        return self.study_config.freq

    @property
    def seed(self) -> int | None:
        return self.study_config.seed

    @property
    def asha_grace_period(self) -> int:
        return self.study_config.asha_grace_period

    @property
    def cpu_per_trial(self) -> float:
        return self.study_config.cpu_per_trial

    @property
    def max_concurrent_trials(self) -> int | None:
        return self.study_config.max_concurrent_trials

    @property
    def max_uid_concurrency(self) -> int | None:
        return self.study_config.max_uid_concurrency

    @property
    def ray_address(self) -> str | None:
        return self.study_config.ray_address

    @property
    def ray_local_mode(self) -> bool:
        return self.study_config.ray_local_mode

    @property
    def tune_storage_path(self) -> str | None:
        return self.study_config.tune_storage_path

    @property
    def results_dir(self) -> str | None:
        return self.study_config.results_dir

    @property
    def tune_experiment_name(self) -> str | None:
        return self.study_config.tune_experiment_name

    @property
    def mlflow_tracking_uri(self) -> str | None:
        return self.study_config.mlflow_tracking_uri

    @property
    def mlflow_experiment_name(self) -> str | None:
        return self.study_config.mlflow_experiment_name

    @property
    def mlflow_parent_run_id(self) -> str | None:
        return self.study_config.mlflow_parent_run_id


@dataclass(frozen=True, init=False)
class PanelTuningTask:
    """Panel/global hyperparameter optimization task."""

    history: pd.DataFrame
    horizon: int
    base_model_config: dict
    search_space: Callable[[optuna.Trial], TuningCandidate]
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    objective: TuningObjective
    study_config: StudyConfig = field(default_factory=StudyConfig)

    def __init__(
        self,
        *,
        history: pd.DataFrame,
        horizon: int,
        base_model_config: dict,
        search_space: Callable[[optuna.Trial], TuningCandidate],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
        objective: TuningObjective,
        study_config: StudyConfig | None = None,
        n_trials: object = _UNSET,
        freq: object = _UNSET,
        seed: object = _UNSET,
        asha_grace_period: object = _UNSET,
        cpu_per_trial: object = _UNSET,
        max_concurrent_trials: object = _UNSET,
        max_uid_concurrency: object = _UNSET,
        ray_address: object = _UNSET,
        ray_local_mode: object = _UNSET,
        tune_storage_path: object = _UNSET,
        results_dir: object = _UNSET,
        tune_experiment_name: object = _UNSET,
        mlflow_tracking_uri: object = _UNSET,
        mlflow_experiment_name: object = _UNSET,
        mlflow_parent_run_id: object = _UNSET,
    ) -> None:
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "base_model_config", base_model_config)
        object.__setattr__(self, "search_space", search_space)
        object.__setattr__(self, "actuals", actuals)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "study_config",
            _coerce_study_config(
                study_config,
                n_trials=n_trials,
                freq=freq,
                seed=seed,
                asha_grace_period=asha_grace_period,
                cpu_per_trial=cpu_per_trial,
                max_concurrent_trials=max_concurrent_trials,
                max_uid_concurrency=max_uid_concurrency,
                ray_address=ray_address,
                ray_local_mode=ray_local_mode,
                tune_storage_path=tune_storage_path,
                results_dir=results_dir,
                tune_experiment_name=tune_experiment_name,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                mlflow_parent_run_id=mlflow_parent_run_id,
            ),
        )

    @property
    def n_trials(self) -> int:
        return self.study_config.n_trials

    @property
    def freq(self) -> str:
        return self.study_config.freq

    @property
    def seed(self) -> int | None:
        return self.study_config.seed

    @property
    def asha_grace_period(self) -> int:
        return self.study_config.asha_grace_period

    @property
    def cpu_per_trial(self) -> float:
        return self.study_config.cpu_per_trial

    @property
    def max_concurrent_trials(self) -> int | None:
        return self.study_config.max_concurrent_trials

    @property
    def max_uid_concurrency(self) -> int | None:
        return self.study_config.max_uid_concurrency

    @property
    def ray_address(self) -> str | None:
        return self.study_config.ray_address

    @property
    def ray_local_mode(self) -> bool:
        return self.study_config.ray_local_mode

    @property
    def tune_storage_path(self) -> str | None:
        return self.study_config.tune_storage_path

    @property
    def results_dir(self) -> str | None:
        return self.study_config.results_dir

    @property
    def tune_experiment_name(self) -> str | None:
        return self.study_config.tune_experiment_name

    @property
    def mlflow_tracking_uri(self) -> str | None:
        return self.study_config.mlflow_tracking_uri

    @property
    def mlflow_experiment_name(self) -> str | None:
        return self.study_config.mlflow_experiment_name

    @property
    def mlflow_parent_run_id(self) -> str | None:
        return self.study_config.mlflow_parent_run_id
