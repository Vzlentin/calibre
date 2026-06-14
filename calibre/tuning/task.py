"""Tuning candidate and the per-trial task it configures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import optuna
import pandas as pd

from calibre.conformal import ConformalRuntime
from calibre.tuning.objectives import TuningObjective


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
    """Shared study/execution settings for local (per-series) and global tuning."""

    n_trials: int = 50
    freq: str = "W"
    seed: int | None = None
    asha_grace_period: int = 1
    cpu_per_trial: float = 1.0
    max_concurrent_trials: int | None = None
    max_uid_concurrency: int | None = None
    ray_address: str | None = None
    tune_storage_path: str | None = None
    results_dir: str | None = "results"
    tune_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_parent_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class LocalTuningTask:
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


@dataclass(frozen=True, slots=True)
class GlobalTuningTask:
    """Global hyperparameter optimization task."""

    history: pd.DataFrame
    horizon: int
    base_model_config: dict
    search_space: Callable[[optuna.Trial], TuningCandidate]
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    objective: TuningObjective
    study_config: StudyConfig = field(default_factory=StudyConfig)
