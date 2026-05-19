from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import optuna
import pandas as pd

from calibre.conformal import ConformalRuntime
from calibre.tuning.objectives import TuningObjective


@dataclass(frozen=True)
class TuningTask:
    """Per-series hyperparameter optimization task."""

    unique_id: str
    history: pd.DataFrame
    horizon: int
    base_model_config: dict
    search_space: Callable[[optuna.Trial], dict]
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    objective: TuningObjective
    n_trials: int = 50
    freq: str = "W"
    conformal_runtime_factory: Callable[[], ConformalRuntime] | None = None
    seed: int | None = None
    asha_grace_period: int = 8
    cpu_per_trial: float = 1.0
    max_concurrent_trials: int | None = None
    max_uid_concurrency: int | None = None
    ray_address: str | None = None
    ray_local_mode: bool = False
    tune_storage_path: str | None = None
    tune_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_parent_run_id: str | None = None
