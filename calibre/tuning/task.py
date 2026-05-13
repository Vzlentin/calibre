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
