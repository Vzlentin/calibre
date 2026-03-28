from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import optuna
import pandas as pd

from calibre.conformal import ConformalPolicyConfig
from calibre.contracts.forecast_frame import Y_HAT, Y
from calibre.tasks.forecast_task import ForecastTask


@dataclass(frozen=True)
class TuningTask:
    """Per-series hyperparameter optimization task.

    Runs Optuna HPO by evaluating candidate configs via walk-forward
    backtesting using the BackendEngine. The search_space callable
    receives an optuna.Trial and returns a dict of candidate params
    to overlay on base_model_config.

    Example::

        def space(trial: optuna.Trial) -> dict:
            return {"season_length": trial.suggest_categorical("season_length", [4, 12, 52])}

        task = TuningTask(
            unique_id="store_A",
            history=df,
            horizon=4,
            base_model_config={"backend": "statsforecast", "model": "SeasonalNaive", "name": "sn"},
            search_space=space,
            actuals=df,
            origins=[cutoff_date],
            metric=smape,
            n_trials=30,
        )
        best_config = task.optimize()
    """

    unique_id: str
    history: pd.DataFrame
    horizon: int
    base_model_config: dict
    search_space: Callable[[optuna.Trial], dict]
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    metric: Callable  # (np.ndarray, np.ndarray) -> float
    n_trials: int = 50
    freq: str = "W"
    conformal_config: ConformalPolicyConfig | None = None

    def optimize(self) -> dict:
        """Run HPO via Optuna. Returns best model_config dict."""
        unique_id = self.unique_id
        history = self.history
        horizon = self.horizon
        base_cfg = self.base_model_config
        search_space = self.search_space
        actuals = self.actuals
        origins = self.origins
        metric = self.metric
        freq = self.freq
        conformal_config = self.conformal_config

        def _objective(trial: optuna.Trial) -> float:
            candidate_config = {**base_cfg, **search_space(trial)}
            task = ForecastTask(
                unique_id=unique_id,
                history=history,
                horizon=horizon,
                model_config=candidate_config,
            )
            result = BackendEngine(
                freq=freq,
                conformal_config=conformal_config,
            ).execute([task], actuals, origins)
            ledger = result.ledger
            resolved = ledger.to_df().dropna(subset=[Y, Y_HAT])
            if resolved.empty:
                return float("inf")
            return float(metric(resolved[Y].to_numpy(), resolved[Y_HAT].to_numpy()))

        from calibre.engine.backend import BackendEngine  # deferred to avoid circular import

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=self.n_trials)
        return {**self.base_model_config, **study.best_params}
