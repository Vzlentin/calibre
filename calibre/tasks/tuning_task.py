from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import optuna
import pandas as pd

from calibre.conformal import ConformalPolicyConfig
from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y_HAT, Y
from calibre.tasks.forecast_task import ForecastTask
from calibre.tuning.objectives import TuningObjective


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
            objective=Accuracy(metric=smape),
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
    objective: TuningObjective
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
        objective = self.objective
        freq = self.freq
        conformal_config = self.conformal_config

        if conformal_config is not None:
            from calibre.engine.backend import BackendEngine  # deferred to avoid circular import

            def _objective(trial: optuna.Trial) -> float:
                candidate_config = {**base_cfg, **search_space(trial)}
                h = history.copy()
                if UNIQUE_ID not in h.columns:
                    h.insert(0, UNIQUE_ID, unique_id)
                task = ForecastTask(
                    history=h,
                    horizon=horizon,
                    model_config=candidate_config,
                )
                result = BackendEngine(
                    freq=freq,
                    conformal_config=conformal_config,
                ).execute([task], actuals, origins)
                resolved = result.ledger.to_df().dropna(subset=[Y, Y_HAT])
                if resolved.empty:
                    return float("inf")
                return float(objective.evaluate(resolved, resolved[Y]))
        else:
            # Fast path: call adapters directly, skipping Fugue partitioning overhead.
            # Equivalent to BackendEngine._run_parallel for a single series but without
            # the fa.transform wrapper — each trial fires n_origins adapter fit/predict
            # calls instead of n_origins fa.transform round-trips.
            from calibre.models.registry import (
                resolve_adapter,  # deferred: avoids circular import via calibre.tasks.__init__
            )

            max_resolved_ds = max(origins)

            def _objective(trial: optuna.Trial) -> float:
                candidate_config = {**base_cfg, **search_space(trial), "freq": freq}
                h = history.copy()
                if UNIQUE_ID not in h.columns:
                    h.insert(0, UNIQUE_ID, unique_id)

                resolved_frames: list[pd.DataFrame] = []

                for origin in origins:
                    hist_slice = h[h[DS] < origin]
                    if hist_slice.empty:
                        continue
                    origin_task = ForecastTask(
                        history=hist_slice,
                        horizon=horizon,
                        model_config=candidate_config,
                        forecast_origin=origin,
                    )
                    adapter = resolve_adapter(candidate_config)
                    adapter.fit(origin_task)
                    preds = adapter.predict(origin_task)
                    # Mirror BackendEngine's resolve_actuals gate: only score
                    # rows with ds ≤ the last origin to match current semantics.
                    preds = preds[preds[DS] <= max_resolved_ds]
                    if preds.empty:
                        continue
                    preds_clean = preds.drop(columns=[Y], errors="ignore")
                    merged = preds_clean.merge(actuals[[DS, Y]], on=DS, how="left").dropna(
                        subset=[Y, Y_HAT]
                    )
                    if merged.empty:
                        continue
                    resolved_frames.append(merged)

                if not resolved_frames:
                    return float("inf")
                resolved = pd.concat(resolved_frames, ignore_index=True)
                return float(objective.evaluate(resolved, resolved[Y]))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=self.n_trials)
        return {**self.base_model_config, **study.best_params}
