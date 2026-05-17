"""Tuning optimizer: runs Optuna studies for TuningTasks."""

from __future__ import annotations

import optuna
import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine
from calibre.forecasting.adapter_registry import resolve_adapter
from calibre.tuning.task import TuningTask


def create_tpe_sampler(seed: int | None) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(seed=seed)


def optimize_task(task: TuningTask) -> dict:
    """Run HPO via Optuna. Returns best model_config dict."""
    unique_id = task.unique_id
    history = task.history
    horizon = task.horizon
    base_cfg = task.base_model_config
    search_space = task.search_space
    actuals = task.actuals
    origins = task.origins
    objective = task.objective
    freq = task.freq
    conformal_runtime_factory = task.conformal_runtime_factory

    if conformal_runtime_factory is not None:

        def _objective(trial: optuna.Trial) -> float:
            candidate_config = {**base_cfg, **search_space(trial)}
            h = history.copy()
            if UNIQUE_ID not in h.columns:
                h.insert(0, UNIQUE_ID, unique_id)
            forecast_task = ForecastTask(
                history=h,
                horizon=horizon,
                model_config=candidate_config,
            )
            result = BackendEngine(
                freq=freq,
                conformal_runtime=conformal_runtime_factory(),
            ).execute([forecast_task], actuals, origins)
            resolved = result.ledger.to_df().dropna(subset=[Y, Y_HAT])
            if resolved.empty:
                return float("inf")
            return float(objective.evaluate(resolved, resolved[Y]))

    else:
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
    study = optuna.create_study(direction="minimize", sampler=create_tpe_sampler(task.seed))
    study.optimize(_objective, n_trials=task.n_trials)
    return {**task.base_model_config, **study.best_params}
