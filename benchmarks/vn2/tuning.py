"""Parallel per-series hyperparameter tuning for the VN2 benchmark."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable

import optuna
import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y
from calibre.metrics import smape
from calibre.tasks.tuning_task import TuningTask


def seasonal_naive_search_space(trial: optuna.Trial) -> dict:
    """Search space for SeasonalNaive: tune season_length."""
    return {
        "season_length": trial.suggest_categorical("season_length", [4, 13, 26, 52]),
    }


def tune_one_series(
    unique_id: str,
    sales: pd.DataFrame,
    horizon: int,
    base_config: dict,
    search_space: Callable[[optuna.Trial], dict] = seasonal_naive_search_space,
    n_trials: int = 20,
    n_origins: int = 5,
    freq: str = "W",
) -> dict:
    """Tune a single series. Returns the best model config dict."""
    series_data = sales[sales[UNIQUE_ID] == unique_id]
    all_dates = sorted(series_data[DS].unique())

    if len(all_dates) < n_origins + horizon:
        n_origins = max(1, len(all_dates) - horizon)

    origins = [pd.Timestamp(d) for d in all_dates[-(n_origins + horizon) : -horizon]]
    if not origins:
        return base_config

    history = series_data[[DS, Y]].sort_values(DS).reset_index(drop=True)
    actuals = series_data[[UNIQUE_ID, DS, Y]].copy()

    task = TuningTask(
        unique_id=unique_id,
        history=history,
        horizon=horizon,
        base_model_config=base_config,
        search_space=search_space,
        actuals=actuals,
        origins=origins,
        metric=smape,
        n_trials=n_trials,
        freq=freq,
    )
    return task.optimize()


def tune_all_series(
    sales: pd.DataFrame,
    horizon: int,
    base_config: dict,
    search_space: Callable[[optuna.Trial], dict] = seasonal_naive_search_space,
    n_trials: int = 20,
    n_origins: int = 5,
    freq: str = "W",
    max_workers: int = 4,
) -> dict[str, dict]:
    """Tune base_config per series in parallel using threads.

    Args:
        sales: Long-format sales DataFrame with unique_id, ds, y columns.
        horizon: Forecast horizon (number of periods ahead).
        base_config: Base model config dict to tune (e.g. SeasonalNaive).
        search_space: Optuna search space callable. Must return a dict of params.
        n_trials: Number of Optuna trials per series.
        n_origins: Number of walk-forward origins to evaluate each trial on.
        freq: Frequency string passed to BackendEngine.
        max_workers: Maximum number of parallel threads.

    Returns:
        Dict mapping unique_id → best model config dict.
    """
    unique_ids = sorted(sales[UNIQUE_ID].unique())

    def _tune(uid: str) -> tuple[str, dict]:
        best = tune_one_series(
            uid, sales, horizon, base_config, search_space, n_trials, n_origins, freq
        )
        return uid, best

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_tune, uid): uid for uid in unique_ids}
        for future in concurrent.futures.as_completed(futures):
            uid, best_config = future.result()
            results[uid] = best_config

    return results
