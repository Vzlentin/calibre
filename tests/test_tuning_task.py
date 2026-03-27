from __future__ import annotations

import pandas as pd
import pytest

import optuna

from calibre.metrics import mae, smape
from calibre.tasks.tuning_task import TuningTask


def _space_season_length(trial: optuna.Trial) -> dict:
    return {"season_length": trial.suggest_categorical("season_length", [2, 4])}


@pytest.fixture
def series_df(dates, repeating_pattern):
    return pd.DataFrame({
        "unique_id": "test_series",
        "ds": dates,
        "y": repeating_pattern,
    })


@pytest.fixture
def tuning_task(series_df, dates):
    return TuningTask(
        unique_id="test_series",
        history=series_df,
        horizon=4,
        base_model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "name": "sn_tuning",
        },
        search_space=_space_season_length,
        actuals=series_df,
        origins=[dates[15]],
        metric=smape,
        n_trials=10,
        freq="W",
    )


def test_optimize_finds_correct_season_length(tuning_task):
    """SeasonalNaive HPO should pick season_length=4 for a period-4 pattern."""
    best_config = tuning_task.optimize()
    assert best_config["season_length"] == 4


def test_optimize_returns_complete_config(tuning_task):
    """Result must contain all base_model_config keys plus tuned params."""
    best_config = tuning_task.optimize()
    assert best_config["backend"] == "statsforecast"
    assert best_config["model"] == "SeasonalNaive"
    assert best_config["name"] == "sn_tuning"
    assert "season_length" in best_config


def test_optimize_single_trial(series_df, dates):
    """n_trials=1 should run without error."""
    task = TuningTask(
        unique_id="test_series",
        history=series_df,
        horizon=4,
        base_model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "name": "sn_tuning",
        },
        search_space=_space_season_length,
        actuals=series_df,
        origins=[dates[15]],
        metric=smape,
        n_trials=1,
        freq="W",
    )
    result = task.optimize()
    assert isinstance(result, dict)
    assert "season_length" in result


def test_optimize_with_mae_metric(series_df, dates):
    """Custom metric (mae) should also converge to season_length=4."""
    task = TuningTask(
        unique_id="test_series",
        history=series_df,
        horizon=4,
        base_model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "name": "sn_tuning",
        },
        search_space=_space_season_length,
        actuals=series_df,
        origins=[dates[15]],
        metric=mae,
        n_trials=10,
        freq="W",
    )
    best_config = task.optimize()
    assert best_config["season_length"] == 4
