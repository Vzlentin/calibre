from __future__ import annotations

import optuna
import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y, quantile_column
from calibre.tuning.objectives import CumulativePinball
from calibre.tuning.optimizer import optimize_global_task
from calibre.tuning.task import GlobalTuningTask, StudyConfig, TuningCandidate


def _constant_n_estimators_space(trial: optuna.Trial) -> TuningCandidate:
    return TuningCandidate(
        model_config={"n_estimators": trial.suggest_categorical("n_estimators", [5])}
    )


def test_cumulative_pinball_averages_cumulative_window_pinball() -> None:
    qcol = quantile_column(0.5)
    origin_a = pd.Timestamp("2024-01-07")
    origin_b = pd.Timestamp("2024-01-14")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "B", "B", "A", "A", "B", "B"],
            FORECAST_ORIGIN: [
                origin_a,
                origin_a,
                origin_a,
                origin_a,
                origin_b,
                origin_b,
                origin_b,
                origin_b,
            ],
            H: [1, 2, 1, 2, 1, 2, 1, 2],
            qcol: [12.0, 13.0, 7.0, 7.0, 3.0, 5.0, 1.0, 2.0],
        }
    )
    actuals = pd.Series([10.0, 20.0, 4.0, 6.0, 4.0, 4.0, 2.0, 3.0])

    value = CumulativePinball(quantile=0.5, tau=0.8).evaluate(frame, actuals)

    assert value == pytest.approx(1.6)


def test_optimize_global_task_returns_complete_global_model_config(tmp_path) -> None:
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    history = pd.concat(
        [
            pd.DataFrame(
                {
                    UNIQUE_ID: "A",
                    DS: dates,
                    Y: [10.0, 20.0, 30.0, 40.0] * 5,
                }
            ),
            pd.DataFrame(
                {
                    UNIQUE_ID: "B",
                    DS: dates,
                    Y: [5.0, 15.0, 25.0, 35.0] * 5,
                }
            ),
        ],
        ignore_index=True,
    )

    result = optimize_global_task(
        GlobalTuningTask(
            history=history,
            horizon=2,
            base_model_config={
                "backend": "mlforecast",
                "scope": "global",
                "model": "lightgbm.LGBMRegressor",
                "objective": "quantile",
                "quantiles": [0.5],
                "strategy": "direct",
                "lags": [1, 2, 3, 4],
                "verbosity": -1,
            },
            search_space=_constant_n_estimators_space,
            actuals=history,
            origins=[dates[15]],
            objective=CumulativePinball(quantile=0.5, tau=0.8),
            study_config=StudyConfig(
                n_trials=1,
                freq="W",
                seed=7,
                ray_local_mode=True,
                tune_storage_path=str(tmp_path / "ray-tune"),
            ),
        )
    )

    assert result["backend"] == "mlforecast"
    assert result["scope"] == "global"
    assert result["model"] == "lightgbm.LGBMRegressor"
    assert result["n_estimators"] == 5
