from __future__ import annotations

import warnings

import optuna
import pandas as pd

import calibre.tuning.optimizer as optimizer
from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.evaluation.point_metrics import smape
from calibre.tuning.objectives import Accuracy
from calibre.tuning.task import LocalTuningTask, StudyConfig, TuningCandidate


def _space_season_length(trial: optuna.Trial) -> TuningCandidate:
    return TuningCandidate(
        model_config={"season_length": trial.suggest_categorical("season_length", [2, 4])}
    )


def test_no_sequential_fallback_when_conformal_in_loop(monkeypatch, tmp_path) -> None:
    # Patch the private fallback directly: whether it ran is not observable
    # through any public surface, so spying on it is the only way to assert the
    # Ray-Tune path handled conformal-in-loop without falling back.
    def _fail_sequential(*args, **kwargs):
        raise AssertionError("conformal tuning used the sequential fallback")

    monkeypatch.setattr(optimizer, "_optimize_task_sequential", _fail_sequential)

    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    series = pd.DataFrame(
        {
            "unique_id": "test_series",
            "ds": dates,
            "y": [10.0, 20.0, 30.0, 40.0] * 5,
        }
    )

    def _runtime_factory() -> SymmetricIntervalRuntime:
        return SymmetricIntervalRuntime(
            SymmetricIntervalConfig(
                method="aci",
                coverage=0.9,
                calibration_window=4,
                gamma=0.05,
            )
        )

    task = LocalTuningTask(
        unique_id="test_series",
        history=series,
        horizon=4,
        base_model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "name": "sn_tuning",
        },
        search_space=_space_season_length,
        actuals=series,
        origins=[dates[15]],
        objective=Accuracy(metric=smape),
        conformal_runtime_factory=_runtime_factory,
        study_config=StudyConfig(
            n_trials=1,
            freq="W",
            seed=3,
            tune_storage_path=str(tmp_path / "ray-tune"),
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimizer.optimize_local_task(task)

    assert result["season_length"] in {2, 4}
    assert not any(
        "falling back to sequential Optuna" in str(warning.message) for warning in caught
    )
