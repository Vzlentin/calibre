from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.vn2 import tuning
from benchmarks.vn2.replay import PolicyApplicationError

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"


def test_infra_exception_not_swallowed_as_inf(monkeypatch) -> None:
    reports: list[dict] = []

    def _raise_infra_failure(*args, **kwargs):
        raise RuntimeError("worker import failed")

    def _fake_report(metrics: dict) -> None:
        reports.append(metrics)

    params = {
        "lag_set_idx": 0,
        "target_mode": "per_horizon",
        "quantile_alpha": 0.5,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "crc_enabled": False,
    }

    def _fake_run_optuna_tune(trainable, search_space, **kwargs):
        del search_space, kwargs
        trainable(params)
        raise AssertionError("trainable should have re-raised the infrastructure failure")

    monkeypatch.setattr(tuning, "build_replay_cache", _raise_infra_failure)
    monkeypatch.setattr("ray.tune.report", _fake_report)

    with pytest.raises(RuntimeError, match="worker import failed"):
        tuning.run_cost_search(
            data_dir=DATA_DIR,
            model_config={"quantiles": [0.5]},
            decision_rounds=1,
            delivery_weeks=0,
            search_forecast=True,
            n_trials=1,
            tune_runner=_fake_run_optuna_tune,
        )

    assert reports[-1]["infra_failure"] == 1


def test_policy_failure_not_classified_as_bad_trial(monkeypatch) -> None:
    reports: list[dict] = []

    def _raise_policy_failure(*args, **kwargs):
        raise PolicyApplicationError(1, ValueError("broken policy"))

    def _fake_report(metrics: dict) -> None:
        reports.append(metrics)

    params = {
        "crc_enabled": False,
    }

    def _fake_run_optuna_tune(trainable, search_space, **kwargs):
        del search_space, kwargs
        trainable(params)
        raise AssertionError("trainable should have re-raised the policy failure")

    monkeypatch.setattr(tuning, "build_replay_cache", lambda *args, **kwargs: object())
    monkeypatch.setattr(tuning, "replay_cached_cost", _raise_policy_failure)
    monkeypatch.setattr("ray.tune.report", _fake_report)

    with pytest.raises(PolicyApplicationError, match="broken policy"):
        tuning.run_cost_search(
            data_dir=DATA_DIR,
            model_config={"quantiles": [0.5]},
            decision_rounds=1,
            delivery_weeks=0,
            search_forecast=False,
            n_trials=1,
            tune_runner=_fake_run_optuna_tune,
        )

    assert reports[-1]["policy_failure"] == 1
    assert "bad_trial" not in reports[-1]
