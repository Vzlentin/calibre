"""Smoke test for the tuned VN2 benchmark (global LGBM + HPO + R,S).

Bypasses the live HPO by passing a pre-built ``best_config``, then runs a
single decision round + delivery week on a 3-series subset of the real
VN2 dataset. Verifies the public surface of ``run_benchmark`` and the
per-product cost dataframe shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest
from mlforecast.lag_transforms import RollingMean

import benchmarks.vn2.replay as replay_module
import benchmarks.vn2.run_benchmark as run_benchmark_module
import benchmarks.vn2.search as search_module
from benchmarks.vn2.config import BEST_CONFIG, CUMULATIVE_BEST_CONFIG, TOP1_CRC_CONFIG
from benchmarks.vn2.data import as_cumulative_decision_frame, prepare_cumulative_target_history
from benchmarks.vn2.diagnostics import optimal_order_path_for_sku
from benchmarks.vn2.replay import (
    build_replay_cache,
    replay_cached_cost,
    round_actuals,
    run_order_conformal_warmup,
)
from benchmarks.vn2.run_benchmark import run_benchmark, run_from_config
from benchmarks.vn2.search import run_cost_search, run_hpo
from benchmarks.vn2.simulator import ProductState, extract_new_actuals, load_initial_states
from calibre.cli.config import load_config_from_mapping
from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    quantile_column,
)
from calibre.execution.data_loading import load_period
from calibre.tuning import CumulativePinball, StudyOutcome

DATA_DIR = Path(__file__).parents[2] / "data" / "vn2"

# Cheap, deterministic global LGBM config with the marker the loop reads.
_FAST_BEST_CONFIG = {
    "backend": "mlforecast",
    "scope": "global",
    "name": "test_global_lgbm",
    "model": "lightgbm.LGBMRegressor",
    "objective": "quantile",
    "quantiles": [0.52],
    "strategy": "direct",
    "lags": [1, 2, 3, 4, 13, 26, 52],
    "lag_transforms": {1: [RollingMean(window_size=4)]},
    "n_estimators": 50,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
    "_quantile_alpha": 0.52,
}


def _get_first_n_series(n: int) -> list[str]:
    states = load_initial_states(DATA_DIR / "week_0_initial_state.csv")
    return sorted(states.keys())[:n]


def _run(series: list[str]):
    return run_benchmark(
        data_dir=DATA_DIR,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=series,
        results_dir=None,
        verbose=False,
        best_config=_FAST_BEST_CONFIG,
    )


@pytest.fixture(scope="module")
def benchmark_costs() -> pd.DataFrame:
    """Run the tuned benchmark once on a 3-series subset; shared across assertions."""
    return _run(_get_first_n_series(3))


class TestVN2BenchmarkIntegration:
    def test_one_row_per_product_with_cost_columns(self, benchmark_costs: pd.DataFrame) -> None:
        series = _get_first_n_series(3)
        assert set(benchmark_costs["unique_id"]) == set(series)
        assert len(benchmark_costs) == len(series)
        assert set(benchmark_costs.columns) == {
            "unique_id",
            "holding_cost",
            "shortage_cost",
            "total_cost",
        }

    def test_costs_are_finite_non_negative_and_additive(
        self, benchmark_costs: pd.DataFrame
    ) -> None:
        costs = benchmark_costs[["holding_cost", "shortage_cost", "total_cost"]].to_numpy()
        assert np.isfinite(costs).all()
        assert costs.min() >= 0.0
        np.testing.assert_allclose(
            benchmark_costs["total_cost"].to_numpy(),
            (benchmark_costs["holding_cost"] + benchmark_costs["shortage_cost"]).to_numpy(),
        )


def test_run_hpo_returns_valid_config(monkeypatch) -> None:
    """run_hpo builds a GlobalTuningTask and returns an engine-ready model config."""
    series = _get_first_n_series(2)

    captured: dict[str, Any] = {}

    def _fake_optimize_panel_task(task):
        captured["task"] = task
        return {k: v for k, v in _FAST_BEST_CONFIG.items() if k != "_quantile_alpha"}

    monkeypatch.setattr(search_module, "optimize_global_task", _fake_optimize_panel_task)

    config = run_hpo(
        data_dir=DATA_DIR,
        horizon=3,
        n_trials=1,
        n_origins=1,
        timeout_sec=120,
        series_filter=series,
        seed=0,
        verbose=False,
    )
    assert "_quantile_alpha" in config
    assert "quantiles" in config and len(config["quantiles"]) == 1
    assert config["backend"] == "mlforecast"
    assert config["scope"] == "global"
    assert 0.0 < config["_quantile_alpha"] < 1.0

    task = captured["task"]
    assert task.base_model_config["scope"] == "global"
    assert task.study_config.n_trials == 1
    assert len(task.origins) == 1
    assert isinstance(task.objective, CumulativePinball)


def test_committed_best_config_is_structurally_complete() -> None:
    required_keys = {
        "backend",
        "scope",
        "model",
        "objective",
        "quantiles",
        "lags",
        "_quantile_alpha",
    }
    assert required_keys.issubset(BEST_CONFIG)
    assert BEST_CONFIG["objective"] == "quantile"
    assert isinstance(BEST_CONFIG["quantiles"], list) and BEST_CONFIG["quantiles"]
    assert 0.0 < float(BEST_CONFIG["_quantile_alpha"]) < 1.0
    assert float(BEST_CONFIG["_quantile_alpha"]) == BEST_CONFIG["quantiles"][0]


def test_top1_and_cumulative_candidate_configs_are_explicit() -> None:
    assert TOP1_CRC_CONFIG.method_name == "capped_crc"
    assert CUMULATIVE_BEST_CONFIG["_target_mode"] == "cumulative"
    assert CUMULATIVE_BEST_CONFIG["_quantile_alpha"] == pytest.approx(1.0 / 1.2)


def test_cumulative_target_history_uses_trailing_protection_period_sum() -> None:
    sales = pd.DataFrame(
        {
            UNIQUE_ID: ["A"] * 4,
            DS: pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            Y: [1.0, 2.0, 3.0, 4.0],
        }
    )

    history = prepare_cumulative_target_history(sales, instock=None, protection_period=3)

    assert history[DS].tolist() == sales[DS].iloc[2:].tolist()
    assert history[Y].tolist() == [6.0, 9.0]


def test_cumulative_decision_frame_keeps_only_terminal_base_forecast() -> None:
    qcol = quantile_column(0.833)
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"] * 3,
            DS: pd.date_range("2024-01-08", periods=3, freq="W-MON"),
            Y: [float("nan")] * 3,
            Y_HAT: [10.0, 20.0, 30.0],
            H: [1, 2, 3],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")] * 3,
            MODEL_NAME: ["M"] * 3,
            qcol: [100.0, 200.0, 300.0],
        }
    )

    cumulative = as_cumulative_decision_frame(frame, protection_period=3)

    assert cumulative[Y_HAT].tolist() == [0.0, 0.0, 30.0]
    assert cumulative[qcol].tolist() == [0.0, 0.0, 300.0]


def test_cumulative_hpo_objective_scores_terminal_horizon_only() -> None:
    """Cumulative-target HPO must collapse to the terminal horizon before scoring.

    Without the collapse, CumulativePinball sums the cumulative prediction across
    every horizon (3+6+9=18) and compares it to the realised cumulative demand
    (9), inflating the loss; the terminal-only objective compares 9 vs 9.
    """
    qcol = quantile_column(0.5)
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"] * 3,
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")] * 3,
            DS: pd.date_range("2024-01-08", periods=3, freq="W-MON"),
            Y_HAT: [3.0, 6.0, 9.0],
            H: [1, 2, 3],
            MODEL_NAME: ["M"] * 3,
            qcol: [3.0, 6.0, 9.0],
        }
    )
    actuals = pd.Series([2.0, 3.0, 4.0])  # cumulative demand over the window = 9.0

    terminal = search_module._CumulativeTerminalPinball(
        quantile=0.5, tau=0.833, protection_period=3
    )
    raw = CumulativePinball(quantile=0.5, tau=0.833)

    assert terminal.evaluate(frame, actuals) == pytest.approx(0.0)
    assert raw.evaluate(frame, actuals) > terminal.evaluate(frame, actuals)


def test_run_benchmark_defaults_to_committed_best_config(monkeypatch) -> None:
    def _fail_hpo(*args, **kwargs):
        raise AssertionError("run_hpo should not be called when tune=False")

    monkeypatch.setattr(run_benchmark_module, "BEST_CONFIG", _FAST_BEST_CONFIG)
    monkeypatch.setattr(run_benchmark_module, "run_hpo", _fail_hpo)

    result = run_benchmark(
        data_dir=DATA_DIR,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=_get_first_n_series(3),
        results_dir=None,
        verbose=False,
    )

    assert not result.empty


def test_run_benchmark_tune_true_runs_hpo_and_threads_its_config(monkeypatch) -> None:
    """tune=True liveness lock: run_hpo fires exactly once and its returned
    config (not BEST_CONFIG) is the one threaded into the benchmark run."""
    hpo_calls: list[dict[str, Any]] = []
    threaded: dict[str, Any] = {}
    tuned_config = {**_FAST_BEST_CONFIG, "name": "hpo_tuned_lgbm"}

    def _fake_run_hpo(**kwargs):
        hpo_calls.append(kwargs)
        return tuned_config

    real_strip_private = run_benchmark_module.strip_private

    def _capture_strip_private(config):
        threaded["config"] = config
        return real_strip_private(config)

    monkeypatch.setattr(run_benchmark_module, "run_hpo", _fake_run_hpo)
    monkeypatch.setattr(run_benchmark_module, "strip_private", _capture_strip_private)

    result = run_benchmark(
        data_dir=DATA_DIR,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=_get_first_n_series(2),
        results_dir=None,
        verbose=False,
        tune=True,
        order_conformal_warmup_origins=1,
    )

    assert len(hpo_calls) == 1
    assert threaded["config"] is tuned_config
    assert not result.empty


def test_run_from_config_threads_tune_flag_to_run_benchmark(monkeypatch) -> None:
    """The run_from_config seam lock: the tune kwarg reaches run_benchmark in
    both directions — a regression lock against the formerly hardcoded
    tune=False that made the knob dead from the config entrypoint."""
    captured: dict[str, Any] = {}

    def _fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({UNIQUE_ID: ["A"], "total_cost": [0.0]})

    monkeypatch.setattr(run_benchmark_module, "run_benchmark", _fake_run_benchmark)
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": "data/vn2"},
            "tasks": [{"model": "global_lgbm", "horizon": 3, "config": {"backend": "mlforecast"}}],
            "origins": {"start": "2024-01-01", "end": "2024-01-01", "freq": "W-MON"},
            "output": {"streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )

    run_from_config(config, tune=True)
    assert captured["tune"] is True

    run_from_config(config)
    assert captured["tune"] is False


def test_cached_replay_matches_run_benchmark_on_small_subset() -> None:
    series = _get_first_n_series(2)
    expected = run_benchmark(
        data_dir=DATA_DIR,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=series,
        results_dir=None,
        verbose=False,
        best_config=_FAST_BEST_CONFIG,
        order_conformal_warmup_origins=1,
    )

    cache = build_replay_cache(
        data_dir=DATA_DIR,
        model_config=_FAST_BEST_CONFIG,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=series,
        order_conformal_warmup_origins=1,
    )
    actual = replay_cached_cost(cache)

    assert actual.total_cost == pytest.approx(float(expected["total_cost"].sum()))


def _small_replay_cache():
    return build_replay_cache(
        data_dir=DATA_DIR,
        model_config=_FAST_BEST_CONFIG,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=_get_first_n_series(2),
        order_conformal_warmup_origins=1,
    )


def _raise_policy_error(*_args: Any, **_kwargs: Any) -> Any:
    raise ValueError("synthetic policy failure")


def test_replay_policy_error_fails_fast(monkeypatch) -> None:
    """P0.4: a policy failure aborts the replay by default — never zero orders."""
    cache = _small_replay_cache()
    monkeypatch.setattr(replay_module, "apply_order_policy", _raise_policy_error)

    with pytest.raises(ValueError, match="synthetic policy failure"):
        replay_cached_cost(cache)


def test_run_benchmark_with_cumulative_target_config_produces_finite_cost() -> None:
    """End-to-end smoke for the direct-cumulative target plumbing."""
    cumulative_config = {**_FAST_BEST_CONFIG, "_target_mode": "cumulative"}
    result = run_benchmark(
        data_dir=DATA_DIR,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=_get_first_n_series(2),
        results_dir=None,
        verbose=False,
        best_config=cumulative_config,
        order_conformal_warmup_origins=1,
    )

    assert not result.empty
    total = float(result["total_cost"].sum())
    assert total >= 0.0
    assert pd.notna(total)


def test_cost_search_smoke_runs_one_cached_trial() -> None:
    study = run_cost_search(
        data_dir=DATA_DIR,
        model_config=_FAST_BEST_CONFIG,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=1,
        series_filter=_get_first_n_series(2),
        n_trials=1,
        seed=0,
    )

    assert len(study.trials) == 1
    assert study.best_value >= 0.0


def test_cost_trial_failure_report_classifies_failures() -> None:
    """P1.3: bad trials report inf; infra failures re-raise, never become inf."""
    pruned = search_module._cost_trial_failure_report(optuna.TrialPruned())
    assert pruned[search_module.OBJECTIVE_METRIC] == float("inf")
    assert pruned["pruned"] == 1

    bad = search_module._cost_trial_failure_report(ValueError("bad params"))
    assert bad[search_module.OBJECTIVE_METRIC] == float("inf")
    assert "bad params" in bad["bad_trial"]

    key = search_module._cost_trial_failure_report(KeyError("missing"))
    assert key[search_module.OBJECTIVE_METRIC] == float("inf")
    assert "bad_trial" in key

    with pytest.raises(RuntimeError, match="infra boom"):
        search_module._cost_trial_failure_report(RuntimeError("infra boom"))


def test_cost_search_uses_ray_tune_scheduler_handoff(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeResult:
        error = None
        metrics = {"objective": 123.0}
        config = {"crc_enabled": False}

    class _FakeResults(list):
        def get_best_result(self, *args, **kwargs):
            return self[0]

    def _fake_run_optuna_study(**kwargs):
        captured.update(kwargs)
        return StudyOutcome(
            best_config={"crc_enabled": False},
            results=_FakeResults([_FakeResult()]),
        )

    monkeypatch.setattr(search_module, "run_optuna_study", _fake_run_optuna_study)

    result = run_cost_search(
        data_dir=DATA_DIR,
        model_config=_FAST_BEST_CONFIG,
        horizon=3,
        lead_time=2,
        review_period=1,
        decision_rounds=2,
        delivery_weeks=1,
        series_filter=_get_first_n_series(2),
        n_trials=2,
        timeout_sec=30,
        seed=0,
        search_forecast=True,
        max_concurrent_trials=1,
    )

    # Scheduler-handoff effect: the Tune result's objective + config flow back
    # out as the returned study's best trial (not just kwargs plumbing).
    assert result.best_value == pytest.approx(123.0)
    assert result.best_params == {"crc_enabled": False}
    # decision_rounds=2 + delivery_weeks=1 → max_t=3, grace_period stays below max_t.
    assert captured["n_trials"] == 2
    assert captured["max_t"] == 3
    assert captured["asha_grace_period"] == 1
    assert captured["max_concurrent_trials"] == 1
    assert captured["metric"] == "objective"
    assert captured["time_attr"] == "tune_step"
    assert captured["fail_fast"] == "raise"


def test_oracle_order_path_matches_simple_known_lead_time_case() -> None:
    orders = optimal_order_path_for_sku(
        ProductState(unique_id="A", end_inventory=0.0, in_transit_w1=0.0, in_transit_w2=0.0),
        {1: 0.0, 2: 0.0, 3: 5.0},
        decision_rounds=1,
        total_weeks=3,
    )

    assert orders == {1: 5.0}


def test_round_actuals_uses_current_round_demand() -> None:
    series = _get_first_n_series(3)
    expected = extract_new_actuals(DATA_DIR, 1)

    actuals = round_actuals(DATA_DIR, 1, {uid: object() for uid in series})

    assert actuals == {uid: expected.get(uid, 0.0) for uid in series}


def test_run_order_conformal_warmup_seeds_residual_pool() -> None:
    """Warmup helper produces at least one residual on real week_0 sales."""
    series = _get_first_n_series(3)
    sales = load_period(DATA_DIR, 0)
    sales = sales[sales["unique_id"].isin(series)]

    horizon = 3
    runtime = CumulativeRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=horizon,
            calibration_window=64,
            base_column=quantile_column(_FAST_BEST_CONFIG["_quantile_alpha"]),
            weight_decay=None,
        )
    )
    engine_config = {k: v for k, v in _FAST_BEST_CONFIG.items() if not k.startswith("_")}

    assert runtime.get_diagnostics()["n_scores"] == 0

    run_order_conformal_warmup(
        sales=sales,
        instock=None,
        model_config=engine_config,
        horizon=horizon,
        warmup_origins=2,
        runtime=runtime,
        series_filter=series,
    )

    assert runtime.get_diagnostics()["n_scores"] > 0
