"""Smoke test for the tuned VN2 benchmark (global LGBM + HPO + R,S).

Bypasses the live HPO by passing a pre-built ``best_config``, then runs a
single decision round + delivery week on a 3-series subset of the real
VN2 dataset. Verifies the public surface of ``run_benchmark`` and the
per-product cost dataframe shape.
"""

from __future__ import annotations

from pathlib import Path

from mlforecast.lag_transforms import RollingMean

from benchmarks.vn2.run_benchmark import (
    _round_actuals,
    _run_order_conformal_warmup,
    run_benchmark,
    run_hpo,
)
from benchmarks.vn2.simulator import extract_new_actuals, load_initial_states
from calibre.conformal.crc import CumulativeConformalRiskConfig, CumulativeConformalRiskRuntime
from calibre.contracts.forecast_frame import quantile_column
from calibre.pipeline.loading import load_period

DATA_DIR = Path(__file__).parent.parent / "data" / "vn2"

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


class TestVN2BenchmarkIntegration:
    def test_full_loop_runs_without_error(self) -> None:
        result = _run(_get_first_n_series(3))
        assert result is not None
        assert not result.empty

    def test_costs_are_non_negative(self) -> None:
        result = _run(_get_first_n_series(3))
        assert (result["holding_cost"] >= 0).all(), "Negative holding costs detected"
        assert (result["shortage_cost"] >= 0).all(), "Negative shortage costs detected"
        assert (result["total_cost"] >= 0).all(), "Negative total costs detected"

    def test_result_has_expected_columns(self) -> None:
        result = _run(_get_first_n_series(3))
        expected_cols = {"unique_id", "holding_cost", "shortage_cost", "total_cost"}
        assert expected_cols.issubset(set(result.columns))

    def test_total_cost_equals_holding_plus_shortage(self) -> None:
        result = _run(_get_first_n_series(3))
        computed = result["holding_cost"] + result["shortage_cost"]
        for idx, (actual, expected) in enumerate(zip(result["total_cost"], computed, strict=False)):
            assert abs(actual - expected) < 1e-9, (
                f"Row {idx}: total_cost {actual} != holding + shortage {expected}"
            )

    def test_result_has_one_row_per_product(self) -> None:
        series = _get_first_n_series(3)
        result = _run(series)
        assert len(result) == len(series)
        assert set(result["unique_id"]) == set(series)


def test_run_hpo_returns_valid_config() -> None:
    """run_hpo with 1 trial returns a complete, engine-ready model config."""
    series = _get_first_n_series(2)
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


def test_round_actuals_uses_current_round_demand() -> None:
    series = _get_first_n_series(3)
    expected = extract_new_actuals(DATA_DIR, 1)

    actuals = _round_actuals(DATA_DIR, 1, {uid: object() for uid in series})

    assert actuals == {uid: expected.get(uid, 0.0) for uid in series}


def test_run_order_conformal_warmup_seeds_residual_pool() -> None:
    """Warmup helper produces at least one residual on real week_0 sales."""
    series = _get_first_n_series(3)
    sales = load_period(DATA_DIR, 0)
    sales = sales[sales["unique_id"].isin(series)]

    horizon = 3
    runtime = CumulativeConformalRiskRuntime(
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

    _run_order_conformal_warmup(
        sales=sales,
        instock=None,
        model_config=engine_config,
        horizon=horizon,
        warmup_origins=2,
        runtime=runtime,
        series_filter=series,
    )

    assert runtime.get_diagnostics()["n_scores"] > 0
