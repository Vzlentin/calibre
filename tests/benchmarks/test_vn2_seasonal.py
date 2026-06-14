"""Integration test for the VN2 seasonal-naive smoke pipeline.

Exercises ``run_seasonal`` — the legacy reference path that drives
``SymmetricIntervalRuntime`` (ACI), the summed-conformal ``apply_rs_policy``,
and ``VN2Simulator`` end-to-end — which the tuned ``run_benchmark`` pipeline
(covered in ``test_vn2_benchmark.py``) bypasses. Uses a 3-product subset, 2
decision rounds, SeasonalNaive only, and skips warmup to keep runtime fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.vn2.run_seasonal import run_seasonal
from benchmarks.vn2.simulator import load_initial_states
from calibre.conformal.runtime import SymmetricIntervalConfig

DATA_DIR = Path(__file__).parents[2] / "data" / "vn2"

_FAST_MODEL_CONFIGS = [
    {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52},
]

_FAST_CONFORMAL_CONFIG = SymmetricIntervalConfig(
    method="aci",
    coverage=0.9,
    gamma=0.05,
    calibration_window=10,
)


def _get_first_n_series(n: int) -> list[str]:
    states = load_initial_states(DATA_DIR / "week_0_initial_state.csv")
    return sorted(states.keys())[:n]


def _run(series: list[str], warmup_origins: int = 0):
    return run_seasonal(
        data_dir=DATA_DIR,
        model_configs=_FAST_MODEL_CONFIGS,
        conformal_config=_FAST_CONFORMAL_CONFIG,
        horizon=3,
        warmup_origins=warmup_origins,  # 0 skips warmup for speed
        lead_time=2,
        review_period=1,
        decision_rounds=2,
        delivery_weeks=1,
        series_filter=series,
        results_dir=None,
        verbose=False,
    )


def test_seasonal_pipeline_produces_valid_per_product_costs() -> None:
    """The seasonal-naive reference loop yields one finite, accounted cost row per series.

    Structural/relational invariants only — never a literal ``total_cost`` (VN2
    cost is cross-arch float-divergent).
    """
    series = _get_first_n_series(3)
    result = _run(series)

    assert set(result["unique_id"]) == set(series)
    assert len(result) == len(series)
    assert {"unique_id", "holding_cost", "shortage_cost", "total_cost"}.issubset(result.columns)

    costs = result[["holding_cost", "shortage_cost", "total_cost"]].to_numpy()
    assert np.isfinite(costs).all()
    assert costs.min() >= 0.0
    np.testing.assert_allclose(
        result["total_cost"].to_numpy(),
        (result["holding_cost"] + result["shortage_cost"]).to_numpy(),
    )


def test_seasonal_warmup_runs_through_observe_pending() -> None:
    """Warmup (warmup_origins=1) drives ``_run_warmup`` through ``observe_pending``.

    Covers the previously-untested warmup observe block after the dispatcher
    cutover (the DS-gate / ``_split_ready`` / local suppression deletions).
    The pipeline must still complete and yield finite, accounted costs.
    """
    series = _get_first_n_series(3)
    result = _run(series, warmup_origins=1)

    assert set(result["unique_id"]) == set(series)
    costs = result[["holding_cost", "shortage_cost", "total_cost"]].to_numpy()
    assert np.isfinite(costs).all()
    assert costs.min() >= 0.0


def test_seasonal_warmup_accumulates_calibration_across_origins() -> None:
    """With two warmup origins, round 2 observes round-1 frames and accumulates scores.

    Round 2 observes round-1 frames via the dispatcher and the calibrator
    accumulates scores. Locks the folded warmup observe path against a silent
    no-op regression:
    at warmup_origins=1 the loop runs once with empty pending and never
    reaches observe, so the smoke test alone cannot catch a warmup that
    stopped calibrating.
    """
    from benchmarks.vn2.run_seasonal import _run_warmup, load_period

    series = _get_first_n_series(3)
    sales = load_period(DATA_DIR, 0)
    sales = sales[sales["unique_id"].isin(series)]

    runtime = _run_warmup(
        sales=sales,
        model_configs=_FAST_MODEL_CONFIGS,
        horizon=3,
        warmup_origins=2,
        conformal_config=_FAST_CONFORMAL_CONFIG,
        series_filter=series,
    )

    score_history = runtime.get_diagnostics().get("calibrator", {}).get("score_history", {})
    assert score_history, "round-2 observe must score the calibrator"
    assert any(len(scores) > 0 for scores in score_history.values())
