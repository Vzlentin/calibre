"""Integration test for the VN2 seasonal-naive smoke pipeline.

Uses a small subset (3 products), 2 decision rounds, SeasonalNaive only,
and skips warmup to keep runtime fast.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.vn2.run_seasonal import run_seasonal
from benchmarks.vn2.simulator import load_initial_states
from calibre.conformal.runtime import ConformalPolicyConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "vn2"

_FAST_MODEL_CONFIGS = [
    {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52},
]

_FAST_CONFORMAL_CONFIG = ConformalPolicyConfig(
    method="aci",
    coverage=0.9,
    gamma=0.05,
    calibration_window=10,
)


def _get_first_n_series(n: int) -> list[str]:
    """Return the first n unique_ids from week_0_initial_state.csv."""
    states = load_initial_states(DATA_DIR / "week_0_initial_state.csv")
    return sorted(states.keys())[:n]


def _run(series: list[str]):
    return run_seasonal(
        data_dir=DATA_DIR,
        model_configs=_FAST_MODEL_CONFIGS,
        conformal_config=_FAST_CONFORMAL_CONFIG,
        horizon=3,
        warmup_origins=0,  # skip warmup for speed
        lead_time=2,
        review_period=1,
        decision_rounds=2,
        delivery_weeks=1,
        series_filter=series,
        results_dir=None,
        verbose=False,
    )


class TestVN2SeasonalIntegration:
    def test_full_loop_runs_without_error(self) -> None:
        """Full benchmark loop completes without raising exceptions."""
        result = _run(_get_first_n_series(3))
        assert result is not None
        assert not result.empty

    def test_costs_are_non_negative(self) -> None:
        """All cost columns must be non-negative."""
        result = _run(_get_first_n_series(3))
        assert (result["holding_cost"] >= 0).all(), "Negative holding costs detected"
        assert (result["shortage_cost"] >= 0).all(), "Negative shortage costs detected"
        assert (result["total_cost"] >= 0).all(), "Negative total costs detected"

    def test_result_has_expected_columns(self) -> None:
        """Result DataFrame has the required columns."""
        result = _run(_get_first_n_series(3))
        expected_cols = {"unique_id", "holding_cost", "shortage_cost", "total_cost"}
        assert expected_cols.issubset(set(result.columns))

    def test_total_cost_equals_holding_plus_shortage(self) -> None:
        """total_cost column must equal holding_cost + shortage_cost for every row."""
        result = _run(_get_first_n_series(3))
        computed = result["holding_cost"] + result["shortage_cost"]
        for idx, (actual, expected) in enumerate(zip(result["total_cost"], computed, strict=False)):
            assert abs(actual - expected) < 1e-9, (
                f"Row {idx}: total_cost {actual} != holding + shortage {expected}"
            )

    def test_result_has_one_row_per_product(self) -> None:
        """Result has exactly one row per series in series_filter."""
        series = _get_first_n_series(3)
        result = _run(series)
        assert len(result) == len(series)
        assert set(result["unique_id"]) == set(series)
