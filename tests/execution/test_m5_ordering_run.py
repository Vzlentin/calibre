"""End-to-end `run_config` proof for the M5 ordering testbed (decision -> order -> cost).

Exercises the full production path (config -> prepare_run -> BackendEngine ->
post-run cost tally) on the committed 4-series x 16-day ``tests/fixtures/m5``
surface: an ``order_conformal`` decision bound feeds the R,S policy, whose order
ledger is replayed against realized fixture demand through the generic
:class:`Simulator` to a finite, sane order cost. A second run with no
cost/inventory kwargs proves the seam stays inert (bundle inventory ``None``,
zero costs, tally skipped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibre.cli.commands import _load_dataset, _tally_order_cost, run_config
from calibre.cli.config import load_config
from calibre.core.forecast_frame import FORECAST_ORIGIN, UNIQUE_ID, H
from calibre.core.order_types import CostStruct

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORDERING_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "smoke-ordering.yaml"
_SMOKE_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "smoke.yaml"


def _config_to_tmp(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    updates: dict[str, object] = {"ledger_path": str(tmp_path / "forecast-ledger.parquet")}
    if config.output.order_ledger_path is not None:
        updates["order_ledger_path"] = str(tmp_path / "order-ledger.parquet")
    return config.model_copy(update={"output": config.output.model_copy(update=updates)})


def test_m5_ordering_run_decision_order_finite_cost(tmp_path: Path) -> None:
    config = _config_to_tmp(_ORDERING_CONFIG, tmp_path)
    assert config.order_conformal is not None
    protection_period = config.order_conformal.protection_period
    coverage = config.order_conformal.coverage
    # coverage 0.5 -> hi_0p5 (the interval_column_names form, "." -> "p").
    upper_col = f"hi_0p{str(coverage).split('.')[1]}"

    bundle = _load_dataset(config)
    result = run_config(config)
    ledger = result.ledger.to_df()

    # 1. Bound emitted on the terminal-horizon rows, NaN below (production path).
    assert upper_col in ledger.columns
    terminal = ledger[ledger[H] == protection_period]
    earlier = ledger[ledger[H] < protection_period]
    assert not terminal.empty
    assert terminal[upper_col].notna().any()
    assert earlier[upper_col].isna().all()

    # 2. R,S consumes the bound: order_qty == max(target_stock_level - ip, 0).
    orders = result.order_ledger.to_df()
    assert not orders.empty
    assert orders["order_qty"].notna().all()
    assert orders["target_stock_level"].notna().all()
    expected_qty = (orders["target_stock_level"] - orders["inventory_position"]).clip(lower=0.0)
    np.testing.assert_allclose(
        orders["order_qty"].to_numpy(), expected_qty.to_numpy(), rtol=0, atol=0
    )

    # 3. Simulator tallies a FINITE, non-negative, non-degenerate total cost.
    tally = _tally_order_cost(bundle, result)
    assert tally is not None
    assert {"holding_cost", "shortage_cost"} <= set(tally.columns)
    total_cost = float((tally["holding_cost"] + tally["shortage_cost"]).sum())
    assert np.isfinite(total_cost)
    assert total_cost >= 0.0
    assert total_cost > 0.0
    assert (tally["demand"] > 0).any()

    # 4. SANE / demand-responsive orders: all >= 0, no NaN, and the per-(uid,
    # origin) resolved demands are not all identical across the four series.
    assert (orders["order_qty"] >= 0).all()
    assert orders["order_qty"].notna().all()
    window = ledger[ledger[H] <= protection_period]
    demand_by_series = window.groupby([UNIQUE_ID, FORECAST_ORIGIN])["y"].sum().dropna()
    assert demand_by_series.nunique() > 1


def test_m5_run_without_costs_keeps_inventory_none_and_zero_costs(tmp_path: Path) -> None:
    """Non-regression: an M5 run with no cost/inventory kwargs is inert.

    The bundle carries ``inventory is None`` and a zero ``CostStruct()``, so the
    tally seam is skipped and the unchanged path runs (criterion 5).
    """
    config = _config_to_tmp(_SMOKE_CONFIG, tmp_path)
    bundle = _load_dataset(config)

    assert bundle.inventory is None
    assert isinstance(bundle.costs, CostStruct)
    assert bundle.costs == CostStruct()

    result = run_config(config)
    # The tally seam is gated off (no inventory), so it returns None even though
    # this config records no order ledger either.
    assert _tally_order_cost(bundle, result) is None
    assert not result.ledger.to_df().empty
