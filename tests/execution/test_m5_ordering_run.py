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

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from calibre.cli.commands import _load_dataset, _tally_order_cost, run_config
from calibre.cli.config import load_config
from calibre.core.forecast_frame import FORECAST_ORIGIN, UNIQUE_ID, H
from calibre.core.order_types import CostStruct
from calibre.execution.backend import BackendResult
from calibre.execution.ledger import InMemoryOrderLedger

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

    # 4. SANE / demand-responsive orders: all >= 0, no NaN, and resolved demand
    # varies ACROSS the four series at a fixed origin (true cross-series spread,
    # not single-series variation across origins).
    assert (orders["order_qty"] >= 0).all()
    assert orders["order_qty"].notna().all()
    window = ledger[ledger[H] <= protection_period]
    resolved_window = window.dropna(subset=["y"])
    # Pick an origin whose protection window resolves multiple series, then
    # assert their summed demand is not all identical.
    series_per_origin = resolved_window.groupby(FORECAST_ORIGIN)[UNIQUE_ID].nunique()
    multi_series_origins = series_per_origin[series_per_origin > 1].index
    assert len(multi_series_origins) > 0
    origin = multi_series_origins[0]
    demand_across_series = (
        resolved_window[resolved_window[FORECAST_ORIGIN] == origin].groupby(UNIQUE_ID)["y"].sum()
    )
    assert demand_across_series.nunique() > 1


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


def test_tally_rejects_per_uid_cost_panel(tmp_path: Path) -> None:
    """A per-uid ``dict[str, CostStruct]`` cost panel trips the tally guard.

    The seam tallies against a single uniform cost struct; a per-uid panel is
    out of scope and must raise rather than silently pick one struct.
    """
    config = _config_to_tmp(_ORDERING_CONFIG, tmp_path)
    bundle = _load_dataset(config)
    result = run_config(config)
    # Sanity: the real ordering path supplies inventory and a non-empty ledger,
    # so the guard is reached (not short-circuited by the None gates above it).
    assert bundle.inventory is not None
    assert not result.order_ledger.to_df().empty

    assert isinstance(bundle.costs, CostStruct)
    cost_panel = {uid: bundle.costs for uid in bundle.inventory}
    panel_bundle = dataclasses.replace(bundle, costs=cost_panel)

    with pytest.raises(ValueError, match="per-uid cost panel"):
        _tally_order_cost(panel_bundle, result)


def test_tally_rejects_heterogeneous_protection_period(tmp_path: Path) -> None:
    """A heterogeneous-``protection_period`` order ledger trips the tally guard.

    The protection window is applied globally from the first row; a ledger
    carrying more than one ``protection_period`` would silently mis-window, so
    the seam rejects it.
    """
    config = _config_to_tmp(_ORDERING_CONFIG, tmp_path)
    bundle = _load_dataset(config)
    result = run_config(config)
    assert bundle.inventory is not None

    order_frame = result.order_ledger.to_df()
    assert not order_frame.empty
    # Make protection_period non-uniform across the (uid, origin) rows.
    heterogeneous = order_frame.copy()
    heterogeneous.loc[heterogeneous.index[0], "protection_period"] = (
        int(heterogeneous["protection_period"].iloc[0]) + 1
    )
    assert heterogeneous["protection_period"].nunique() > 1

    poisoned_ledger = InMemoryOrderLedger()
    poisoned_ledger.append(heterogeneous)
    poisoned_result = BackendResult(ledger=result.ledger, order_ledger=poisoned_ledger)

    with pytest.raises(ValueError, match="single protection period"):
        _tally_order_cost(bundle, poisoned_result)
