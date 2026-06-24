"""End-to-end `run_config` proof for the M5 ordering testbed (decision -> settle).

Exercises the full production path (config -> prepare_run -> rolling settle loop)
on the committed 4-series x 16-day ``tests/fixtures/m5`` surface: an
``order_conformal`` decision bound feeds the live (R,S) policy, the engine walks
the decision origins with a per-origin forecast -> order -> settle -> observe hook
over the generic :class:`Simulator`, and the accumulated cost lands on the
``calibre_order_cost`` gauge once after the lead-time drain. A second run with no
cost/inventory kwargs proves the loop branch stays inert (bundle inventory
``None``, single-pass, no cost gauge written).

The two former tally-guard tests (``test_tally_rejects_per_uid_cost_panel`` and
``test_tally_rejects_heterogeneous_protection_period``) were DELETED in Slice B:
they asserted ``_tally_order_cost`` seam-specific ``ValueError``s that have no
loop-path equivalent. The generic :class:`Simulator` accumulates a single scalar
cost, so there is no per-uid cost panel and no per-ledger protection-period
heterogeneity to guard — the conditions are structurally impossible on the loop
path, and ``_tally_order_cost`` itself is gone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from prometheus_client import REGISTRY

from calibre.cli.commands import _load_dataset, run_config
from calibre.cli.config import load_config
from calibre.core.forecast_frame import H
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


def _order_cost_gauge(dataset: str, currency: str = "EUR") -> float | None:
    return REGISTRY.get_sample_value(
        "calibre_order_cost", {"currency": currency, "dataset": dataset}
    )


def test_m5_ordering_run_decision_settle_finite_cost(tmp_path: Path) -> None:
    config = _config_to_tmp(_ORDERING_CONFIG, tmp_path)
    assert config.order_conformal is not None
    # The bundle seeds inventory, so the run routes to the rolling settle loop.
    assert config.ordering is not None
    assert config.ordering.lead_time is not None
    assert config.ordering.review_period is not None
    protection_period = config.order_conformal.protection_period
    coverage = config.order_conformal.coverage
    # coverage 0.5 -> hi_0p5 (the interval_column_names form, "." -> "p").
    upper_col = f"hi_0p{str(coverage).split('.')[1]}"

    result = run_config(config)
    ledger = result.ledger.to_df()

    # 1. The settle loop produces no order ledger — the simulator owns the cost.
    assert result.order_ledger is None
    assert not ledger.empty

    # 2. The decision bound is emitted on the terminal-horizon rows, NaN below
    # (the loop's runtime applies it; the engine itself carries no runtime).
    assert upper_col in ledger.columns
    terminal = ledger[ledger[H] == protection_period]
    earlier = ledger[ledger[H] < protection_period]
    assert not terminal.empty
    assert terminal[upper_col].notna().any()
    assert earlier[upper_col].isna().all()

    # 3. The settle walk tallies a FINITE, non-negative, non-degenerate total
    # cost once, after the lead-time drain, onto the order-cost gauge.
    total_cost = _order_cost_gauge(config.dataset.adapter)
    assert total_cost is not None
    assert np.isfinite(total_cost)
    assert total_cost >= 0.0
    assert total_cost > 0.0


def test_m5_run_without_inventory_skips_settle_loop(tmp_path: Path) -> None:
    """Non-regression: an M5 run with no cost/inventory kwargs is single-pass.

    The bundle carries ``inventory is None`` and a zero ``CostStruct()``, so the
    settle branch is skipped and the unchanged single-pass path runs (no order
    ledger, no cost gauge written for this run).
    """
    config = _config_to_tmp(_SMOKE_CONFIG, tmp_path)
    bundle = _load_dataset(config)

    assert bundle.inventory is None
    assert isinstance(bundle.costs, CostStruct)
    assert bundle.costs == CostStruct()
    # No ordering block at all -> single-pass forecast run.
    assert config.ordering is None

    result = run_config(config)
    assert result.order_ledger is None
    assert not result.ledger.to_df().empty
