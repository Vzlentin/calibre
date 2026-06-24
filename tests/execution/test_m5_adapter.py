"""Tests for the M5 dataset adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from calibre.core.order_types import CostStruct
from calibre.execution.m5_adapter import M5DatasetAdapter
from calibre.execution.validation import validate_dataset_bundle
from calibre.ordering.simulation.state import ProductState

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"

_ORDERING_KWARGS = dict(
    underage_cost=1.0,
    overage_cost=0.2,
    holding_cost=0.2,
    shortage_cost=1.0,
    initial_inventory=5.0,
    lead_time=1,
    review_period=1,
)


def test_m5_adapter_loads_bundle() -> None:
    bundle = M5DatasetAdapter().load(_FIXTURE)
    assert bundle.hierarchy is not None
    assert bundle.future_x is None
    assert bundle.censoring is None
    assert isinstance(bundle.costs, CostStruct)

    history_uids = set(bundle.history["unique_id"].astype(str))
    hierarchy_uids = set(bundle.hierarchy["unique_id"].astype(str))
    assert history_uids <= hierarchy_uids
    for col in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        assert col in bundle.hierarchy.columns

    validate_dataset_bundle(bundle)


def test_m5_adapter_missing_calendar_raises(tmp_path: Path) -> None:
    sales = _FIXTURE / "sales_train_evaluation.csv"
    (tmp_path / "sales_train_evaluation.csv").write_text(sales.read_text(), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="calendar.csv"):
        M5DatasetAdapter().load(tmp_path)


def test_m5_adapter_default_load_keeps_zero_costs_and_no_inventory() -> None:
    """No cost/inventory kwargs -> byte-identical to the pre-ordering path.

    Pins today's behavior: a zero ``CostStruct()`` and ``inventory is None`` so
    the run_config tally seam is skipped and the VN2/diagnostic paths are
    unaffected.
    """
    bundle = M5DatasetAdapter().load(_FIXTURE)

    assert isinstance(bundle.costs, CostStruct)
    assert bundle.costs == CostStruct()
    assert bundle.costs.holding_cost == 0.0
    assert bundle.costs.shortage_cost == 0.0
    assert bundle.inventory is None


def test_m5_adapter_builds_costs_and_inventory_from_kwargs() -> None:
    bundle = M5DatasetAdapter().load(_FIXTURE, **_ORDERING_KWARGS)

    assert isinstance(bundle.costs, CostStruct)
    assert bundle.costs.critical_ratio == pytest.approx(0.8333, abs=1e-3)
    assert bundle.costs.holding_cost == 0.2
    assert bundle.costs.shortage_cost == 1.0

    assert isinstance(bundle.inventory, dict)
    history_uids = set(bundle.history["unique_id"].astype(str))
    assert set(bundle.inventory) == history_uids
    for state in bundle.inventory.values():
        assert isinstance(state, ProductState)
        assert state.lead_time_depth == 1
        assert state.end_inventory == 5.0

    validate_dataset_bundle(bundle)


def test_m5_adapter_ordering_requires_positive_lead_time() -> None:
    with pytest.raises(ValueError, match="lead_time >= 1"):
        M5DatasetAdapter().load(_FIXTURE, **{**_ORDERING_KWARGS, "lead_time": 0})


def test_validate_dataset_bundle_rejects_out_of_history_inventory_key() -> None:
    bundle = M5DatasetAdapter().load(_FIXTURE, **_ORDERING_KWARGS)
    assert bundle.inventory is not None
    rogue = dict(bundle.inventory)
    rogue["NOT_A_REAL_UID"] = ProductState(
        unique_id="NOT_A_REAL_UID", end_inventory=0.0, pipeline=next(iter(rogue.values())).pipeline
    )
    polluted = replace(bundle, inventory=rogue)
    with pytest.raises(ValueError, match="inventory contains unknown unique_id"):
        validate_dataset_bundle(polluted)
