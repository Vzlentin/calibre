"""Tests for the inventory-position adapter."""

from __future__ import annotations

import pandas as pd

from benchmarks.vn2.simulator import VN2Simulator
from calibre.execution.dataset import SnapshotInventoryAdapter, SyntheticInventoryAdapter
from calibre.ordering.simulation.state import ProductState, make_pipeline


def test_synthetic_matches_today() -> None:
    state = ProductState(
        unique_id="A",
        end_inventory=10.0,
        pipeline=make_pipeline([1.0, 2.0], 2),
        cumulative_costs={"holding": 3.0},
    )
    adapter = SyntheticInventoryAdapter({"A": state})

    loaded = adapter.load_state("A", pd.Timestamp("2024-01-01"))

    assert loaded == state
    assert loaded is not state
    assert adapter.load_lead_times() == {"A": 2}


def test_snapshot_loads_from_parquet(tmp_path) -> None:
    snapshot_path = tmp_path / "inventory.parquet"
    pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "as_of": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")],
            "end_inventory": [10.0, 12.0],
            "pipeline_0": [1.0, 3.0],
            "pipeline_1": [2.0, 4.0],
            "lead_time_depth": [2, 2],
        }
    ).to_parquet(snapshot_path)
    adapter = SnapshotInventoryAdapter(snapshot_path)

    state = adapter.load_state("A", pd.Timestamp("2024-01-09"))

    assert state.end_inventory == 12.0
    assert list(state.pipeline) == [3.0, 4.0]
    assert adapter.load_lead_times() == {"A": 2}


def test_injected_state_propagates_to_simulator() -> None:
    state = ProductState(
        unique_id="A",
        end_inventory=7.0,
        pipeline=make_pipeline([2.0, 0.0], 2),
    )

    simulator = VN2Simulator({"A": state})
    [result] = simulator.step(1, orders={"A": 5.0}, actual_demand={"A": 3.0})

    assert result.start_inventory == 9.0
    assert simulator.states["A"].in_transit_w2 == 5.0
