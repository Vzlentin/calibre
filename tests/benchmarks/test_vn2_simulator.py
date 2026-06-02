"""Tests for the VN2 inventory simulator."""

from __future__ import annotations

from pathlib import Path

import fsspec
import pytest

from benchmarks.vn2.simulator import (
    ProductState,
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)

DATA_DIR = Path(__file__).parents[2] / "data" / "vn2"
INITIAL_STATE_PATH = DATA_DIR / "week_0_initial_state.csv"


def _single_product_sim(
    end_inventory: float = 0.0,
    in_transit_w1: float = 0.0,
    in_transit_w2: float = 0.0,
) -> VN2Simulator:
    states = {
        "1_1": ProductState(
            unique_id="1_1",
            end_inventory=end_inventory,
            in_transit_w1=in_transit_w1,
            in_transit_w2=in_transit_w2,
        )
    }
    return VN2Simulator(states)


class TestLeadTime:
    def test_order_placed_week1_arrives_week3(self) -> None:
        """Order placed at end of week 1 arrives at start of week 3 (2-week lead time).

        The simulator uses two in-transit slots: in_transit_w1 (arrives next week)
        and in_transit_w2 (arrives in 2 weeks). An order placed in week 1 enters
        in_transit_w2, shifts to in_transit_w1 at end of week 2, and arrives at
        start of week 3. Config uses lead_time=2.
        """
        sim = _single_product_sim(end_inventory=0.0, in_transit_w1=0.0, in_transit_w2=0.0)

        # Week 1: place order of 100, zero demand → order enters in_transit_w2
        sim.step(1, orders={"1_1": 100.0}, actual_demand={"1_1": 0.0})
        # Week 2: no new order, no demand → order shifts w2→w1, no arrivals yet
        w2 = sim.step(2, orders={"1_1": 0.0}, actual_demand={"1_1": 0.0})[0]
        assert w2.arrivals == pytest.approx(0.0), "Order should not arrive in week 2"

        # Week 3: order arrives (in_transit_w1 from previous step)
        results_w3 = sim.step(3, orders={"1_1": 0.0}, actual_demand={"1_1": 0.0})
        w3 = results_w3[0]

        assert w3.arrivals == pytest.approx(100.0)
        assert w3.start_inventory == pytest.approx(100.0)
        assert w3.end_inventory == pytest.approx(100.0)

    def test_order_not_available_in_week2(self) -> None:
        sim = _single_product_sim(end_inventory=0.0)

        sim.step(1, orders={"1_1": 50.0}, actual_demand={"1_1": 0.0})
        w2 = sim.step(2, orders={}, actual_demand={"1_1": 0.0})[0]

        assert w2.arrivals == pytest.approx(0.0)


class TestStockout:
    def test_demand_exceeds_inventory_caps_sales(self) -> None:
        sim = _single_product_sim(end_inventory=10.0)
        results = sim.step(1, orders={}, actual_demand={"1_1": 25.0})
        r = results[0]

        assert r.sales == pytest.approx(10.0)
        assert r.missed_sales == pytest.approx(15.0)
        assert r.end_inventory == pytest.approx(0.0)

    def test_zero_inventory_all_missed(self) -> None:
        sim = _single_product_sim(end_inventory=0.0)
        r = sim.step(1, orders={}, actual_demand={"1_1": 5.0})[0]

        assert r.sales == pytest.approx(0.0)
        assert r.missed_sales == pytest.approx(5.0)
        assert r.shortage_cost == pytest.approx(5.0)


class TestCostAccumulation:
    def test_holding_cost_rate(self) -> None:
        """Holding cost = end_inventory * 0.2."""
        sim = _single_product_sim(end_inventory=50.0)
        r = sim.step(1, orders={}, actual_demand={"1_1": 0.0})[0]
        assert r.holding_cost == pytest.approx(50.0 * 0.2)

    def test_shortage_cost_rate(self) -> None:
        """Shortage cost = missed_sales * 1.0."""
        sim = _single_product_sim(end_inventory=0.0)
        r = sim.step(1, orders={}, actual_demand={"1_1": 7.0})[0]
        assert r.shortage_cost == pytest.approx(7.0 * 1.0)

    def test_cumulative_across_three_weeks(self) -> None:
        sim = _single_product_sim(end_inventory=10.0)

        # Week 1: sell 3, end=7, hold=1.4, short=0
        sim.step(1, orders={}, actual_demand={"1_1": 3.0})
        # Week 2: sell 5, end=2, hold=0.4, short=0
        sim.step(2, orders={}, actual_demand={"1_1": 5.0})
        # Week 3: sell 5 but only 2 available, end=0, hold=0, short=3
        sim.step(3, orders={}, actual_demand={"1_1": 5.0})

        state = sim.states["1_1"]
        expected_holding = 7.0 * 0.2 + 2.0 * 0.2 + 0.0 * 0.2
        expected_shortage = 0.0 + 0.0 + 3.0 * 1.0
        assert state.cumulative_holding_cost == pytest.approx(expected_holding)
        assert state.cumulative_shortage_cost == pytest.approx(expected_shortage)
        assert sim.total_cost() == pytest.approx(expected_holding + expected_shortage)

    def test_total_cost_multi_product(self) -> None:
        states = {
            "A": ProductState("A", end_inventory=10.0, in_transit_w1=0.0, in_transit_w2=0.0),
            "B": ProductState("B", end_inventory=20.0, in_transit_w1=0.0, in_transit_w2=0.0),
        }
        sim = VN2Simulator(states)
        sim.step(1, orders={}, actual_demand={"A": 0.0, "B": 0.0})

        # A: end=10, hold=2.0; B: end=20, hold=4.0
        assert sim.total_cost() == pytest.approx(6.0)


class TestInventoryAccountingInvariant:
    def test_start_minus_sales_equals_end(self) -> None:
        """Invariant: start_inventory - sales = end_inventory."""
        sim = _single_product_sim(end_inventory=15.0, in_transit_w1=5.0)
        for week, demand in enumerate([8.0, 30.0, 0.0], start=1):
            results = sim.step(week, orders={}, actual_demand={"1_1": demand})
            r = results[0]
            assert r.start_inventory - r.sales == pytest.approx(r.end_inventory), (
                f"Week {week}: invariant broken"
            )


class TestLoadInitialStates:
    def test_loads_from_csv(self) -> None:
        states = load_initial_states(INITIAL_STATE_PATH)
        assert len(states) > 0
        for uid, state in states.items():
            assert "_" in uid
            assert isinstance(state, ProductState)
            assert state.end_inventory >= 0
            assert state.in_transit_w1 >= 0
            assert state.in_transit_w2 >= 0

    def test_unique_id_format(self) -> None:
        states = load_initial_states(INITIAL_STATE_PATH)
        for uid in states:
            parts = uid.split("_")
            assert len(parts) == 2, f"unexpected uid format: {uid}"
            assert parts[0].isdigit()
            assert parts[1].isdigit()

    def test_simulator_deep_copy(self) -> None:
        """VN2Simulator deep-copies states so stepping never mutates the caller's dict."""
        states = load_initial_states(INITIAL_STATE_PATH)
        original_uid = next(iter(states))
        original_end_inv = states[original_uid].end_inventory

        sim = VN2Simulator(states)
        sim.step(1, orders={}, actual_demand={uid: 0.0 for uid in states})

        # Original states dict must be unchanged
        assert states[original_uid].end_inventory == original_end_inv


class TestToDataFrame:
    def test_history_dataframe_records_step_values(self) -> None:
        sim = _single_product_sim(end_inventory=5.0)
        sim.step(1, orders={}, actual_demand={"1_1": 2.0})
        df = sim.to_dataframe()

        expected_cols = {
            "unique_id",
            "week",
            "start_inventory",
            "arrivals",
            "demand",
            "sales",
            "missed_sales",
            "end_inventory",
            "holding_cost",
            "shortage_cost",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 1

        # Start 5, no arrivals, demand 2 fully met → end 3, holding 3 * 0.2.
        row = df.iloc[0]
        assert row["unique_id"] == "1_1"
        assert row["week"] == 1
        assert row["start_inventory"] == pytest.approx(5.0)
        assert row["arrivals"] == pytest.approx(0.0)
        assert row["demand"] == pytest.approx(2.0)
        assert row["sales"] == pytest.approx(2.0)
        assert row["missed_sales"] == pytest.approx(0.0)
        assert row["end_inventory"] == pytest.approx(3.0)
        assert row["holding_cost"] == pytest.approx(0.6)
        assert row["shortage_cost"] == pytest.approx(0.0)


class TestExtractNewActuals:
    def test_extracts_week_1_actuals(self) -> None:
        actuals = extract_new_actuals(DATA_DIR, 1)
        assert len(actuals) > 0
        for uid, val in actuals.items():
            assert "_" in uid
            assert isinstance(val, float)
            assert val >= 0.0

    def test_unique_id_format(self) -> None:
        actuals = extract_new_actuals(DATA_DIR, 1)
        for uid in actuals:
            parts = uid.split("_")
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()


def test_vn2_simulator_loads_initial_state_and_actuals_from_fsspec_uri() -> None:
    fs = fsspec.filesystem("memory")
    root = "calibre-vn2-simulator"
    if fs.exists(root):
        fs.rm(root, recursive=True)
    fs.mkdirs(root)
    with fs.open(f"{root}/week_0_initial_state.csv", "wt", encoding="utf-8") as handle:
        handle.write("Store,Product,End Inventory,In Transit W+1,In Transit W+2\n1,10,5,1,2\n")
    with fs.open(f"{root}/week_0_sales.csv", "wt", encoding="utf-8") as handle:
        handle.write("Store,Product,2024-01-01\n1,10,3\n")
    with fs.open(f"{root}/week_1_sales.csv", "wt", encoding="utf-8") as handle:
        handle.write("Store,Product,2024-01-01,2024-01-08\n1,10,3,7\n")

    states = load_initial_states(f"memory://{root}/week_0_initial_state.csv")
    actuals = extract_new_actuals(f"memory://{root}", 1)

    assert states["1_10"].end_inventory == pytest.approx(5.0)
    assert states["1_10"].in_transit_w1 == pytest.approx(1.0)
    assert states["1_10"].in_transit_w2 == pytest.approx(2.0)
    assert actuals == {"1_10": 7.0}
