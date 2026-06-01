"""Tests for the generic discrete-period inventory simulator."""

from __future__ import annotations

import pytest

from calibre.ordering.simulation import (
    LinearCostModel,
    LostSalesRule,
    PeriodResult,
    ProductState,
    Simulator,
    make_pipeline,
)


def _state(uid: str, end_inventory: float, in_transit: list[float], depth: int) -> ProductState:
    return ProductState(
        unique_id=uid,
        end_inventory=end_inventory,
        pipeline=make_pipeline(in_transit, depth),
    )


def _vn2_like_sim(
    end_inventory: float = 0.0,
    in_transit: list[float] | None = None,
    depth: int = 2,
) -> Simulator:
    state = _state("X", end_inventory, in_transit or [], depth)
    return Simulator(
        states={"X": state},
        rule=LostSalesRule(lead_time_depth=depth),
        cost_model=LinearCostModel(rates={"holding": 0.2, "shortage": 1.0}),
    )


class TestPipelineDepth:
    def test_depth_three_order_arrives_after_three_periods(self) -> None:
        """With lead_time_depth=3, an order placed period 1 arrives period 4."""
        sim = _vn2_like_sim(end_inventory=0.0, depth=3)

        sim.step(1, orders={"X": 50.0}, actual_demand={"X": 0.0})
        for period in (2, 3):
            result = sim.step(period, orders={}, actual_demand={"X": 0.0})[0]
            assert result.arrivals == pytest.approx(0.0)

        result = sim.step(4, orders={}, actual_demand={"X": 0.0})[0]
        assert result.arrivals == pytest.approx(50.0)
        assert result.start_inventory == pytest.approx(50.0)

    def test_depth_one_order_arrives_next_period(self) -> None:
        """With lead_time_depth=1, an order placed period 1 arrives period 2."""
        sim = _vn2_like_sim(end_inventory=0.0, depth=1)

        sim.step(1, orders={"X": 10.0}, actual_demand={"X": 0.0})
        result = sim.step(2, orders={}, actual_demand={"X": 0.0})[0]
        assert result.arrivals == pytest.approx(10.0)

    def test_depth_zero_rejects_orders(self) -> None:
        sim = _vn2_like_sim(end_inventory=5.0, depth=0)

        with pytest.raises(ValueError, match="lead_time_depth=0"):
            sim.step(1, orders={"X": 1.0}, actual_demand={"X": 0.0})

    def test_state_pipeline_mismatches_rule_depth_raises(self) -> None:
        """Rule must reject a state whose pipeline depth differs from its own."""
        state = _state("X", 0.0, in_transit=[], depth=2)
        rule = LostSalesRule(lead_time_depth=3)
        with pytest.raises(ValueError, match="depth"):
            rule.transition(state, period=1, demand=0.0, order=0.0)


class TestInventoryAccountingInvariant:
    def test_start_minus_sales_equals_end_across_multiple_periods(self) -> None:
        sim = _vn2_like_sim(end_inventory=15.0, in_transit=[5.0, 0.0], depth=2)
        for period, demand in enumerate([8.0, 30.0, 0.0], start=1):
            result = sim.step(period, orders={}, actual_demand={"X": demand})[0]
            assert result.start_inventory - result.sales == pytest.approx(result.end_inventory)


class TestCostBreakdown:
    def test_linear_cost_model_with_custom_components(self) -> None:
        """LinearCostModel can express arbitrary attribute-based linear costs."""
        sim = Simulator(
            states={"X": _state("X", end_inventory=5.0, in_transit=[], depth=1)},
            rule=LostSalesRule(lead_time_depth=1),
            cost_model=LinearCostModel(
                rates={"holding": 0.5, "shortage": 2.0, "handling": 0.1},
                attributes={
                    "holding": "end_inventory",
                    "shortage": "missed_sales",
                    "handling": "sales",
                },
            ),
        )

        sim.step(1, orders={"X": 0.0}, actual_demand={"X": 8.0})
        breakdown = sim.cost_breakdown()

        # start=5, demand=8 → sales=5, missed=3, end=0
        assert breakdown["holding"] == pytest.approx(0.0)
        assert breakdown["shortage"] == pytest.approx(3.0 * 2.0)
        assert breakdown["handling"] == pytest.approx(5.0 * 0.1)
        assert sim.total_cost() == pytest.approx(6.5)

    def test_unknown_rate_component_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown components"):
            LinearCostModel(rates={"mystery": 1.0})

    def test_pluggable_cost_model_not_required_to_be_linear(self) -> None:
        """A user-defined CostModel can return any breakdown dict."""

        class StepCostModel:
            def cost(self, state: ProductState, result: PeriodResult) -> dict[str, float]:
                del state
                penalty = 100.0 if result.missed_sales > 0 else 0.0
                return {"penalty": penalty}

        sim = Simulator(
            states={"X": _state("X", end_inventory=2.0, in_transit=[], depth=1)},
            rule=LostSalesRule(lead_time_depth=1),
            cost_model=StepCostModel(),
        )
        sim.step(1, orders={}, actual_demand={"X": 5.0})
        sim.step(2, orders={}, actual_demand={"X": 0.0})

        assert sim.cost_breakdown() == {"penalty": 100.0}


class TestSimulatorIO:
    def test_deep_copies_states(self) -> None:
        """The simulator must not mutate caller-owned states."""
        original = _state("X", end_inventory=10.0, in_transit=[1.0, 2.0], depth=2)
        sim = Simulator(
            states={"X": original},
            rule=LostSalesRule(lead_time_depth=2),
            cost_model=LinearCostModel(rates={"holding": 0.2, "shortage": 1.0}),
        )
        sim.step(1, orders={"X": 5.0}, actual_demand={"X": 0.0})
        assert original.end_inventory == pytest.approx(10.0)
        assert list(original.pipeline) == [1.0, 2.0]

    def test_to_dataframe_includes_named_cost_columns(self) -> None:
        sim = _vn2_like_sim(end_inventory=5.0, depth=1)
        sim.step(1, orders={}, actual_demand={"X": 2.0})
        df = sim.to_dataframe()

        expected = {
            "unique_id",
            "period",
            "start_inventory",
            "arrivals",
            "demand",
            "sales",
            "missed_sales",
            "end_inventory",
            "order",
            "holding_cost",
            "shortage_cost",
        }
        assert expected.issubset(set(df.columns))
        assert len(df) == 1


class TestProductStateHelpers:
    def test_inventory_position_sums_on_hand_and_in_transit(self) -> None:
        state = _state("X", end_inventory=10.0, in_transit=[3.0, 2.0], depth=2)
        assert state.inventory_position == pytest.approx(15.0)

    def test_make_pipeline_pads_with_zeros(self) -> None:
        pipeline = make_pipeline([7.0], lead_time_depth=3)
        assert list(pipeline) == [7.0, 0.0, 0.0]
        assert pipeline.maxlen == 3

    def test_make_pipeline_rejects_overflow(self) -> None:
        with pytest.raises(ValueError, match="exceeds lead_time_depth"):
            make_pipeline([1.0, 2.0, 3.0], lead_time_depth=2)
