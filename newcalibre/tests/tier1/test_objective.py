"""Lock chapter 08 objective reducers to the authoritative settlement facts."""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionTiming,
    EmissionScope,
    InventoryPosition,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine import SettlementError, SettlementRequest, SettlementSnapshot, settle
from newcalibre.ledger import OrderRow, SettlementRecord
from newcalibre.ordering import (
    DEFAULT_OBJECTIVE,
    CostValue,
    DiagnosticWindow,
    ObjectiveError,
    diagnostic_cost,
    key_aligned_regret,
    settle_path_cost,
)

pytestmark = pytest.mark.tier1

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
PERIODS = tuple(pd.date_range("2026-01-01", periods=4, freq="D"))
TIMING = DecisionTiming(lead_time=1, review_period=1)
COST = CostStructure(underage=3.0, overage=2.0, holding=0.5, shortage=2.0)


def _session(
    *,
    series_keys: tuple[str, ...] = ("a", "b"),
    tenant: str = "tenant-a",
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=TIMING.protection_period,
        model_config={"backend": "fixture"},
        ordering_policy={"name": "rs", "coverage": 0.8},
        decision_series_keys=series_keys,
        cost_structure=COST,
        decision_timing=TIMING,
        stockout_rule=StockoutRule.LOST_SALES,
    )


def _snapshot(
    session: SessionIdentity,
    *,
    periods: tuple[pd.Timestamp, ...],
    positions: dict[str, InventoryPosition],
) -> SettlementSnapshot:
    return SettlementSnapshot(
        session=session,
        calendar=CALENDAR,
        periods=periods,
        frontier=None,
        latest_positions={},
        open_order_quantities={
            series_key: position.on_order for series_key, position in positions.items()
        },
        due_arrivals={},
        actuals_semantics=None,
    )


def _settled_records(
    *,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> tuple[SettlementRecord, ...]:
    session = _session()
    positions = {
        "a": InventoryPosition(5.0, 0.0, 0.0),
        "b": InventoryPosition(0.0, 0.0, 0.0),
    }
    periods = PERIODS[:2]
    actuals = {
        ("a", periods[0]): 3.0,
        ("b", periods[0]): 2.0,
        ("a", periods[1]): 3.0,
        ("b", periods[1]): 0.0,
    }
    return settle(
        SettlementRequest(
            session=session,
            snapshot=_snapshot(session, periods=periods, positions=positions),
            actuals=actuals,
            inventory_positions=positions,
            actuals_semantics=actuals_semantics,
        )
    ).records


def _window(
    *,
    series_key: str = "a",
    origin: pd.Timestamp = PERIODS[0],
    mode: EmissionScope = EmissionScope.PER_STEP,
    quantities: tuple[float, ...] = (4.0, 2.0),
    demands: tuple[float, ...] = (3.0, 5.0),
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> DiagnosticWindow:
    return DiagnosticWindow(
        series_key=series_key,
        origin=origin,
        mode=mode,
        quantities=quantities,
        demands=demands,
        costs=COST,
        actuals_semantics=actuals_semantics,
    )


def test_obj1_per_step_cost_is_hand_recomputable_and_order_independent() -> None:
    first = _window()
    second = _window(
        series_key="b",
        quantities=(1.0,),
        demands=(1.0,),
    )

    objective = diagnostic_cost((second, first), mode=EmissionScope.PER_STEP)
    replay = diagnostic_cost((first, second), mode=EmissionScope.PER_STEP)

    assert objective == replay
    assert list(objective.by_decision) == [("a", PERIODS[0]), ("b", PERIODS[0])]
    assert objective.by_decision[("a", PERIODS[0])].value == 11.0
    assert objective.by_decision[("b", PERIODS[0])].value == 0.0
    assert objective.total == CostValue(11.0, ActualsSemantics.DEMAND)


def test_obj1_window_sum_uses_one_decision_against_summed_demand() -> None:
    window = _window(
        mode=EmissionScope.WINDOW_SUM,
        quantities=(4.0,),
        demands=(1.0, 5.0),
    )

    objective = diagnostic_cost((window,), mode=EmissionScope.WINDOW_SUM)

    assert objective.by_decision[window.key].value == 6.0
    assert objective.total.value == 6.0


def test_obj1_rejects_mode_mismatch_mixing_and_multiple_window_sum_groups() -> None:
    per_step = _window()
    window_sum = _window(
        series_key="b",
        mode=EmissionScope.WINDOW_SUM,
        quantities=(4.0,),
    )
    second_window_sum = _window(
        series_key="c",
        mode=EmissionScope.WINDOW_SUM,
        quantities=(4.0,),
    )

    with pytest.raises(ObjectiveError, match="mismatches or mixes"):
        diagnostic_cost((window_sum,), mode=EmissionScope.PER_STEP)
    with pytest.raises(ObjectiveError, match="mismatches or mixes"):
        diagnostic_cost((per_step, window_sum), mode=EmissionScope.PER_STEP)
    with pytest.raises(ObjectiveError, match="exactly one window"):
        diagnostic_cost(
            (window_sum, second_window_sum),
            mode=EmissionScope.WINDOW_SUM,
        )


@pytest.mark.parametrize(
    ("mode", "quantities", "demands", "message"),
    [
        (EmissionScope.PER_STEP, (), (), "must not be empty"),
        (EmissionScope.PER_STEP, (1.0,), (1.0, 2.0), "must match demand"),
        (EmissionScope.WINDOW_SUM, (1.0, 2.0), (3.0,), "exactly one decision"),
    ],
)
def test_obj1_window_shape_is_explicit(
    mode: EmissionScope,
    quantities: tuple[float, ...],
    demands: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ObjectiveError, match=message):
        _window(mode=mode, quantities=quantities, demands=demands)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0])
def test_obj1_refuses_nonfinite_or_negative_consumed_values(value: float) -> None:
    with pytest.raises(ObjectiveError, match="non-negative and finite"):
        _window(quantities=(value,), demands=(0.0,))
    with pytest.raises(ObjectiveError, match="non-negative and finite"):
        _window(quantities=(0.0,), demands=(value,))


def test_obj1_surrogate_requires_an_explicit_matching_binding() -> None:
    window = _window(
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )

    with pytest.raises(ObjectiveError, match="semantics"):
        diagnostic_cost((window,), mode=EmissionScope.PER_STEP)

    objective = diagnostic_cost(
        (window,),
        mode=EmissionScope.PER_STEP,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    assert objective.total.actuals_semantics is ActualsSemantics.CENSORED_SALES_SURROGATE
    assert {cost.actuals_semantics for cost in objective.by_decision.values()} == {
        ActualsSemantics.CENSORED_SALES_SURROGATE
    }


def test_obj2_reduces_only_booked_settlement_components() -> None:
    records = _settled_records()

    objective = settle_path_cost(records)

    assert DEFAULT_OBJECTIVE is settle_path_cost
    assert objective.feasible
    assert objective.session == records[0].session
    assert [cost.value for cost in objective.by_origin.values()] == [5.0, 2.0]
    assert {key: cost.value for key, cost in objective.by_series.items()} == {
        "a": 3.0,
        "b": 4.0,
    }
    assert [partial.cost.value for partial in objective.partials] == [5.0, 7.0]
    assert objective.holding.value == 1.0
    assert objective.shortage.value == 6.0
    assert objective.total.value == 7.0
    assert objective.total.value == sum(
        components.holding.value + components.shortage.value
        for components in objective.by_decision.values()
    )


def test_obj2_every_derived_number_preserves_actuals_semantics() -> None:
    records = _settled_records(
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )

    with pytest.raises(ObjectiveError, match="explicit objective binding"):
        settle_path_cost(records)

    objective = settle_path_cost(
        records,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    derived = [
        objective.holding,
        objective.shortage,
        objective.total,
        *objective.by_origin.values(),
        *objective.by_series.values(),
        *(partial.cost for partial in objective.partials),
        *(
            component
            for costs in objective.by_decision.values()
            for component in (
                costs.holding,
                costs.shortage,
                costs.total,
            )
        ),
    ]
    assert derived
    assert {value.actuals_semantics for value in derived} == {
        ActualsSemantics.CENSORED_SALES_SURROGATE
    }


def test_obj2_refuses_mixed_or_dropped_semantics() -> None:
    records = list(_settled_records())
    records[-1] = replace(
        records[-1],
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )

    with pytest.raises(ObjectiveError, match="semantics"):
        settle_path_cost(records)
    with pytest.raises(TypeError, match="SettlementRecord"):
        settle_path_cost([records[0], object()])  # type: ignore[list-item]


def test_obj2_empty_candidate_is_an_explicit_labeled_infeasible_score() -> None:
    objective = settle_path_cost(())

    assert not objective.feasible
    assert objective.total == CostValue(math.inf, ActualsSemantics.DEMAND)
    assert objective.total.is_infeasible
    assert objective.infeasible_reason == "candidate emitted no settlement records"
    assert objective.by_decision == {}


def test_obj2_is_deterministic_and_snapshots_the_record_iterable() -> None:
    records = list(_settled_records())
    forward = settle_path_cost(records)
    reverse = settle_path_cost(reversed(records))
    records.clear()

    assert forward == reverse
    assert forward.feasible
    assert forward.total.value == 7.0


def test_obj2_refuses_duplicate_keys_and_multiple_sessions() -> None:
    records = list(_settled_records())
    with pytest.raises(ObjectiveError, match="unique decision keys"):
        settle_path_cost((records[0], records[0]))

    other_session = _session(tenant="tenant-b")
    foreign = replace(records[0], session=other_session)
    with pytest.raises(ObjectiveError, match="share one session"):
        settle_path_cost((records[0], foreign))


def test_obj2_propagates_infrastructure_errors_instead_of_scoring_infinity() -> None:
    record = _settled_records()[0]

    def failing_stream():
        yield record
        raise RuntimeError("infrastructure failed")

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        settle_path_cost(failing_stream())


def test_obj7_aligns_by_key_not_position_and_clips_negative_regret() -> None:
    a0 = ("a", PERIODS[0])
    b0 = ("b", PERIODS[0])
    a1 = ("a", PERIODS[1])
    c0 = ("c", PERIODS[0])
    candidate = {
        a1: CostValue(3.0, ActualsSemantics.DEMAND),
        b0: CostValue(1.0, ActualsSemantics.DEMAND),
        a0: CostValue(5.0, ActualsSemantics.DEMAND),
    }
    oracle = {
        c0: CostValue(4.0, ActualsSemantics.DEMAND),
        a0: CostValue(2.0, ActualsSemantics.DEMAND),
        b0: CostValue(2.0, ActualsSemantics.DEMAND),
    }

    objective = key_aligned_regret(candidate, oracle)
    candidate.clear()
    oracle.clear()

    assert list(objective.by_decision) == [a0, b0]
    assert objective.by_decision[a0].value == 3.0
    assert objective.by_decision[b0].value == 0.0
    assert objective.total == CostValue(3.0, ActualsSemantics.DEMAND)


def test_obj7_empty_alignment_is_exactly_zero() -> None:
    objective = key_aligned_regret(
        {("a", PERIODS[0]): CostValue(2.0, ActualsSemantics.DEMAND)},
        {("b", PERIODS[0]): CostValue(1.0, ActualsSemantics.DEMAND)},
    )

    assert objective.by_decision == {}
    assert objective.total == CostValue(0.0, ActualsSemantics.DEMAND)


def test_obj7_refuses_unlabeled_surrogate_and_infeasible_stream_values() -> None:
    key = ("a", PERIODS[0])
    with pytest.raises(ObjectiveError, match="semantics"):
        key_aligned_regret(
            {key: CostValue(2.0, ActualsSemantics.CENSORED_SALES_SURROGATE)},
            {key: CostValue(1.0, ActualsSemantics.CENSORED_SALES_SURROGATE)},
        )
    with pytest.raises(ObjectiveError, match="finite"):
        key_aligned_regret(
            {key: CostValue(math.inf, ActualsSemantics.DEMAND)},
            {key: CostValue(1.0, ActualsSemantics.DEMAND)},
        )


def test_sim1_to_sim5_conformance_flows_through_the_u5_settlement_interface() -> None:
    session = _session(series_keys=("a",))
    periods = PERIODS[:2]
    positions = {"a": InventoryPosition(0.0, 0.0, 0.0)}
    actuals = {("a", periods[0]): 2.0, ("a", periods[1]): 1.0}
    order = OrderRow(
        session=session,
        series_key="a",
        origin=periods[0],
        model_name="policy",
        quantity=4.0,
        arrival_period=periods[1],
    )
    request = SettlementRequest(
        session=session,
        snapshot=_snapshot(session, periods=periods, positions=positions),
        actuals=actuals,
        inventory_positions=positions,
        orders=(order,),
        actuals_semantics=ActualsSemantics.DEMAND,
    )

    first = settle(request)
    second = settle(request)
    objective = settle_path_cost(first.records)

    assert first == second
    assert positions == {"a": InventoryPosition(0.0, 0.0, 0.0)}
    assert actuals == {("a", periods[0]): 2.0, ("a", periods[1]): 1.0}
    assert [record.arrivals for record in first.records] == [0.0, 4.0]
    assert [record.transition.unmet_demand for record in first.records] == [2.0, 0.0]
    assert [record.transition.closing_on_hand for record in first.records] == [0.0, 3.0]
    assert [record.inventory_position.on_order for record in first.records] == [4.0, 0.0]
    assert objective.holding.value == 1.5
    assert objective.shortage.value == 4.0
    assert objective.total.value == 5.5


def test_sim6_refuses_a_window_missing_resolved_demand_before_settlement() -> None:
    session = _session(series_keys=("a",))
    periods = PERIODS[:2]
    positions = {"a": InventoryPosition(0.0, 0.0, 0.0)}

    with pytest.raises(SettlementError, match="exactly match the window"):
        SettlementRequest(
            session=session,
            snapshot=_snapshot(session, periods=periods, positions=positions),
            actuals={("a", periods[0]): 0.0},
            inventory_positions=positions,
            actuals_semantics=ActualsSemantics.DEMAND,
        )
