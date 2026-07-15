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
    CandidateInfeasible,
    CostComponents,
    CostValue,
    DiagnosticWindow,
    ObjectiveError,
    SettlementObjective,
    diagnostic_cost,
    evaluate_settlement_candidate,
    key_aligned_regret,
    realized_cost,
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


def _single_series_records(
    demand: float,
    *,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> tuple[SettlementRecord, ...]:
    session = _session(series_keys=("a",))
    positions = {"a": InventoryPosition(0.0, 0.0, 0.0)}
    periods = PERIODS[:1]
    return settle(
        SettlementRequest(
            session=session,
            snapshot=_snapshot(session, periods=periods, positions=positions),
            actuals={("a", periods[0]): demand},
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

    assert DEFAULT_OBJECTIVE is realized_cost
    assert DEFAULT_OBJECTIVE(records) == objective.total
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
    assert objective.total.value == sum(cost.value for cost in objective.by_decision.values())
    assert objective.by_decision == {
        key: components.total for key, components in objective.components_by_decision.items()
    }


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
        *objective.by_decision.values(),
        *(
            component
            for costs in objective.components_by_decision.values()
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
    assert objective.components_by_decision == {}


def test_obj2_tuning_default_returns_a_labeled_scalar_for_surrogate_actuals() -> None:
    records = _single_series_records(
        1.0,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )

    objective = DEFAULT_OBJECTIVE(
        records,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )

    assert objective == CostValue(2.0, ActualsSemantics.CENSORED_SALES_SURROGATE)


def test_obj3_preserves_cumulative_trajectory_across_float_reassociation() -> None:
    semantics = ActualsSemantics.DEMAND
    components_by_decision = {
        ("a", PERIODS[0]): CostComponents(
            holding=CostValue(0.0, semantics),
            shortage=CostValue(1.0, semantics),
        ),
        ("b", PERIODS[0]): CostComponents(
            holding=CostValue(2.0, semantics),
            shortage=CostValue(1e16, semantics),
        ),
        ("a", PERIODS[1]): CostComponents(
            holding=CostValue(1e-16, semantics),
            shortage=CostValue(0.0, semantics),
        ),
    }

    objective = SettlementObjective(
        session=_session(),
        actuals_semantics=semantics,
        components_by_decision=components_by_decision,
    )
    component_grouped_total = objective.holding.value + objective.shortage.value

    assert [cost.value for cost in objective.by_origin.values()] == [
        1.0000000000000004e16,
        1e-16,
    ]
    assert [partial.cost.value for partial in objective.partials] == [
        1.0000000000000004e16,
        1.0000000000000004e16,
    ]
    assert component_grouped_total == 1.0000000000000002e16
    assert component_grouped_total < objective.partials[0].cost.value
    assert objective.total is objective.partials[-1].cost
    assert objective.total.value != component_grouped_total


def test_obj2_one_booked_cost_quantum_changes_the_objective_once() -> None:
    lower_records = _single_series_records(1.0)
    higher_records = _single_series_records(1.5)

    lower = settle_path_cost(lower_records)
    higher = settle_path_cost(higher_records)
    booked_delta = higher_records[0].shortage.amount - lower_records[0].shortage.amount

    assert booked_delta == 1.0
    assert higher.total.value - lower.total.value == booked_delta


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


def test_obj2_preserves_iterator_type_error_identity() -> None:
    failure = TypeError("settlement source failed")

    def failing_stream():
        yield _settled_records()[0]
        raise failure

    with pytest.raises(TypeError) as captured:
        settle_path_cost(failing_stream())

    assert captured.value is failure


def test_obj5_scores_only_typed_candidate_infeasibility_as_infinity() -> None:
    failure = CandidateInfeasible("candidate parameters are degenerate")

    def infeasible_candidate() -> tuple[SettlementRecord, ...]:
        raise failure

    objective = evaluate_settlement_candidate(infeasible_candidate)

    assert not objective.feasible
    assert objective.total == CostValue(math.inf, ActualsSemantics.DEMAND)
    assert objective.infeasible_reason == str(failure)


@pytest.mark.parametrize("failure", [SettlementError("engine failed"), RuntimeError("I/O failed")])
def test_obj5_candidate_evaluator_preserves_engine_and_infrastructure_error_identity(
    failure: Exception,
) -> None:
    def failing_candidate() -> tuple[SettlementRecord, ...]:
        raise failure

    with pytest.raises(type(failure)) as captured:
        evaluate_settlement_candidate(failing_candidate)

    assert captured.value is failure


@pytest.mark.parametrize(
    "actuals_semantics",
    [ActualsSemantics.DEMAND, ActualsSemantics.CENSORED_SALES_SURROGATE],
)
def test_obj7_composes_directly_with_settlement_totals_and_preserves_semantics(
    actuals_semantics: ActualsSemantics,
) -> None:
    candidate = settle_path_cost(
        _single_series_records(2.0, actuals_semantics=actuals_semantics),
        actuals_semantics=actuals_semantics,
    )
    oracle = settle_path_cost(
        _single_series_records(1.0, actuals_semantics=actuals_semantics),
        actuals_semantics=actuals_semantics,
    )

    regret = key_aligned_regret(
        candidate.by_decision,
        oracle.by_decision,
        actuals_semantics=actuals_semantics,
    )

    assert candidate.by_decision == {
        key: components.total for key, components in candidate.components_by_decision.items()
    }
    assert list(regret.by_decision.values()) == [CostValue(2.0, actuals_semantics)]
    assert regret.total == CostValue(2.0, actuals_semantics)


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
