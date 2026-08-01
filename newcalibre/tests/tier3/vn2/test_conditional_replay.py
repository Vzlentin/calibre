"""Replay one promoted VN2 decision stream through independent accounting."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import pandas as pd
import pytest

from newcalibre.domain import (
    ActualsSemantics,
    AppliedBinding,
    Calendar,
    DecisionEvidence,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeDescriptor,
    GuaranteeType,
    InventoryPosition,
    ScoredSeries,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine import (
    InMemoryIndexedRunStore,
    SettlementRequest,
    SettlementResult,
    settle,
)
from newcalibre.ledger import OrderRow, SettlementRecord
from newcalibre.oracle import CaptureBundle
from newcalibre.protocols.vn2 import VN2Dataset
from oracle.reference import (
    ReferenceOrder,
    ReferenceRow,
    ReferenceSeries,
    ReferenceTrajectory,
    calculate_reference_trajectory,
)

pytestmark = pytest.mark.tier3


@dataclass(frozen=True, slots=True)
class CapturedDecision:
    """Freeze one shared oracle order before either accounting path consumes it."""

    series_key: str
    origin_index: int
    quantity: float


@dataclass(frozen=True, slots=True)
class CapturedInitialState:
    """Freeze initial stock and both pipeline weeks as shared scalar inputs."""

    series_key: str
    on_hand: float
    pipeline: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReferenceInputs:
    """Freeze the exact inputs consumed only by the independent accounting path."""

    periods: tuple[str, ...]
    series: tuple[ReferenceSeries, ...]
    demand: Mapping[tuple[str, str], float]
    orders: tuple[ReferenceOrder, ...]
    lead_time: int
    initial_arrivals: Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class ReplayInputs:
    """Own one immutable replay specification plus its frozen expectation."""

    session: SessionIdentity
    calendar: Calendar
    actuals_semantics: ActualsSemantics
    stockout_rule: StockoutRule
    periods: tuple[pd.Timestamp, ...]
    decision_origins: tuple[pd.Timestamp, ...]
    drain_periods: int
    initial_positions: Mapping[str, InventoryPosition]
    initial_arrivals: Mapping[tuple[str, pd.Timestamp], float]
    actuals: Mapping[tuple[str, pd.Timestamp], float]
    orders: tuple[OrderRow, ...]
    reference: ReferenceInputs
    expectation: ReferenceTrajectory


@dataclass(frozen=True, slots=True)
class CanonicalReplay:
    """Bind the complete successor result to immutable inputs and fingerprints."""

    inputs: ReplayInputs
    result: SettlementResult
    replay_spec_fingerprint: str
    expectation_fingerprint: str


@pytest.fixture(scope="session")
def canonical_replay(
    validated_promoted_capture: CaptureBundle,
    exact_vn2_dataset: VN2Dataset,
) -> CanonicalReplay:
    """Freeze the expectation first, then invoke the successor settlement once."""
    return _build_replay(validated_promoted_capture, exact_vn2_dataset)


@pytest.mark.oracle_gate("vn2-conditional-replay")
def test_promoted_orders_match_independent_conditional_replay(
    canonical_replay: CanonicalReplay,
) -> None:
    _assert_conditional_replay(canonical_replay, canonical_replay.result)


@pytest.mark.oracle_witness("vn2-conditional-replay")
def test_conditional_replay_rejects_one_successor_order_unit(
    canonical_replay: CanonicalReplay,
) -> None:
    _assert_one_successor_order_unit_is_rejected(canonical_replay)


def _assert_one_successor_order_unit_is_rejected(replay: CanonicalReplay) -> None:
    """Keep shared-spec validation outside the witness' expected-failure scope."""

    _assert_fingerprint_integrity(replay)
    inputs = replay.inputs
    orders = list(inputs.orders)
    last_origin = max(order.origin for order in orders)
    target_index = min(
        (index for index, order in enumerate(orders) if order.origin == last_origin),
        key=lambda index: orders[index].series_key.encode(),
    )
    orders[target_index] = replace(
        orders[target_index],
        quantity=orders[target_index].quantity + 1.0,
    )

    drifted = _settle_orders(inputs, tuple(orders))

    with pytest.raises(AssertionError, match="conditional replay"):
        _assert_settlement_result(inputs, drifted)


def test_replay_fingerprints_reject_shared_or_expectation_replacement(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    series_key = min(inputs.initial_positions, key=str.encode)
    positions = dict(inputs.initial_positions)
    positions[series_key] = replace(
        positions[series_key],
        on_hand=positions[series_key].on_hand + 1.0,
    )
    _assert_replay_spec_rejects(
        canonical_replay,
        replace(inputs, initial_positions=MappingProxyType(positions)),
    )

    replaced_expectation = replace(
        inputs.expectation,
        total_cost=inputs.expectation.total_cost + 1.0,
    )
    replaced_replay = replace(
        canonical_replay,
        inputs=replace(inputs, expectation=replaced_expectation),
    )
    with pytest.raises(AssertionError, match="expectation fingerprint"):
        _assert_conditional_replay(replaced_replay, canonical_replay.result)

    demand = cast(dict[tuple[str, str], float], inputs.reference.demand)
    with pytest.raises(TypeError):
        demand[next(iter(demand))] = 0.0


def test_replay_spec_fingerprint_binds_session_preimage(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    replacement_session = _different_session(inputs)
    assert replacement_session.value != inputs.session.value
    assert replacement_session.to_bytes() != inputs.session.to_bytes()

    _assert_replay_spec_rejects(
        canonical_replay,
        replace(inputs, session=replacement_session),
    )


def test_replay_spec_fingerprint_binds_periods(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    changed_periods = (inputs.periods[1], inputs.periods[0], *inputs.periods[2:])

    _assert_replay_spec_rejects(
        canonical_replay,
        replace(inputs, periods=changed_periods),
    )


def test_replay_spec_fingerprint_binds_actuals_semantics_and_config(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    changed_semantics = (
        ActualsSemantics.DEMAND
        if inputs.actuals_semantics is not ActualsSemantics.DEMAND
        else ActualsSemantics.CENSORED_SALES_SURROGATE
    )
    changed_rule = (
        StockoutRule.BACKORDER
        if inputs.stockout_rule is not StockoutRule.BACKORDER
        else StockoutRule.LOST_SALES
    )
    changed_configurations = (
        replace(inputs, actuals_semantics=changed_semantics),
        replace(inputs, calendar=Calendar("D", phase=inputs.periods[0])),
        replace(inputs, stockout_rule=changed_rule),
        replace(
            inputs,
            decision_origins=(
                inputs.decision_origins[1],
                inputs.decision_origins[0],
                *inputs.decision_origins[2:],
            ),
        ),
        replace(inputs, drain_periods=inputs.drain_periods + 1),
    )

    for changed in changed_configurations:
        _assert_replay_spec_rejects(canonical_replay, changed)


def test_replay_spec_fingerprint_binds_independent_reference_inputs(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    reference = inputs.reference
    demand_key = next(iter(reference.demand))
    changed_demand = dict(reference.demand)
    changed_demand[demand_key] += 1.0
    arrival_key = next(iter(reference.initial_arrivals))
    changed_arrivals = dict(reference.initial_arrivals)
    changed_arrivals[arrival_key] += 1.0
    changed_references = (
        replace(
            reference,
            periods=(reference.periods[1], reference.periods[0], *reference.periods[2:]),
        ),
        replace(
            reference,
            series=(
                replace(
                    reference.series[0],
                    initial_on_hand=reference.series[0].initial_on_hand + 1.0,
                ),
                *reference.series[1:],
            ),
        ),
        replace(reference, demand=MappingProxyType(changed_demand)),
        replace(
            reference,
            orders=(
                replace(
                    reference.orders[0],
                    quantity=reference.orders[0].quantity + 1.0,
                ),
                *reference.orders[1:],
            ),
        ),
        replace(reference, lead_time=reference.lead_time + 1),
        replace(reference, initial_arrivals=MappingProxyType(changed_arrivals)),
    )

    for changed_reference in changed_references:
        _assert_replay_spec_rejects(
            canonical_replay,
            replace(inputs, reference=changed_reference),
        )


def test_replay_spec_fingerprint_binds_full_order_identity_and_evidence(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    order = inputs.orders[0]
    evidence = _decision_evidence()
    changed_orders = (
        replace(order, session=_different_session(inputs)),
        replace(order, series_key=f"{order.series_key}-changed"),
        replace(order, origin=order.origin + pd.Timedelta(days=7)),
        replace(order, model_name=f"{order.model_name}-changed"),
        replace(order, quantity=order.quantity + 1.0),
        replace(order, arrival_period=order.arrival_period + pd.Timedelta(days=7)),
        replace(order, evidence=evidence),
    )
    baseline_payload = _order_row_payload(order)
    assert all(_order_row_payload(changed) != baseline_payload for changed in changed_orders)

    evidence_order = replace(order, evidence=evidence)
    evidence_payload = _order_row_payload(evidence_order)
    assert evidence.reorder_point is not None
    changed_evidence = (
        replace(evidence, raw_target=evidence.raw_target + 1.0),
        replace(evidence, target=evidence.target + 1.0),
        replace(evidence, source_columns=(*evidence.source_columns, "extra")),
        replace(
            evidence,
            source_descriptor=replace(
                evidence.source_descriptor,
                level=evidence.source_descriptor.level + 0.1,
            ),
        ),
        replace(
            evidence,
            effective_descriptor=replace(
                evidence.effective_descriptor,
                level=evidence.effective_descriptor.level + 0.1,
            ),
        ),
        replace(
            evidence,
            bindings=(replace(evidence.bindings[0], value=evidence.bindings[0].value + 1.0),),
        ),
        replace(evidence, reorder_point=evidence.reorder_point + 1.0),
    )
    assert all(
        _order_row_payload(replace(evidence_order, evidence=changed)) != evidence_payload
        for changed in changed_evidence
    )

    _assert_replay_spec_rejects(
        canonical_replay,
        replace(inputs, orders=(changed_orders[-1], *inputs.orders[1:])),
    )


def test_witness_rejects_wrong_shared_mutation_before_expected_failure(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    first_order = inputs.orders[0]
    wrong_shared_orders = (
        replace(first_order, quantity=first_order.quantity + 1.0),
        *inputs.orders[1:],
    )
    wrong_shared_replay = replace(
        canonical_replay,
        inputs=replace(inputs, orders=wrong_shared_orders),
    )

    with pytest.raises(AssertionError, match="replay-spec fingerprint"):
        _assert_one_successor_order_unit_is_rejected(wrong_shared_replay)


def test_replay_rejects_per_period_on_order_drift(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    result = canonical_replay.result
    target_index = next(
        index
        for index, record in enumerate(result.records)
        if record.period != inputs.periods[-1] and record.inventory_position.on_order > 0.0
    )
    target = result.records[target_index]
    records = list(result.records)
    records[target_index] = replace(
        target,
        inventory_position=replace(
            target.inventory_position,
            on_order=target.inventory_position.on_order + 1.0,
        ),
    )

    with pytest.raises(AssertionError, match="inventory on-order"):
        _assert_settlement_result(inputs, replace(result, records=tuple(records)))


def test_replay_rejects_terminal_returned_position_drift(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    result = canonical_replay.result
    series_key = min(result.inventory_positions, key=str.encode)
    positions = dict(result.inventory_positions)
    terminal = positions[series_key]
    positions[series_key] = replace(terminal, on_hand=terminal.on_hand + 1.0)

    with pytest.raises(AssertionError, match="terminal returned position"):
        _assert_settlement_result(
            inputs,
            replace(result, inventory_positions=positions),
        )


def test_replay_requires_terminal_open_orders_to_be_drained(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    result = canonical_replay.result
    series_key = min(result.inventory_positions, key=str.encode)
    target_index = next(
        index
        for index, record in enumerate(result.records)
        if record.series_key == series_key and record.period == inputs.periods[-1]
    )
    target = result.records[target_index]
    drifted_position = replace(target.inventory_position, on_order=1.0)
    records = list(result.records)
    records[target_index] = replace(target, inventory_position=drifted_position)
    positions = dict(result.inventory_positions)
    positions[series_key] = drifted_position

    with pytest.raises(AssertionError, match="terminal open orders were not drained"):
        _assert_settlement_result(
            inputs,
            replace(
                result,
                records=tuple(records),
                inventory_positions=positions,
            ),
        )


def test_replay_rejects_extra_duplicate_zero_cost_record(
    canonical_replay: CanonicalReplay,
) -> None:
    inputs = canonical_replay.inputs
    result = canonical_replay.result
    duplicate = next(
        record
        for record in result.records
        if record.period != inputs.periods[-1] and record.realized_cost == 0.0
    )
    records = (*result.records, duplicate)
    unique_keys = {(record.series_key, record.period) for record in records}
    assert len(records) == 4_793
    assert len(unique_keys) == 4_792

    with pytest.raises(AssertionError, match="raw settlement row count"):
        _assert_settlement_result(inputs, replace(result, records=records))


def _assert_replay_spec_rejects(
    replay: CanonicalReplay,
    changed_inputs: ReplayInputs,
) -> None:
    with pytest.raises(AssertionError, match="replay-spec fingerprint"):
        _assert_fingerprint_integrity(replace(replay, inputs=changed_inputs))


def _different_session(inputs: ReplayInputs) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="conditional-replay-fingerprint-mutation",
        series_keys=("mutation-series",),
        calendar=inputs.calendar,
        horizon=1,
        model_config={"name": "mutation-model"},
    )


def _decision_evidence() -> DecisionEvidence:
    descriptor = GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.NONE,
            currency=None,
            declared_slack=None,
        ),
        level=0.5,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )
    return DecisionEvidence(
        raw_target=11.0,
        target=10.0,
        source_columns=("q0.5",),
        source_descriptor=descriptor,
        effective_descriptor=descriptor,
        bindings=(AppliedBinding(name="target-cap", value=10.0),),
        reorder_point=4.0,
    )


def _build_replay(bundle: CaptureBundle, dataset: VN2Dataset) -> CanonicalReplay:
    config = dataset.config
    first = dataset.round_input(1)
    store_column, product_column = config.columns.series_keys
    series_keys = tuple(
        f"{int(store)}_{int(product)}"
        for store, product in zip(
            first.sales[store_column],
            first.sales[product_column],
            strict=True,
        )
    )
    assert len(series_keys) == config.series_count == 599
    assert len(set(series_keys)) == len(series_keys)

    periods = config.realized_periods
    reference_periods = tuple(period.strftime("%Y-%m-%d") for period in periods)
    actuals = {
        (f"{int(row[store_column])}_{int(row[product_column])}", weekly.period): float(row["sales"])
        for week_number in range(1, len(periods) + 1)
        for weekly in (dataset.weekly_actuals(week_number),)
        for row in weekly.sales.to_dict(orient="records")
    }
    assert len(actuals) == 599 * 8 == 4_792
    reference_demand = {
        (series_key, period_name): actuals[(series_key, period)]
        for period, period_name in zip(periods, reference_periods, strict=True)
        for series_key in series_keys
    }

    captured_initial_states: list[CapturedInitialState] = []
    reference_arrivals: dict[tuple[str, str], float] = {}
    reference_series: list[ReferenceSeries] = []
    for row in first.initial_state.to_dict(orient="records"):
        series_key = f"{int(row[store_column])}_{int(row[product_column])}"
        pipeline = tuple(float(row[column]) for column in config.columns.initial_pipeline)
        initial_state = CapturedInitialState(
            series_key=series_key,
            on_hand=float(row[config.columns.initial_on_hand]),
            pipeline=pipeline,
        )
        captured_initial_states.append(initial_state)
        reference_series.append(
            ReferenceSeries(
                series_key=series_key,
                initial_on_hand=initial_state.on_hand,
                holding_rate=config.holding_rate,
                shortage_rate=config.shortage_rate,
            )
        )
        for period_name, quantity in zip(
            reference_periods[: config.timing.lead_time],
            pipeline,
            strict=True,
        ):
            reference_arrivals[(series_key, period_name)] = quantity
    assert {item.series_key for item in captured_initial_states} == set(series_keys)
    assert len(reference_arrivals) == 599 * 2

    captured_decisions: list[CapturedDecision] = []
    for origin_index, expected_origin in enumerate(config.decision_origins):
        path = bundle.root / "orders" / f"round-{origin_index + 1}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["round_num"] == origin_index + 1
        assert pd.Timestamp(payload["origin"]) == expected_origin
        captured_orders = payload["orders"]
        assert isinstance(captured_orders, dict)
        assert set(captured_orders) == set(series_keys)
        captured_decisions.extend(
            CapturedDecision(
                series_key=series_key,
                origin_index=origin_index,
                quantity=float(quantity),
            )
            for series_key, quantity in captured_orders.items()
        )
    frozen_decisions = tuple(captured_decisions)
    assert len(frozen_decisions) == 599 * 6 == 3_594

    # This expectation is deliberately complete before SessionIdentity,
    # OrderRow, InventoryPosition, the sink, or production settlement exists.
    reference = ReferenceInputs(
        periods=reference_periods,
        series=tuple(reference_series),
        demand=MappingProxyType(reference_demand),
        orders=tuple(
            ReferenceOrder(item.series_key, item.origin_index, item.quantity)
            for item in frozen_decisions
        ),
        lead_time=config.timing.lead_time,
        initial_arrivals=MappingProxyType(reference_arrivals),
    )
    expectation = calculate_reference_trajectory(
        periods=reference.periods,
        series=reference.series,
        demand=reference.demand,
        orders=reference.orders,
        lead_time=reference.lead_time,
        initial_arrivals=reference.initial_arrivals,
    )
    expectation_fingerprint = _expectation_fingerprint(expectation)

    initial_positions = {
        item.series_key: InventoryPosition(
            on_hand=item.on_hand,
            on_order=math.fsum(item.pipeline),
            backorders=0.0,
        )
        for item in captured_initial_states
    }
    initial_arrivals = {
        (item.series_key, period): quantity
        for item in captured_initial_states
        for period, quantity in zip(
            periods[: config.timing.lead_time],
            item.pipeline,
            strict=True,
        )
    }

    session = SessionIdentity.derive(
        tenant=config.dataset,
        series_keys=series_keys,
        calendar=config.calendar,
        horizon=config.task_horizon,
        model_config=config.model_config,
        conformal_config=config.conformal_config,
        ordering_policy=config.ordering_policy,
        decision_series_keys=series_keys,
        cost_structure=config.cost_structure,
        decision_timing=config.timing,
        stockout_rule=config.stockout_rule,
    )
    model_name = config.model_config["model_name"]
    assert isinstance(model_name, str)
    orders = tuple(
        OrderRow(
            session=session,
            series_key=item.series_key,
            origin=config.decision_origins[item.origin_index],
            model_name=model_name,
            quantity=item.quantity,
            arrival_period=config.calendar.advance(
                config.decision_origins[item.origin_index],
                config.timing.lead_time,
            ),
        )
        for item in frozen_decisions
    )

    inputs = ReplayInputs(
        session=session,
        calendar=config.calendar,
        actuals_semantics=config.actuals_semantics,
        stockout_rule=config.stockout_rule,
        periods=periods,
        decision_origins=config.decision_origins,
        drain_periods=config.drain_periods,
        initial_positions=MappingProxyType(initial_positions),
        initial_arrivals=MappingProxyType(initial_arrivals),
        actuals=MappingProxyType(actuals),
        orders=orders,
        reference=reference,
        expectation=expectation,
    )
    result = _settle_orders(inputs, inputs.orders)
    return CanonicalReplay(
        inputs=inputs,
        result=result,
        replay_spec_fingerprint=_replay_spec_fingerprint(inputs),
        expectation_fingerprint=expectation_fingerprint,
    )


def _settle_orders(inputs: ReplayInputs, orders: Sequence[OrderRow]) -> SettlementResult:
    sink = InMemoryIndexedRunStore(
        session=inputs.session,
        calendar=inputs.calendar,
        actuals_semantics=inputs.actuals_semantics,
        initial_arrivals=inputs.initial_arrivals,
    )
    request = SettlementRequest(
        session=inputs.session,
        snapshot=sink.settlement_snapshot(inputs.periods),
        actuals=inputs.actuals,
        inventory_positions=inputs.initial_positions,
        orders=orders,
        actuals_semantics=inputs.actuals_semantics,
    )
    return settle(request)


def _assert_conditional_replay(
    replay: CanonicalReplay,
    result: SettlementResult,
) -> None:
    _assert_fingerprint_integrity(replay)
    _assert_settlement_result(replay.inputs, result)


def _assert_fingerprint_integrity(replay: CanonicalReplay) -> None:
    assert _replay_spec_fingerprint(replay.inputs) == replay.replay_spec_fingerprint, (
        "conditional replay replay-spec fingerprint changed"
    )
    assert _expectation_fingerprint(replay.inputs.expectation) == replay.expectation_fingerprint, (
        "conditional replay expectation fingerprint changed"
    )


def _assert_settlement_result(
    inputs: ReplayInputs,
    result: SettlementResult,
) -> None:
    _assert_terminal_inventory_positions(inputs, result)
    _assert_settlement_records(inputs, result.records)


def _assert_terminal_inventory_positions(
    inputs: ReplayInputs,
    result: SettlementResult,
) -> None:
    assert inputs.drain_periods >= inputs.reference.lead_time
    assert inputs.periods[: -inputs.drain_periods] == inputs.decision_origins
    terminal_period = inputs.periods[-1]
    last_records = {
        record.series_key: record for record in result.records if record.period == terminal_period
    }
    assert set(last_records) == set(result.inventory_positions) == set(inputs.initial_positions)

    for series_key in sorted(last_records, key=str.encode):
        final_position = last_records[series_key].inventory_position
        assert final_position.on_order == 0.0, (
            f"conditional replay terminal open orders were not drained for {series_key!r}"
        )
        assert result.inventory_positions[series_key] == final_position, (
            f"conditional replay terminal returned position differs from the last row "
            f"for {series_key!r}"
        )


def _assert_settlement_records(
    inputs: ReplayInputs,
    records: Sequence[SettlementRecord],
) -> None:
    expected_rows = inputs.expectation.rows
    expected_count = 599 * 8
    assert len(expected_rows) == expected_count == 4_792
    assert len(records) == expected_count, (
        "conditional replay raw settlement row count differs: "
        f"actual={len(records)} expected={expected_count}"
    )

    expected: dict[tuple[str, str], ReferenceRow] = {}
    expected_keys: list[tuple[str, str]] = []
    reference_series = {item.series_key: item for item in inputs.reference.series}
    expected_costs = {period: [] for period in inputs.reference.periods}
    expected_terminal_costs: list[float] = []
    for row in expected_rows:
        key = (row.series_key, row.period)
        expected_keys.append(key)
        expected[key] = row
        expected_costs[row.period].append(row.total_cost)
        expected_terminal_costs.append(row.total_cost)

    actual: dict[tuple[str, str], SettlementRecord] = {}
    actual_keys: list[tuple[str, str]] = []
    actual_costs = {period: [] for period in inputs.reference.periods}
    actual_terminal_costs: list[float] = []
    for record in records:
        period = record.period.strftime("%Y-%m-%d")
        key = (record.series_key, period)
        assert key not in actual, f"conditional replay duplicate settlement row key: {key!r}"
        actual_keys.append(key)
        actual[key] = record
        actual_costs[period].append(record.realized_cost)
        actual_terminal_costs.append(record.realized_cost)
    assert tuple(actual_keys) == tuple(expected_keys), (
        "conditional replay settlement row key sequence differs"
    )
    assert len(expected) == len(actual) == expected_count

    for key in expected_keys:
        row = expected[key]
        record = actual[key]
        series = reference_series[record.series_key]
        assert record.session == inputs.session
        assert record.actuals_semantics is inputs.actuals_semantics
        assert record.transition.rule is inputs.stockout_rule
        assert record.holding.rate == series.holding_rate
        assert record.shortage.rate == series.shortage_rate
        _assert_inventory_position(
            record.inventory_position,
            on_hand=row.closing,
            on_order=row.on_order,
            backorders=0.0,
            name=repr(key),
        )
        _assert_few_ulps(record.arrivals, row.arrivals, name=f"{key!r} arrivals")
        _assert_few_ulps(record.transition.demand, row.demand, name=f"{key!r} demand")
        _assert_few_ulps(
            record.transition.fulfilled_demand,
            row.fulfilled,
            name=f"{key!r} fulfilled demand",
        )
        _assert_few_ulps(
            record.transition.closing_on_hand,
            row.closing,
            name=f"{key!r} closing on-hand",
        )
        _assert_few_ulps(
            record.transition.unmet_demand,
            row.shortage,
            name=f"{key!r} shortage",
        )
        _assert_few_ulps(
            record.transition.available_inventory,
            row.opening + row.arrivals,
            name=f"{key!r} available inventory",
        )
        _assert_few_ulps(
            record.holding.amount,
            row.holding_cost,
            name=f"{key!r} holding cost",
        )
        _assert_few_ulps(
            record.shortage.amount,
            row.shortage_cost,
            name=f"{key!r} shortage cost",
        )

    for period in inputs.reference.periods:
        _assert_gamma_sum(
            math.fsum(actual_costs[period]),
            inputs.expectation.cost_by_period[period],
            terms=expected_costs[period],
            name=f"period {period} cost",
        )
    _assert_gamma_sum(
        math.fsum(actual_terminal_costs),
        inputs.expectation.total_cost,
        terms=expected_terminal_costs,
        name="terminal cost",
    )


def _assert_inventory_position(
    actual: InventoryPosition,
    *,
    on_hand: float,
    on_order: float,
    backorders: float,
    name: str,
) -> None:
    _assert_few_ulps(actual.on_hand, on_hand, name=f"{name} inventory on-hand")
    _assert_few_ulps(actual.on_order, on_order, name=f"{name} inventory on-order")
    _assert_few_ulps(actual.backorders, backorders, name=f"{name} inventory backorders")


def _assert_few_ulps(actual: float, expected: float, *, name: str) -> None:
    if actual == expected:
        return
    bound = 4 * max(math.ulp(actual), math.ulp(expected))
    assert abs(actual - expected) <= bound, (
        f"conditional replay {name} differs by more than four ulps: "
        f"actual={actual!r} expected={expected!r} bound={bound!r}"
    )


def _assert_gamma_sum(
    actual: float,
    expected: float,
    *,
    terms: Sequence[float],
    name: str,
) -> None:
    unit_roundoff = sys.float_info.epsilon / 2.0
    gamma_n = len(terms) * unit_roundoff / (1.0 - len(terms) * unit_roundoff)
    magnitude = math.fsum(abs(term) for term in terms)
    component_roundoff = math.fsum(4 * math.ulp(term) for term in terms)
    bound = 2 * gamma_n * magnitude + component_roundoff + 4 * math.ulp(expected)
    assert abs(actual - expected) <= bound, (
        f"conditional replay {name} exceeds its class-2 gamma_n bound: "
        f"actual={actual!r} expected={expected!r} bound={bound!r}"
    )


def _replay_spec_fingerprint(inputs: ReplayInputs) -> str:
    payload = {
        "schema": "newcalibre.vn2-conditional-replay-spec.v1",
        "session": {
            "identity": inputs.session.value,
            "preimage": json.loads(inputs.session.to_bytes()),
        },
        "calendar": {
            "frequency": inputs.calendar.frequency,
            "phase": (None if inputs.calendar.phase is None else inputs.calendar.phase.isoformat()),
        },
        "actuals_semantics": inputs.actuals_semantics.value,
        "stockout_rule": inputs.stockout_rule.value,
        "periods": [period.isoformat() for period in inputs.periods],
        "decision_origins": [origin.isoformat() for origin in inputs.decision_origins],
        "drain_periods": inputs.drain_periods,
        "actuals": [
            [series_key, period.isoformat(), _float_hex(value)]
            for (series_key, period), value in sorted(
                inputs.actuals.items(),
                key=lambda item: (item[0][1], item[0][0].encode()),
            )
        ],
        "initial_arrivals": [
            [series_key, period.isoformat(), _float_hex(value)]
            for (series_key, period), value in sorted(
                inputs.initial_arrivals.items(),
                key=lambda item: (item[0][1], item[0][0].encode()),
            )
        ],
        "initial_positions": [
            [
                key,
                _float_hex(value.on_hand),
                _float_hex(value.on_order),
                _float_hex(value.backorders),
            ]
            for key, value in sorted(
                inputs.initial_positions.items(),
                key=lambda item: item[0].encode(),
            )
        ],
        "orders": [_order_row_payload(order) for order in inputs.orders],
        "reference": {
            "periods": list(inputs.reference.periods),
            "series": [
                [
                    item.series_key,
                    _float_hex(item.initial_on_hand),
                    _float_hex(item.holding_rate),
                    _float_hex(item.shortage_rate),
                ]
                for item in inputs.reference.series
            ],
            "demand": [
                [series_key, period, _float_hex(value)]
                for (series_key, period), value in sorted(
                    inputs.reference.demand.items(),
                    key=lambda item: (item[0][1].encode(), item[0][0].encode()),
                )
            ],
            "orders": [
                [item.series_key, item.origin_index, _float_hex(item.quantity)]
                for item in inputs.reference.orders
            ],
            "lead_time": inputs.reference.lead_time,
            "initial_arrivals": [
                [series_key, period, _float_hex(value)]
                for (series_key, period), value in sorted(
                    inputs.reference.initial_arrivals.items(),
                    key=lambda item: (item[0][1].encode(), item[0][0].encode()),
                )
            ],
        },
    }
    return _fingerprint(payload)


def _order_row_payload(order: OrderRow) -> dict[str, object]:
    return {
        "session": order.session.value,
        "series_key": order.series_key,
        "origin": order.origin.isoformat(),
        "model_name": order.model_name,
        "quantity": _float_hex(order.quantity),
        "arrival_period": order.arrival_period.isoformat(),
        "evidence": (
            None if order.evidence is None else _decision_evidence_payload(order.evidence)
        ),
    }


def _decision_evidence_payload(evidence: DecisionEvidence) -> dict[str, object]:
    return {
        "raw_target": _float_hex(evidence.raw_target),
        "target": _float_hex(evidence.target),
        "source_columns": list(evidence.source_columns),
        "source_descriptor": _guarantee_descriptor_payload(evidence.source_descriptor),
        "effective_descriptor": _guarantee_descriptor_payload(evidence.effective_descriptor),
        "bindings": [
            {
                "name": binding.name,
                "value": _float_hex(binding.value),
                "bound": binding.bound,
            }
            for binding in evidence.bindings
        ],
        "reorder_point": (
            None if evidence.reorder_point is None else _float_hex(evidence.reorder_point)
        ),
    }


def _guarantee_descriptor_payload(descriptor: GuaranteeDescriptor) -> dict[str, object]:
    return {
        "type": {
            "claim": descriptor.type.claim.value,
            "currency": (
                None if descriptor.type.currency is None else descriptor.type.currency.value
            ),
            "declared_slack": (
                None
                if descriptor.type.declared_slack is None
                else _float_hex(descriptor.type.declared_slack)
            ),
        },
        "level": _float_hex(descriptor.level),
        "scored_series": descriptor.scored_series.value,
        "window": descriptor.window.value,
        "scope": {
            "kind": descriptor.scope.kind.value,
            "class_system_name": descriptor.scope.class_system_name,
        },
    }


def _expectation_fingerprint(expectation: ReferenceTrajectory) -> str:
    payload = {
        "cost_by_period": [
            [period, _float_hex(value)] for period, value in expectation.cost_by_period.items()
        ],
        "rows": [
            [
                row.series_key,
                row.period,
                _float_hex(row.opening),
                _float_hex(row.arrivals),
                _float_hex(row.demand),
                _float_hex(row.fulfilled),
                _float_hex(row.closing),
                _float_hex(row.on_order),
                _float_hex(row.shortage),
                _float_hex(row.holding_cost),
                _float_hex(row.shortage_cost),
            ]
            for row in expectation.rows
        ],
        "total_cost": _float_hex(expectation.total_cost),
    }
    return _fingerprint(payload)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float_hex(value: float) -> str:
    return float(value).hex()
