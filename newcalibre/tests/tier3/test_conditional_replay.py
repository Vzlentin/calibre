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

from newcalibre.domain import InventoryPosition, SessionIdentity
from newcalibre.engine import InMemoryLedgerSink, SettlementRequest, SettlementResult, settle
from newcalibre.ledger import OrderRow, SettlementRecord
from newcalibre.oracle import CaptureBundle
from newcalibre.protocols.vn2 import VN2Dataset
from oracle.reference import (
    ReferenceOrder,
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
class CanonicalReplay:
    """Own immutable shared inputs, the prior expectation, and successor facts."""

    dataset: VN2Dataset
    session: SessionIdentity
    periods: tuple[pd.Timestamp, ...]
    initial_positions: Mapping[str, InventoryPosition]
    initial_arrivals: Mapping[tuple[str, pd.Timestamp], float]
    actuals: Mapping[tuple[str, pd.Timestamp], float]
    captured_decisions: tuple[CapturedDecision, ...]
    orders: tuple[OrderRow, ...]
    reference_series: tuple[ReferenceSeries, ...]
    reference_periods: tuple[str, ...]
    reference_demand: Mapping[tuple[str, str], float]
    reference_arrivals: Mapping[tuple[str, str], float]
    expectation: ReferenceTrajectory
    records: tuple[SettlementRecord, ...]
    shared_fingerprint: str
    expectation_fingerprint: str
    settle_calls: int


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
    assert canonical_replay.settle_calls == 1
    _assert_conditional_replay(canonical_replay, canonical_replay.records)


@pytest.mark.oracle_witness("vn2-conditional-replay")
def test_conditional_replay_rejects_one_successor_order_unit(
    canonical_replay: CanonicalReplay,
) -> None:
    before = (
        _shared_fingerprint(canonical_replay),
        _expectation_fingerprint(canonical_replay.expectation),
    )
    orders = list(canonical_replay.orders)
    last_origin = max(order.origin for order in orders)
    target_index = min(
        (index for index, order in enumerate(orders) if order.origin == last_origin),
        key=lambda index: orders[index].series_key.encode(),
    )
    orders[target_index] = replace(
        orders[target_index],
        quantity=orders[target_index].quantity + 1.0,
    )

    drifted = _settle_once(canonical_replay, tuple(orders))

    with pytest.raises(AssertionError, match="conditional replay"):
        _assert_conditional_replay(canonical_replay, drifted.records)
    assert before == (
        _shared_fingerprint(canonical_replay),
        _expectation_fingerprint(canonical_replay.expectation),
    )


def test_replay_fingerprints_reject_shared_or_expectation_replacement(
    canonical_replay: CanonicalReplay,
) -> None:
    first = canonical_replay.captured_decisions[0]
    changed_decisions = (
        replace(first, quantity=first.quantity + 1.0),
        *canonical_replay.captured_decisions[1:],
    )
    replaced_shared = replace(
        canonical_replay,
        captured_decisions=changed_decisions,
    )
    with pytest.raises(AssertionError, match="shared-input fingerprint"):
        _assert_conditional_replay(replaced_shared, canonical_replay.records)

    replaced_expectation = replace(
        canonical_replay.expectation,
        total_cost=canonical_replay.expectation.total_cost + 1.0,
    )
    replaced_replay = replace(canonical_replay, expectation=replaced_expectation)
    with pytest.raises(AssertionError, match="expectation fingerprint"):
        _assert_conditional_replay(replaced_replay, canonical_replay.records)

    demand = cast(dict[tuple[str, str], float], canonical_replay.reference_demand)
    with pytest.raises(TypeError):
        demand[next(iter(demand))] = 0.0


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
    expectation = calculate_reference_trajectory(
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

    replay_without_records = CanonicalReplay(
        dataset=dataset,
        session=session,
        periods=periods,
        initial_positions=MappingProxyType(initial_positions),
        initial_arrivals=MappingProxyType(initial_arrivals),
        actuals=MappingProxyType(actuals),
        captured_decisions=frozen_decisions,
        orders=orders,
        reference_series=tuple(reference_series),
        reference_periods=reference_periods,
        reference_demand=MappingProxyType(reference_demand),
        reference_arrivals=MappingProxyType(reference_arrivals),
        expectation=expectation,
        records=(),
        shared_fingerprint="",
        expectation_fingerprint=expectation_fingerprint,
        settle_calls=0,
    )
    shared_fingerprint = _shared_fingerprint(replay_without_records)
    result = _settle_once(replay_without_records, orders)
    return replace(
        replay_without_records,
        records=result.records,
        shared_fingerprint=shared_fingerprint,
        settle_calls=1,
    )


def _settle_once(replay: CanonicalReplay, orders: Sequence[OrderRow]) -> SettlementResult:
    sink = InMemoryLedgerSink(
        session=replay.session,
        calendar=replay.dataset.config.calendar,
        initial_arrivals=replay.initial_arrivals,
    )
    request = SettlementRequest(
        session=replay.session,
        snapshot=sink.settlement_snapshot(replay.periods),
        actuals=replay.actuals,
        inventory_positions=replay.initial_positions,
        orders=orders,
        actuals_semantics=replay.dataset.config.actuals_semantics,
    )
    return settle(request)


def _assert_conditional_replay(
    replay: CanonicalReplay,
    records: Sequence[SettlementRecord],
) -> None:
    assert _shared_fingerprint(replay) == replay.shared_fingerprint, (
        "conditional replay shared-input fingerprint changed"
    )
    assert _expectation_fingerprint(replay.expectation) == replay.expectation_fingerprint, (
        "conditional replay expectation fingerprint changed"
    )
    expected = {(row.series_key, row.period): row for row in replay.expectation.rows}
    actual = {(record.series_key, record.period.strftime("%Y-%m-%d")): record for record in records}
    assert len(expected) == len(actual) == 599 * 8 == 4_792
    assert set(actual) == set(expected)

    for key in sorted(expected, key=lambda value: (value[1].encode(), value[0].encode())):
        row = expected[key]
        record = actual[key]
        assert record.session == replay.session
        assert record.actuals_semantics is replay.dataset.config.actuals_semantics
        assert record.transition.rule is replay.dataset.config.stockout_rule
        assert record.holding.rate == replay.dataset.config.holding_rate
        assert record.shortage.rate == replay.dataset.config.shortage_rate
        assert record.inventory_position.backorders == 0.0
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

    for period in replay.reference_periods:
        terms = tuple(row.total_cost for row in replay.expectation.rows if row.period == period)
        actual_cost = math.fsum(
            record.realized_cost
            for record in records
            if record.period.strftime("%Y-%m-%d") == period
        )
        _assert_gamma_sum(
            actual_cost,
            replay.expectation.cost_by_period[period],
            terms=terms,
            name=f"period {period} cost",
        )
    _assert_gamma_sum(
        math.fsum(record.realized_cost for record in records),
        replay.expectation.total_cost,
        terms=tuple(row.total_cost for row in replay.expectation.rows),
        name="terminal cost",
    )


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


def _shared_fingerprint(replay: CanonicalReplay) -> str:
    payload = {
        "actuals": [
            [series_key, period.isoformat(), _float_hex(value)]
            for (series_key, period), value in sorted(
                replay.actuals.items(),
                key=lambda item: (item[0][1], item[0][0].encode()),
            )
        ],
        "captured_decisions": [
            [item.series_key, item.origin_index, _float_hex(item.quantity)]
            for item in replay.captured_decisions
        ],
        "initial_arrivals": [
            [series_key, period.isoformat(), _float_hex(value)]
            for (series_key, period), value in sorted(
                replay.initial_arrivals.items(),
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
                replay.initial_positions.items(),
                key=lambda item: item[0].encode(),
            )
        ],
    }
    return _fingerprint(payload)


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
