"""Apply the single pure inventory transition and accounting implementation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    DecisionTiming,
    InventoryPosition,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine._session import (
    SessionCosts,
    decision_from_definition,
    session_definition,
    session_series_and_frequency,
)
from newcalibre.engine.errors import EngineError
from newcalibre.engine.ports import ActualKey, SettlementSnapshot
from newcalibre.ledger import (
    BookedCost,
    LedgerError,
    OrderKey,
    OrderRow,
    SettlementRecord,
    lost_sales_transition,
)


class SettlementError(EngineError):
    """Report a settlement request that cannot produce honest durable facts."""


@dataclass(frozen=True, slots=True, init=False)
class SettlementRequest:
    """Declare an immutable, calendar-ordered settlement window."""

    session: SessionIdentity
    snapshot: SettlementSnapshot
    actuals: Mapping[ActualKey, float]
    inventory_positions: Mapping[str, InventoryPosition]
    orders: tuple[OrderRow, ...]
    actuals_semantics: ActualsSemantics
    costs_by_series: SessionCosts = field(init=False)
    _series_keys: tuple[str, ...] = field(repr=False)

    def __init__(
        self,
        *,
        session: SessionIdentity,
        snapshot: SettlementSnapshot,
        actuals: Mapping[ActualKey, float],
        inventory_positions: Mapping[str, InventoryPosition],
        orders: Sequence[OrderRow] = (),
        actuals_semantics: ActualsSemantics,
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("settlement session must be a SessionIdentity")
        if not isinstance(snapshot, SettlementSnapshot):
            raise TypeError("settlement snapshot must be a SettlementSnapshot")
        if snapshot.session != session:
            raise SettlementError("settlement session must match its snapshot")
        if not isinstance(actuals_semantics, ActualsSemantics):
            raise TypeError("settlement actuals semantics must be ActualsSemantics")

        definition = session_definition(session)
        _session_series, frequency = session_series_and_frequency(definition)
        if frequency != snapshot.calendar.frequency:
            raise SettlementError("settlement snapshot calendar must match its session")
        decision = decision_from_definition(definition)
        if decision is None:
            raise SettlementError("settlement session has no decision configuration")
        series_keys = decision.series_keys
        series_key_set = set(series_keys)
        period_set = set(snapshot.periods)
        positions = _validated_positions(inventory_positions)
        if set(positions) != series_key_set:
            raise SettlementError(
                "settlement inventory series must exactly match the decision series set"
            )
        costs_by_series = decision.costs_by_series
        timing = decision.timing
        stockout_rule = decision.stockout_rule
        if timing.lead_time < 1:
            raise SettlementError("settlement lead time must be at least one period")
        if stockout_rule is not StockoutRule.LOST_SALES:
            raise SettlementError(f"unsupported settlement stock-out rule: {stockout_rule!r}")
        validate_snapshot_state(
            snapshot=snapshot,
            positions=positions,
            series_keys=series_keys,
            actuals_semantics=actuals_semantics,
        )
        staged_orders = _validated_orders(
            orders,
            session=session,
            period_set=period_set,
            series_keys=series_key_set,
            timing=timing,
            snapshot=snapshot,
        )
        frozen_actuals = validate_actuals_window(
            actuals,
            periods=snapshot.periods,
            series_keys=series_keys,
        )
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "actuals", MappingProxyType(frozen_actuals))
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        object.__setattr__(self, "orders", staged_orders)
        object.__setattr__(self, "actuals_semantics", actuals_semantics)
        object.__setattr__(self, "costs_by_series", costs_by_series)
        object.__setattr__(self, "_series_keys", series_keys)


@dataclass(frozen=True, slots=True)
class SettlementResult:
    """Return append-only settlement facts and the resulting inventory positions."""

    records: tuple[SettlementRecord, ...]
    inventory_positions: Mapping[str, InventoryPosition]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, SettlementRecord) for record in records):
            raise TypeError("settlement result records must contain SettlementRecord values")
        positions = _validated_positions(self.inventory_positions)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))


def settle(request: SettlementRequest) -> SettlementResult:
    """Settle an explicit window without I/O or caller mutation."""
    if not isinstance(request, SettlementRequest):
        raise TypeError("settle requires a SettlementRequest")

    series_keys = request._series_keys
    positions = {series_key: request.inventory_positions[series_key] for series_key in series_keys}
    staged_by_origin: dict[pd.Timestamp, dict[str, list[float]]] = {}
    staged_by_arrival: dict[pd.Timestamp, dict[str, list[float]]] = {}
    for order in request.orders:
        staged_by_origin.setdefault(order.origin, {}).setdefault(order.series_key, []).append(
            order.quantity
        )
        staged_by_arrival.setdefault(order.arrival_period, {}).setdefault(
            order.series_key,
            [],
        ).append(order.quantity)

    records: list[SettlementRecord] = []
    for period in request.snapshot.periods:
        staged_origin_by_series = staged_by_origin.get(period, {})
        staged_arrival_by_series = staged_by_arrival.get(period, {})
        for series_key in series_keys:
            cost_structure = request.costs_by_series[series_key]
            opening = positions[series_key]
            arrivals = _finite_sum(
                (
                    request.snapshot.due_arrivals.get((series_key, period), 0.0),
                    *staged_arrival_by_series.get(series_key, ()),
                ),
                name="settlement arrivals",
            )
            demand = request.actuals[(series_key, period)]
            try:
                transition = lost_sales_transition(
                    opening=opening,
                    arrivals=arrivals,
                    demand=demand,
                )
            except LedgerError as error:
                raise SettlementError(str(error)) from error
            current_quantity = _finite_sum(
                (
                    request.snapshot.origin_order_quantities.get((series_key, period), 0.0),
                    *staged_origin_by_series.get(series_key, ()),
                ),
                name="current-order quantity",
            )
            on_order = _finite_nonnegative_sum(
                (opening.on_order, -arrivals, current_quantity),
                name="open-order quantity",
            )
            next_position = InventoryPosition(
                on_hand=transition.closing_on_hand,
                on_order=on_order,
                backorders=transition.closing_backorders,
            )
            positions[series_key] = next_position
            records.append(
                SettlementRecord(
                    session=request.session,
                    series_key=series_key,
                    period=period,
                    arrivals=arrivals,
                    actuals_semantics=request.actuals_semantics,
                    transition=transition,
                    inventory_position=next_position,
                    holding=_book_cost(
                        cost_structure.holding,
                        transition.closing_on_hand,
                    ),
                    shortage=_book_cost(
                        cost_structure.shortage,
                        transition.unmet_demand,
                    ),
                )
            )

    return SettlementResult(records=tuple(records), inventory_positions=positions)


def validate_snapshot_state(
    *,
    snapshot: SettlementSnapshot,
    positions: Mapping[str, InventoryPosition],
    series_keys: tuple[str, ...],
    actuals_semantics: ActualsSemantics,
) -> None:
    series_key_set = set(series_keys)
    if set(snapshot.open_order_quantities) != series_key_set:
        raise SettlementError(
            "settlement snapshot open quantities must exactly match the decision series set"
        )
    unknown_due_series = sorted(
        {series_key for series_key, _period in snapshot.due_arrivals} - series_key_set,
        key=lambda value: value.encode(),
    )
    if unknown_due_series:
        raise SettlementError(
            f"settlement snapshot due arrivals contain unknown series: {unknown_due_series!r}"
        )
    if snapshot.frontier is None:
        if snapshot.latest_positions:
            raise SettlementError(
                "settlement snapshot cannot expose latest positions without a frontier"
            )
    else:
        if snapshot.periods[0] != snapshot.calendar.advance(snapshot.frontier, 1):
            raise SettlementError(
                "settlement window must immediately follow the ledger settlement frontier"
            )
        if set(snapshot.latest_positions) != series_key_set:
            raise SettlementError(
                "settlement snapshot latest positions must cover every decision series"
            )
        if dict(positions) != dict(snapshot.latest_positions):
            raise SettlementError(
                "settlement opening inventory must match the durable settlement frontier"
            )
        if snapshot.actuals_semantics is None:
            raise SettlementError("settlement snapshot frontier is missing actuals semantics")
    if (
        snapshot.actuals_semantics is not None
        and snapshot.actuals_semantics is not actuals_semantics
    ):
        raise SettlementError("settlement actuals semantics must match the durable session")
    for series_key, position in positions.items():
        if position.backorders != 0.0:
            raise SettlementError("lost-sales settlement requires zero opening backorders")
        if snapshot.window_opening_positions:
            expected_position = snapshot.window_opening_positions.get(series_key)
            if expected_position != position:
                raise SettlementError(
                    "settlement opening inventory must match the compact ledger index"
                )
            expected_opening = position.on_order
        else:
            window_orders = _finite_sum(
                (
                    snapshot.origin_order_quantities.get((series_key, period), 0.0)
                    for period in snapshot.periods
                ),
                name="settlement window order quantity",
            )
            expected_opening = _finite_nonnegative_sum(
                (snapshot.open_order_quantities[series_key], -window_orders),
                name="settlement opening on_order",
            )
        if position.on_order != expected_opening:
            raise SettlementError(
                f"inventory on_order for {series_key!r} does not match the compact ledger index"
            )


def _validated_positions(
    value: Mapping[str, InventoryPosition],
) -> dict[str, InventoryPosition]:
    if not isinstance(value, Mapping):
        raise TypeError("settlement inventory positions must be a mapping")
    if not value:
        raise SettlementError("settlement inventory positions must not be empty")
    positions: dict[str, InventoryPosition] = {}
    for series_key, position in value.items():
        _require_identifier(series_key, name="inventory series key")
        if not isinstance(position, InventoryPosition):
            raise TypeError("settlement inventory positions must contain InventoryPosition values")
        positions[series_key] = position
    return positions


def validate_actuals_window(
    value: Mapping[ActualKey, float],
    *,
    periods: tuple[pd.Timestamp, ...],
    series_keys: tuple[str, ...],
) -> dict[ActualKey, float]:
    """Validate and normalize one exact settlement-demand window."""
    if not isinstance(value, Mapping):
        raise TypeError("settlement actuals must be a mapping")
    period_set = set(periods)
    series_key_set = set(series_keys)
    expected_count = len(periods) * len(series_keys)
    exact_window = len(value) == expected_count and all(
        _actual_key_in_window(
            key,
            period_set=period_set,
            series_key_set=series_key_set,
        )
        for key in value
    )
    if not exact_window:
        expected = {(series_key, period) for period in periods for series_key in series_keys}
        try:
            supplied = set(value)
        except (TypeError, ValueError) as error:
            raise TypeError("settlement actual keys must be hashable") from error
        missing = sorted(expected - supplied, key=repr)
        extra = sorted(supplied - expected, key=repr)
        raise SettlementError(
            f"settlement actual keys must exactly match the window; missing={missing!r}, "
            f"extra={extra!r}"
        )
    actuals: dict[ActualKey, float] = {}
    for period in periods:
        for series_key in series_keys:
            key = (series_key, period)
            raw = value[key]
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"settlement demand for {key!r} must be a real number")
            try:
                demand = float(raw)
            except (OverflowError, TypeError, ValueError) as error:
                raise SettlementError(
                    f"settlement demand for {key!r} must be finite and non-negative"
                ) from error
            if not math.isfinite(demand) or demand < 0.0:
                raise SettlementError(
                    f"settlement demand for {key!r} must be finite and non-negative"
                )
            actuals[key] = 0.0 if demand == 0.0 else demand
    return actuals


def _actual_key_in_window(
    key: object,
    *,
    period_set: set[pd.Timestamp],
    series_key_set: set[str],
) -> bool:
    if not isinstance(key, tuple) or len(key) != 2:
        return False
    series_key, period = key
    if not isinstance(series_key, str) or not isinstance(period, pd.Timestamp):
        return False
    return series_key in series_key_set and period in period_set


def _validated_orders(
    value: Sequence[OrderRow],
    *,
    session: SessionIdentity,
    period_set: set[pd.Timestamp],
    series_keys: set[str],
    timing: DecisionTiming,
    snapshot: SettlementSnapshot,
) -> tuple[OrderRow, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("settlement staged orders must be a sequence")
    orders = tuple(value)
    staged_keys: set[OrderKey] = set()
    for order in orders:
        if not isinstance(order, OrderRow):
            raise TypeError("settlement staged orders must contain OrderRow values")
        if order.session != session:
            raise SettlementError("staged order session must match the settlement session")
        if order.series_key not in series_keys:
            raise SettlementError("staged order has no opening inventory position")
        if order.origin not in period_set:
            raise SettlementError("staged order origin must lie inside the settlement window")
        _require_arrival_law(order, timing=timing, snapshot=snapshot)
        if order.key in staged_keys:
            raise SettlementError(f"duplicate staged order key: {order.key!r}")
        staged_keys.add(order.key)
    return orders


def _require_arrival_law(
    order: OrderRow,
    *,
    timing: DecisionTiming,
    snapshot: SettlementSnapshot,
) -> None:
    try:
        expected = snapshot.calendar.advance(order.origin, timing.lead_time)
    except (TypeError, ValueError) as error:
        raise SettlementError(
            f"order origin is invalid for the ledger calendar: {error}"
        ) from error
    if order.arrival_period != expected:
        raise SettlementError(
            f"order arrival {order.arrival_period!s} must equal calendar.advance"
            f"({order.origin!s}, {timing.lead_time})"
        )


def _book_cost(rate: float, basis: float) -> BookedCost:
    try:
        amount = rate * basis
    except OverflowError as error:
        raise SettlementError("settlement cost exceeds finite float range") from error
    if not math.isfinite(amount):
        raise SettlementError("settlement cost exceeds finite float range")
    return BookedCost(rate=rate, basis=basis, amount=amount)


def _finite_sum(values: Iterable[float], *, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise SettlementError(f"{name} exceeds finite float range") from error
    if not math.isfinite(result):
        raise SettlementError(f"{name} exceeds finite float range")
    return 0.0 if result == 0.0 else result


def _finite_nonnegative_sum(values: Iterable[float], *, name: str) -> float:
    result = _finite_sum(values, name=name)
    if result < 0.0:
        raise SettlementError(f"{name} must be non-negative")
    return result


def _require_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise SettlementError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise SettlementError(f"{name} must be valid UTF-8") from error


__all__ = [
    "SettlementError",
    "SettlementRequest",
    "SettlementResult",
    "settle",
    "validate_actuals_window",
    "validate_snapshot_state",
]
