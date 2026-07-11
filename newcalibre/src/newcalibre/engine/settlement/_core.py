"""Apply the single pure inventory transition and accounting implementation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from types import MappingProxyType

import pandas as pd

from newcalibre.domain import CostStructure, DecisionTiming, InventoryPosition, SessionIdentity
from newcalibre.engine._session import session_cost_structure, session_definition
from newcalibre.engine.errors import EngineError
from newcalibre.engine.ports import ActualKey, LedgerSnapshot
from newcalibre.ledger import BookedCost, OrderRow, SettlementRecord, StockoutTransition


class SettlementError(EngineError):
    """Report a settlement request that cannot produce honest durable facts."""


class StockoutRule(StrEnum):
    """Name the stock-out transitions implemented for Gate A."""

    LOST_SALES = "lost-sales"


class ActualsSemantics(StrEnum):
    """Label the meaning of every demand value used for realized cost."""

    DEMAND = "demand"
    CENSORED_SALES_SURROGATE = "censored_sales_surrogate"


@dataclass(frozen=True, slots=True, init=False)
class SettlementRequest:
    """Declare an immutable, calendar-ordered settlement window."""

    session: SessionIdentity
    periods: tuple[pd.Timestamp, ...]
    ledger: LedgerSnapshot
    actuals: Mapping[ActualKey, float]
    inventory_positions: Mapping[str, InventoryPosition]
    orders: tuple[OrderRow, ...]
    timing: DecisionTiming
    stockout_rule: StockoutRule
    actuals_semantics: ActualsSemantics
    cost_structure: CostStructure = field(init=False)

    def __init__(
        self,
        *,
        session: SessionIdentity,
        periods: Sequence[pd.Timestamp],
        ledger: LedgerSnapshot,
        actuals: Mapping[ActualKey, float],
        inventory_positions: Mapping[str, InventoryPosition],
        orders: Sequence[OrderRow] = (),
        timing: DecisionTiming,
        stockout_rule: StockoutRule,
        actuals_semantics: ActualsSemantics,
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("settlement session must be a SessionIdentity")
        if not isinstance(ledger, LedgerSnapshot):
            raise TypeError("settlement ledger must be a LedgerSnapshot")
        if ledger.session != session:
            raise SettlementError("settlement session must match its ledger snapshot")
        if not isinstance(timing, DecisionTiming):
            raise TypeError("settlement timing must be DecisionTiming")
        if timing.lead_time < 1:
            raise SettlementError("settlement lead time must be at least one period")
        if not isinstance(stockout_rule, StockoutRule):
            raise TypeError("settlement stock-out rule must be a StockoutRule")
        if stockout_rule is not StockoutRule.LOST_SALES:
            raise SettlementError(f"unsupported settlement stock-out rule: {stockout_rule!r}")
        if not isinstance(actuals_semantics, ActualsSemantics):
            raise TypeError("settlement actuals semantics must be ActualsSemantics")

        expected_series = _session_series_keys(session, ledger=ledger)
        frozen_periods = _validated_periods(periods, ledger=ledger)
        positions = _validated_positions(inventory_positions)
        if set(positions) != expected_series:
            raise SettlementError(
                "settlement inventory series must exactly match the session series set"
            )
        cost_structure = session_cost_structure(session)
        if cost_structure is None:
            raise SettlementError("settlement session has no decision cost structure")
        _validate_existing_facts(
            ledger=ledger,
            periods=frozen_periods,
            positions=positions,
            timing=timing,
            stockout_rule=stockout_rule,
            actuals_semantics=actuals_semantics,
            cost_structure=cost_structure,
        )
        staged_orders = _validated_orders(
            orders,
            session=session,
            periods=frozen_periods,
            series_keys=set(positions),
            timing=timing,
            ledger=ledger,
        )
        frozen_actuals = _validated_actuals(
            actuals,
            expected={
                (series_key, period) for period in frozen_periods for series_key in positions
            },
        )
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "periods", frozen_periods)
        object.__setattr__(self, "ledger", ledger)
        object.__setattr__(self, "actuals", MappingProxyType(frozen_actuals))
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        object.__setattr__(self, "orders", staged_orders)
        object.__setattr__(self, "timing", timing)
        object.__setattr__(self, "stockout_rule", stockout_rule)
        object.__setattr__(self, "actuals_semantics", actuals_semantics)
        object.__setattr__(self, "cost_structure", cost_structure)


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

    series_keys = tuple(sorted(request.inventory_positions, key=str.encode))
    positions = dict(request.inventory_positions)
    settled_periods = {(record.series_key, record.period) for record in request.ledger.settlements}
    open_orders: dict[str, list[OrderRow]] = {series_key: [] for series_key in series_keys}
    for order in request.ledger.orders:
        if (
            order.series_key in open_orders
            and (
                order.series_key,
                order.arrival_period,
            )
            not in settled_periods
        ):
            open_orders[order.series_key].append(order)
    staged_by_origin: dict[pd.Timestamp, list[OrderRow]] = {
        period: [] for period in request.periods
    }
    for order in request.orders:
        staged_by_origin[order.origin].append(order)

    records: list[SettlementRecord] = []
    for period in request.periods:
        for series_key in series_keys:
            opening = positions[series_key]
            due, future = _partition_orders(open_orders[series_key], period=period)
            arrivals = _finite_sum(
                (order.quantity for order in due),
                name="settlement arrivals",
            )
            demand = request.actuals[(series_key, period)]
            transition = _lost_sales_transition(
                opening=opening,
                arrivals=arrivals,
                demand=demand,
            )
            current_orders = [
                order for order in staged_by_origin[period] if order.series_key == series_key
            ]
            next_open = [*future, *current_orders]
            on_order = _finite_sum(
                (order.quantity for order in next_open),
                name="open-order quantity",
            )
            positions[series_key] = InventoryPosition(
                on_hand=transition.closing_on_hand,
                on_order=on_order,
                backorders=transition.closing_backorders,
            )
            records.append(
                SettlementRecord(
                    session=request.session,
                    series_key=series_key,
                    period=period,
                    arrivals=arrivals,
                    actuals_semantics=request.actuals_semantics.value,
                    transition=transition,
                    holding=_book_cost(
                        request.cost_structure.holding,
                        transition.closing_on_hand,
                    ),
                    shortage=_book_cost(
                        request.cost_structure.shortage,
                        transition.unmet_demand,
                    ),
                )
            )
            open_orders[series_key] = next_open

    return SettlementResult(records=tuple(records), inventory_positions=positions)


def _validated_periods(
    periods: Sequence[pd.Timestamp],
    *,
    ledger: LedgerSnapshot,
) -> tuple[pd.Timestamp, ...]:
    if isinstance(periods, (str, bytes)):
        raise TypeError("settlement periods must be a sequence of timestamps")
    frozen = tuple(periods)
    if not frozen:
        raise SettlementError("settlement periods must not be empty")
    for index, period in enumerate(frozen):
        try:
            ledger.calendar.require_member(period, name="settlement period")
        except (TypeError, ValueError) as error:
            raise SettlementError(str(error)) from error
        if index and period != ledger.calendar.advance(frozen[index - 1], 1):
            raise SettlementError("settlement periods must be contiguous calendar members")
    return frozen


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


def _validated_actuals(
    value: Mapping[ActualKey, float],
    *,
    expected: set[ActualKey],
) -> dict[ActualKey, float]:
    if not isinstance(value, Mapping):
        raise TypeError("settlement actuals must be a mapping")
    try:
        supplied = set(value)
    except (TypeError, ValueError) as error:
        raise TypeError("settlement actual keys must be hashable") from error
    if supplied != expected:
        missing = sorted(expected - supplied, key=repr)
        extra = sorted(supplied - expected, key=repr)
        raise SettlementError(
            f"settlement actual keys must exactly match the window; missing={missing!r}, "
            f"extra={extra!r}"
        )
    actuals: dict[ActualKey, float] = {}
    for key in sorted(expected, key=repr):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError("settlement demand must be a real number")
        demand = float(raw)
        if not math.isfinite(demand) or demand < 0.0:
            raise SettlementError("settlement demand must be finite and non-negative")
        actuals[key] = 0.0 if demand == 0.0 else demand
    return actuals


def _validated_orders(
    value: Sequence[OrderRow],
    *,
    session: SessionIdentity,
    periods: tuple[pd.Timestamp, ...],
    series_keys: set[str],
    timing: DecisionTiming,
    ledger: LedgerSnapshot,
) -> tuple[OrderRow, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("settlement staged orders must be a sequence")
    orders = tuple(value)
    existing_keys = {order.key for order in ledger.orders}
    staged_keys: set[object] = set()
    for order in orders:
        if not isinstance(order, OrderRow):
            raise TypeError("settlement staged orders must contain OrderRow values")
        if order.session != session:
            raise SettlementError("staged order session must match the settlement session")
        if order.series_key not in series_keys:
            raise SettlementError("staged order has no opening inventory position")
        if order.origin not in periods:
            raise SettlementError("staged order origin must lie inside the settlement window")
        _require_arrival_law(order, timing=timing, ledger=ledger)
        if order.key in existing_keys or order.key in staged_keys:
            raise SettlementError(f"duplicate staged order key: {order.key!r}")
        staged_keys.add(order.key)
    return orders


def _validate_existing_facts(
    *,
    ledger: LedgerSnapshot,
    periods: tuple[pd.Timestamp, ...],
    positions: Mapping[str, InventoryPosition],
    timing: DecisionTiming,
    stockout_rule: StockoutRule,
    actuals_semantics: ActualsSemantics,
    cost_structure: CostStructure,
) -> None:
    settlement_keys: set[object] = set()
    settlement_series: dict[pd.Timestamp, set[str]] = {}
    latest_by_series: dict[str, SettlementRecord | None] = {
        series_key: None for series_key in positions
    }
    for record in ledger.settlements:
        if not isinstance(record, SettlementRecord):
            raise TypeError("ledger settlements must contain SettlementRecord values")
        if record.session != ledger.session:
            raise SettlementError("ledger settlement session must match its snapshot")
        if record.series_key not in positions:
            raise SettlementError("ledger settlement series must belong to its session")
        if record.transition.rule != stockout_rule.value:
            raise SettlementError("ledger settlement stock-out rule must match the request")
        if record.actuals_semantics != actuals_semantics.value:
            raise SettlementError("ledger settlement actuals semantics must match the request")
        if (
            record.holding.rate != cost_structure.holding
            or record.shortage.rate != cost_structure.shortage
        ):
            raise SettlementError("ledger settlement cost rates must match the session")
        try:
            ledger.calendar.require_member(record.period, name="ledger settlement period")
        except (TypeError, ValueError) as error:
            raise SettlementError(str(error)) from error
        if record.key in settlement_keys:
            raise SettlementError(f"duplicate ledger settlement key: {record.key!r}")
        settlement_keys.add(record.key)
        settlement_series.setdefault(record.period, set()).add(record.series_key)
        latest = latest_by_series[record.series_key]
        if latest is None or record.period > latest.period:
            latest_by_series[record.series_key] = record

    requested_keys = {
        (ledger.session, series_key, period) for period in periods for series_key in positions
    }
    duplicate = next(
        (record.key for record in ledger.settlements if record.key in requested_keys),
        None,
    )
    if duplicate is not None:
        raise SettlementError(f"settlement period is already booked: {duplicate!r}")

    if settlement_series:
        expected_series = set(positions)
        if any(series != expected_series for series in settlement_series.values()):
            raise SettlementError("ledger settlement history must contain every session series")
        history_periods = tuple(sorted(settlement_series))
        if any(
            period != ledger.calendar.advance(previous, 1)
            for previous, period in zip(history_periods, history_periods[1:], strict=False)
        ):
            raise SettlementError("ledger settlement history must be calendar-contiguous")
        frontier_periods = {
            record.period for record in latest_by_series.values() if record is not None
        }
        if len(frontier_periods) != 1:
            raise SettlementError("ledger settlement history must share one series frontier")
        frontier = next(iter(frontier_periods))
        if periods[0] != ledger.calendar.advance(frontier, 1):
            raise SettlementError(
                "settlement window must immediately follow the ledger settlement frontier"
            )
        for series_key, latest in latest_by_series.items():
            if latest is None:
                raise SettlementError("ledger settlement history is missing a session series")
            opening = positions[series_key]
            if opening.on_hand != latest.transition.closing_on_hand:
                raise SettlementError(
                    f"opening on_hand for {series_key!r} must match the settlement frontier"
                )
            if opening.backorders != latest.transition.closing_backorders:
                raise SettlementError(
                    f"opening backorders for {series_key!r} must match the settlement frontier"
                )

    settled_periods = {(record.series_key, record.period) for record in ledger.settlements}
    open_by_series: dict[str, list[OrderRow]] = {series_key: [] for series_key in positions}
    order_keys: set[object] = set()
    for order in ledger.orders:
        if not isinstance(order, OrderRow):
            raise TypeError("ledger orders must contain OrderRow values")
        if order.session != ledger.session:
            raise SettlementError("ledger order session must match its snapshot")
        if order.series_key not in positions:
            raise SettlementError("ledger order series must belong to its session")
        if order.key in order_keys:
            raise SettlementError(f"duplicate ledger order key: {order.key!r}")
        order_keys.add(order.key)
        if order.origin >= periods[0]:
            raise SettlementError("ledger orders must predate the settlement window")
        _require_arrival_law(order, timing=timing, ledger=ledger)
        if (order.series_key, order.arrival_period) in settled_periods:
            continue
        if order.arrival_period < periods[0]:
            raise SettlementError("open order is overdue before the settlement window")
        open_by_series[order.series_key].append(order)

    for series_key, position in positions.items():
        if position.backorders != 0.0:
            raise SettlementError("lost-sales settlement requires zero opening backorders")
        derived_on_order = _finite_sum(
            (order.quantity for order in open_by_series[series_key]),
            name="open-order quantity",
        )
        if position.on_order != derived_on_order:
            raise SettlementError(
                f"inventory on_order for {series_key!r} does not match ledger open orders"
            )


def _require_arrival_law(
    order: OrderRow,
    *,
    timing: DecisionTiming,
    ledger: LedgerSnapshot,
) -> None:
    try:
        expected = ledger.calendar.advance(order.origin, timing.lead_time)
    except (TypeError, ValueError) as error:
        raise SettlementError(
            f"order origin is invalid for the ledger calendar: {error}"
        ) from error
    if order.arrival_period != expected:
        raise SettlementError(
            f"order arrival {order.arrival_period!s} must equal calendar.advance"
            f"({order.origin!s}, {timing.lead_time})"
        )


def _partition_orders(
    orders: Sequence[OrderRow],
    *,
    period: pd.Timestamp,
) -> tuple[list[OrderRow], list[OrderRow]]:
    overdue = next((order for order in orders if order.arrival_period < period), None)
    if overdue is not None:
        raise SettlementError(f"open order is overdue at {period}: {overdue.key!r}")
    due = [order for order in orders if order.arrival_period == period]
    future = [order for order in orders if order.arrival_period > period]
    return due, future


def _lost_sales_transition(
    *,
    opening: InventoryPosition,
    arrivals: float,
    demand: float,
) -> StockoutTransition:
    if opening.backorders != 0.0:
        raise SettlementError("lost-sales settlement requires zero opening backorders")
    available = _finite_sum((opening.on_hand, arrivals), name="available inventory")
    fulfilled = min(available, demand)
    unmet = demand - fulfilled
    closing = available - fulfilled
    return StockoutTransition(
        rule=StockoutRule.LOST_SALES.value,
        demand=demand,
        fulfilled_demand=fulfilled,
        unmet_demand=unmet,
        closing_on_hand=closing,
        closing_backorders=0.0,
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


def _require_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SettlementError(f"{name} must be a non-empty trimmed string")


def _session_series_keys(
    session: SessionIdentity,
    *,
    ledger: LedgerSnapshot,
) -> set[str]:
    definition = session_definition(session)
    series_set = definition.get("series_set")
    frequency = definition.get("calendar_frequency")
    if (
        not isinstance(series_set, list)
        or not series_set
        or any(not isinstance(series_key, str) for series_key in series_set)
    ):
        raise SettlementError("settlement session has an invalid series set")
    if frequency != ledger.calendar.frequency:
        raise SettlementError("settlement ledger calendar must match its session")
    return {series_key for series_key in series_set if isinstance(series_key, str)}


__all__ = [
    "ActualsSemantics",
    "SettlementError",
    "SettlementRequest",
    "SettlementResult",
    "StockoutRule",
    "settle",
]
