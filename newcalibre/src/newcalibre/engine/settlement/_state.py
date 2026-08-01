"""Maintain the compact incremental state behind settlement snapshots."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from types import MappingProxyType

import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionTiming,
    InventoryPosition,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine._session import SessionCosts
from newcalibre.engine.run_store import ActualKey, SettlementSnapshot
from newcalibre.ledger import (
    LedgerError,
    OrderKey,
    OrderRow,
    SettlementRecord,
    validate_lost_sales_transition,
)


@dataclass(frozen=True, slots=True)
class SettlementIndexWork:
    """Count only newly indexed and newly resolved settlement facts."""

    new_orders: int = 0
    settlement_records: int = 0
    due_orders: int = 0


@dataclass(frozen=True, slots=True)
class SettlementIndexAudit:
    """Expose deterministic index work and compact active-state cardinality."""

    active_orders: int
    due_buckets: int
    last_work: SettlementIndexWork
    rebuild_work: SettlementIndexWork | None


@dataclass(frozen=True, slots=True)
class _SettlementDelta:
    """Carry a fully validated, no-fail mutation of the settlement index."""

    added_orders: tuple[OrderRow, ...]
    consumed_due_keys: tuple[ActualKey, ...]
    open_quantities: dict[str, float]
    inventory_positions: dict[str, InventoryPosition]
    frontier: pd.Timestamp | None
    actuals_semantics: ActualsSemantics | None
    work: SettlementIndexWork


class SettlementIndex:
    """Index active orders and the durable settlement frontier incrementally."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        series_keys: Sequence[str],
        costs_by_series: SessionCosts,
        timing: DecisionTiming,
        stockout_rule: StockoutRule,
        initial_arrivals: Mapping[ActualKey, float] | None = None,
    ) -> None:
        self._session = session
        self._calendar = calendar
        self._series_keys = tuple(series_keys)
        self._series_key_set = set(self._series_keys)
        self._costs_by_series = _costs_by_series(
            costs_by_series,
            series_keys=self._series_keys,
        )
        self._timing = timing
        self._stockout_rule = stockout_rule
        self._initial_arrival_seed = _validated_initial_arrivals(
            initial_arrivals,
            calendar=calendar,
            series_keys=self._series_key_set,
        )
        self._initial_due_arrivals = dict(self._initial_arrival_seed)
        self._active_orders: set[OrderKey] = set()
        self._due_orders: dict[ActualKey, dict[OrderKey, float]] = {}
        self._origin_orders: dict[ActualKey, dict[OrderKey, float]] = {}
        self._due_heap: list[tuple[pd.Timestamp, str]] = [
            (period, series_key) for series_key, period in self._initial_due_arrivals
        ]
        heapq.heapify(self._due_heap)
        initial_quantities = {series_key: [] for series_key in self._series_keys}
        for (series_key, _period), quantity in self._initial_due_arrivals.items():
            initial_quantities[series_key].append(quantity)
        self._open_quantities = {
            series_key: _finite_quantity_sum(
                initial_quantities[series_key],
                name="initial settlement on_order",
                nonnegative=True,
            )
            for series_key in self._series_keys
        }
        self._inventory_positions: dict[str, InventoryPosition] = {}
        self._frontier: pd.Timestamp | None = None
        self._actuals_semantics: ActualsSemantics | None = None
        self._last_work = SettlementIndexWork()
        self._rebuild_work: SettlementIndexWork | None = None

    @classmethod
    def rebuild(
        cls,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        series_keys: Sequence[str],
        costs_by_series: SessionCosts,
        timing: DecisionTiming,
        stockout_rule: StockoutRule,
        initial_arrivals: Mapping[ActualKey, float] | None,
        orders: Sequence[OrderRow],
        settlements: Sequence[SettlementRecord],
    ) -> SettlementIndex:
        """Reconstruct and audit the compact state in one canonical history pass."""
        rebuilt = cls(
            session=session,
            calendar=calendar,
            series_keys=series_keys,
            costs_by_series=costs_by_series,
            timing=timing,
            stockout_rule=stockout_rule,
            initial_arrivals=initial_arrivals,
        )
        delta = rebuilt.validate_delta(orders=orders, settlements=settlements)
        rebuilt.apply(delta)
        rebuilt._rebuild_work = delta.work
        rebuilt._last_work = SettlementIndexWork()
        return rebuilt

    def snapshot(self, periods: Sequence[pd.Timestamp]) -> SettlementSnapshot:
        """Project only due buckets and constant-per-series frontier state."""
        frozen_periods = self._validated_periods(periods)
        due_arrivals: dict[ActualKey, float] = {}
        origin_order_quantities: dict[ActualKey, float] = {}
        for period in frozen_periods:
            for series_key in self._series_keys:
                key = (series_key, period)
                bucket = self._due_orders.get(key)
                initial = self._initial_due_arrivals.get(key)
                if bucket is not None or initial is not None:
                    due_arrivals[key] = _finite_quantity_sum(
                        (
                            *((initial,) if initial is not None else ()),
                            *(bucket.values() if bucket is not None else ()),
                        ),
                        name="settlement snapshot due arrivals",
                    )
                origin_bucket = self._origin_orders.get(key)
                if origin_bucket is not None:
                    origin_order_quantities[key] = _finite_quantity_sum(
                        origin_bucket.values(),
                        name="settlement snapshot origin-order quantities",
                    )
        current_positions = {
            series_key: InventoryPosition(
                on_hand=position.on_hand,
                on_order=self._open_quantities[series_key],
                backorders=position.backorders,
            )
            for series_key, position in self._inventory_positions.items()
        }
        return SettlementSnapshot(
            session=self._session,
            calendar=self._calendar,
            periods=frozen_periods,
            frontier=self._frontier,
            latest_positions=({} if self._frontier is None else self._inventory_positions),
            open_order_quantities=self._open_quantities,
            due_arrivals=due_arrivals,
            actuals_semantics=self._actuals_semantics,
            origin_order_quantities=origin_order_quantities,
            current_positions=current_positions,
            window_opening_positions=self._inventory_positions,
        )

    @property
    def initial_arrivals(self) -> Mapping[ActualKey, float]:
        """Return the immutable pipeline seed used to construct this index."""
        return MappingProxyType(self._initial_arrival_seed)

    @property
    def has_initial_positions(self) -> bool:
        """Return whether the one-time opening inventory has been registered."""
        return bool(self._inventory_positions)

    def validate_initial_positions(
        self,
        values: Mapping[str, InventoryPosition],
    ) -> dict[str, InventoryPosition]:
        """Validate the one-time opening inventory without changing indexed state."""
        if self._inventory_positions:
            raise LedgerError("initial inventory positions are already registered")
        if not isinstance(values, Mapping) or set(values) != self._series_key_set:
            raise LedgerError("initial inventory must exactly match the decision series set")
        positions: dict[str, InventoryPosition] = {}
        for series_key in self._series_keys:
            position = values[series_key]
            if not isinstance(position, InventoryPosition):
                raise TypeError("initial inventory must contain InventoryPosition values")
            if position.backorders != 0.0:
                raise LedgerError("lost-sales initial inventory requires zero backorders")
            if position.on_order != self._open_quantities[series_key]:
                raise LedgerError("initial inventory on_order does not match seeded arrivals")
            positions[series_key] = position
        return positions

    def apply_initial_positions(
        self,
        positions: Mapping[str, InventoryPosition],
    ) -> None:
        """Register prevalidated one-time opening inventory."""
        self._inventory_positions = dict(positions)

    def validate_delta(
        self,
        *,
        orders: Sequence[OrderRow],
        settlements: Sequence[SettlementRecord],
        initial_positions: Mapping[str, InventoryPosition] | None = None,
    ) -> _SettlementDelta:
        """Validate one append without copying active orders or durable history."""
        staged_orders = self._validated_orders(orders)
        records_by_period = self._validated_records(settlements)
        periods = tuple(sorted(records_by_period))
        if periods:
            self._require_frontier_extension(periods)
            earliest_due = self._earliest_due_period()
            staged_due = min(
                (order.arrival_period for order in staged_orders),
                default=None,
            )
            overdue = min(
                (period for period in (earliest_due, staged_due) if period is not None),
                default=None,
            )
            if overdue is not None and overdue < periods[0]:
                raise LedgerError("open order is overdue before the settlement window")

        by_origin: dict[ActualKey, list[float]] = {}
        by_arrival: dict[ActualKey, list[OrderRow]] = {}
        for order in staged_orders:
            by_origin.setdefault((order.series_key, order.origin), []).append(order.quantity)
            by_arrival.setdefault((order.series_key, order.arrival_period), []).append(order)

        open_quantities = dict(self._open_quantities)
        first_period = periods[0] if periods else None
        for order in staged_orders:
            if first_period is None or order.origin < first_period:
                open_quantities[order.series_key] = _finite_quantity_sum(
                    (open_quantities[order.series_key], order.quantity),
                    name="settlement on_order",
                )

        positions = dict(
            self._inventory_positions if initial_positions is None else initial_positions
        )
        semantics = self._actuals_semantics
        consumed_due_keys: list[ActualKey] = []
        due_order_count = 0
        for period in periods:
            records = records_by_period[period]
            for series_key in self._series_keys:
                record = records[series_key]
                self._validate_record_configuration(record)
                if semantics is None:
                    semantics = record.actuals_semantics
                elif record.actuals_semantics is not semantics:
                    raise LedgerError("settlement actuals semantics changed within the session")

                due_key = (series_key, period)
                durable_due = self._due_orders.get(due_key, {})
                initial_due = self._initial_due_arrivals.get(due_key)
                staged_due_orders = by_arrival.get(due_key, ())
                due_order_count += (
                    len(durable_due)
                    + len(staged_due_orders)
                    + (1 if initial_due is not None else 0)
                )
                arrivals = _finite_quantity_sum(
                    (
                        *((initial_due,) if initial_due is not None else ()),
                        *durable_due.values(),
                        *(order.quantity for order in staged_due_orders),
                    ),
                    name="settlement arrivals",
                )
                if record.arrivals != arrivals:
                    raise LedgerError("settlement arrivals do not match durable due orders")
                if durable_due or initial_due is not None:
                    consumed_due_keys.append(due_key)

                previous_position = positions.get(series_key)
                try:
                    validate_lost_sales_transition(
                        transition=record.transition,
                        arrivals=record.arrivals,
                        opening=previous_position,
                    )
                except LedgerError as error:
                    raise LedgerError(
                        "settlement transition does not continue durable inventory"
                    ) from error

                staged_quantity = _finite_quantity_sum(
                    by_origin.get(due_key, ()),
                    name="staged settlement current-order quantity",
                )
                durable_origin_orders = self._origin_orders.get(due_key, {})
                current_quantity = _finite_quantity_sum(
                    (*durable_origin_orders.values(), staged_quantity),
                    name="settlement current-order quantity",
                )
                indexed_next_on_order = _finite_quantity_sum(
                    (open_quantities[series_key], -arrivals, staged_quantity),
                    name="indexed settlement on_order",
                    nonnegative=True,
                )
                expected_next_on_order = (
                    indexed_next_on_order
                    if previous_position is None
                    else _finite_quantity_sum(
                        (previous_position.on_order, -arrivals, current_quantity),
                        name="settlement on_order",
                        nonnegative=True,
                    )
                )
                if record.inventory_position.on_order != expected_next_on_order:
                    raise LedgerError(
                        "settlement inventory on_order does not match durable open orders"
                    )
                open_quantities[series_key] = indexed_next_on_order
                positions[series_key] = record.inventory_position

        if periods:
            last_period = periods[-1]
            for order in staged_orders:
                if order.origin > last_period:
                    open_quantities[order.series_key] = _finite_quantity_sum(
                        (open_quantities[order.series_key], order.quantity),
                        name="settlement on_order",
                    )
        frontier = self._frontier if not periods else periods[-1]
        added_orders = tuple(
            order for order in staged_orders if frontier is None or order.arrival_period > frontier
        )
        return _SettlementDelta(
            added_orders=added_orders,
            consumed_due_keys=tuple(consumed_due_keys),
            open_quantities=open_quantities,
            inventory_positions=positions,
            frontier=frontier,
            actuals_semantics=semantics,
            work=SettlementIndexWork(
                new_orders=len(staged_orders),
                settlement_records=sum(len(records) for records in records_by_period.values()),
                due_orders=due_order_count,
            ),
        )

    def apply(self, delta: _SettlementDelta) -> None:
        """Apply a validated delta using only assignments and idempotent removals."""
        for due_key in delta.consumed_due_keys:
            self._initial_due_arrivals.pop(due_key, None)
            bucket = self._due_orders.pop(due_key, {})
            for order_key in bucket:
                self._active_orders.discard(order_key)
                _session, series_key, origin, _model_name = order_key
                origin_key = (series_key, origin)
                origin_bucket = self._origin_orders[origin_key]
                del origin_bucket[order_key]
                if not origin_bucket:
                    del self._origin_orders[origin_key]
        for order in delta.added_orders:
            self._active_orders.add(order.key)
            due_key = (order.series_key, order.arrival_period)
            bucket = self._due_orders.get(due_key)
            if bucket is None:
                bucket = {}
                self._due_orders[due_key] = bucket
                heapq.heappush(self._due_heap, (order.arrival_period, order.series_key))
            bucket[order.key] = order.quantity
            self._origin_orders.setdefault((order.series_key, order.origin), {})[order.key] = (
                order.quantity
            )
        self._open_quantities = delta.open_quantities
        self._inventory_positions = delta.inventory_positions
        self._frontier = delta.frontier
        self._actuals_semantics = delta.actuals_semantics
        self._last_work = delta.work

    def audit(self) -> SettlementIndexAudit:
        """Return deterministic complexity evidence without exposing indexed rows."""
        return SettlementIndexAudit(
            active_orders=len(self._active_orders),
            due_buckets=len(set(self._due_orders) | set(self._initial_due_arrivals)),
            last_work=self._last_work,
            rebuild_work=self._rebuild_work,
        )

    def _validated_periods(
        self,
        periods: Sequence[pd.Timestamp],
    ) -> tuple[pd.Timestamp, ...]:
        if isinstance(periods, (str, bytes)):
            raise TypeError("settlement snapshot periods must be a sequence")
        frozen = tuple(periods)
        if not frozen:
            raise LedgerError("settlement snapshot periods must not be empty")
        for index, period in enumerate(frozen):
            try:
                self._calendar.require_member(period, name="settlement snapshot period")
            except (TypeError, ValueError) as error:
                raise LedgerError(str(error)) from error
            if index and period != self._calendar.advance(frozen[index - 1], 1):
                raise LedgerError("settlement snapshot periods must be calendar-contiguous")
        return frozen

    def _validated_orders(self, values: Sequence[OrderRow]) -> tuple[OrderRow, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError("settlement index orders must be a sequence")
        orders = tuple(values)
        keys: set[OrderKey] = set()
        for order in orders:
            if not isinstance(order, OrderRow):
                raise TypeError("settlement index orders must contain OrderRow values")
            if order.session != self._session:
                raise LedgerError("order session does not match the settlement index")
            if order.series_key not in self._series_key_set:
                raise LedgerError("order series does not belong to the session")
            if order.key in keys or order.key in self._active_orders:
                raise LedgerError(f"duplicate order key: {order.key!r}")
            keys.add(order.key)
            expected_arrival = self._calendar.advance(order.origin, self._timing.lead_time)
            if order.arrival_period != expected_arrival:
                raise LedgerError(
                    "order arrival period must equal calendar.advance(origin, lead_time)"
                )
            if self._frontier is not None and order.origin <= self._frontier:
                raise LedgerError("order origin is already settled")
            if self._frontier is not None and order.arrival_period <= self._frontier:
                raise LedgerError("order arrival period is already settled")
        return orders

    def _validated_records(
        self,
        values: Sequence[SettlementRecord],
    ) -> dict[pd.Timestamp, dict[str, SettlementRecord]]:
        if isinstance(values, (str, bytes)):
            raise TypeError("settlement index records must be a sequence")
        records_by_period: dict[pd.Timestamp, dict[str, SettlementRecord]] = {}
        for record in values:
            if not isinstance(record, SettlementRecord):
                raise TypeError("settlement index records must contain SettlementRecord values")
            if record.session != self._session:
                raise LedgerError("settlement session does not match the settlement index")
            if record.series_key not in self._series_key_set:
                raise LedgerError("settlement series does not belong to the session")
            self._calendar.require_member(record.period, name="settlement period")
            records = records_by_period.setdefault(record.period, {})
            if record.series_key in records:
                raise LedgerError(f"duplicate settlement key: {record.key!r}")
            records[record.series_key] = record
        for records in records_by_period.values():
            if set(records) != self._series_key_set:
                raise LedgerError("settlement periods must cover the complete decision series set")
        return records_by_period

    def _require_frontier_extension(self, periods: tuple[pd.Timestamp, ...]) -> None:
        if self._frontier is not None and periods[0] != self._calendar.advance(
            self._frontier,
            1,
        ):
            raise LedgerError("settlement periods must extend the durable frontier")
        if any(
            period != self._calendar.advance(previous, 1) for previous, period in pairwise(periods)
        ):
            raise LedgerError("settlement periods must extend the durable frontier")

    def _validate_record_configuration(self, record: SettlementRecord) -> None:
        if record.transition.rule is not self._stockout_rule:
            raise LedgerError("settlement stock-out rule does not match the session")
        if (
            record.holding.rate != self._costs_by_series[record.series_key].holding
            or record.shortage.rate != self._costs_by_series[record.series_key].shortage
        ):
            raise LedgerError("settlement cost rates do not match the session")

    def _earliest_due_period(self) -> pd.Timestamp | None:
        while self._due_heap:
            period, series_key = self._due_heap[0]
            if (series_key, period) in self._due_orders or (
                series_key,
                period,
            ) in self._initial_due_arrivals:
                return period
            heapq.heappop(self._due_heap)
        return None


def _finite_quantity_sum(
    values: Iterable[float],
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise LedgerError(f"{name} must be finite") from error
    if not math.isfinite(total):
        raise LedgerError(f"{name} must be finite")
    if nonnegative and total < 0.0:
        raise LedgerError(f"{name} must be non-negative")
    return 0.0 if total == 0.0 else total


def _validated_initial_arrivals(
    value: Mapping[ActualKey, float] | None,
    *,
    calendar: Calendar,
    series_keys: set[str],
) -> dict[ActualKey, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("initial arrivals must be a mapping")
    normalized: dict[ActualKey, float] = {}
    for key, quantity in value.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise LedgerError("initial arrival keys must be (series_key, period) tuples")
        series_key, period = key
        if not isinstance(series_key, str) or series_key not in series_keys:
            raise LedgerError("initial arrival series does not belong to the session")
        try:
            calendar.require_member(period, name="initial arrival period")
        except (TypeError, ValueError) as error:
            raise LedgerError(str(error)) from error
        if isinstance(quantity, bool) or not isinstance(quantity, Real):
            raise LedgerError("initial arrival quantity must be a finite non-negative real")
        normalized_quantity = float(quantity)
        if not math.isfinite(normalized_quantity):
            raise LedgerError("initial arrival quantity must be finite")
        if normalized_quantity < 0.0:
            raise LedgerError("initial arrival quantity must be non-negative")
        if normalized_quantity != 0.0:
            normalized[(series_key, period)] = normalized_quantity
    return normalized


def _costs_by_series(
    value: SessionCosts,
    *,
    series_keys: tuple[str, ...],
) -> dict[str, CostStructure]:
    if not isinstance(value, Mapping) or set(value) != set(series_keys):
        raise LedgerError("settlement costs must exactly match the decision series set")
    if any(not isinstance(cost, CostStructure) for cost in value.values()):
        raise LedgerError("settlement costs must contain CostStructure values")
    return {series_key: value[series_key] for series_key in series_keys}


__all__ = ["SettlementIndex", "SettlementIndexAudit", "SettlementIndexWork"]
