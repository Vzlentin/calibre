"""Independently recompute lost-sales settlement for oracle tests only.

This module intentionally imports no production package. It is the sole
test-only duplicate of settlement arithmetic permitted by KTD-A11.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

type DemandKey = tuple[str, str]


class ReferenceInputError(ValueError):
    """Report an incomplete or invalid closed-form replay input."""


@dataclass(frozen=True, slots=True)
class ReferenceSeries:
    """Declare one decision series' initial state and linear cost rates."""

    series_key: str
    initial_on_hand: float
    holding_rate: float
    shortage_rate: float


@dataclass(frozen=True, slots=True)
class ReferenceOrder:
    """Place one order by period index for arrival after the fixed lead time."""

    series_key: str
    origin_index: int
    quantity: float


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    """Expose every hand-recomputable transition and cost component."""

    series_key: str
    period: str
    opening: float
    arrivals: float
    demand: float
    fulfilled: float
    closing: float
    shortage: float
    holding_cost: float
    shortage_cost: float

    @property
    def total_cost(self) -> float:
        """Return the short two-term closed-form cost."""
        return self.holding_cost + self.shortage_cost


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """Return canonical rows plus per-period and terminal cost totals."""

    rows: tuple[ReferenceRow, ...]
    cost_by_period: Mapping[str, float]
    total_cost: float


def calculate_reference_trajectory(
    *,
    periods: Sequence[str],
    series: Sequence[ReferenceSeries],
    demand: Mapping[DemandKey, float],
    orders: Sequence[ReferenceOrder],
    lead_time: int,
    initial_arrivals: Mapping[DemandKey, float] | None = None,
) -> ReferenceTrajectory:
    """Apply arrive, lost-sale, then linear-cost identities in period order."""
    frozen_periods = tuple(periods)
    if not frozen_periods or len(set(frozen_periods)) != len(frozen_periods):
        raise ReferenceInputError("reference periods must be non-empty and unique")
    if lead_time < 1:
        raise ReferenceInputError("reference lead time must be positive")
    series_by_key: dict[str, ReferenceSeries] = {}
    for item in series:
        if not isinstance(item, ReferenceSeries):
            raise TypeError("reference series must contain ReferenceSeries values")
        if not item.series_key or item.series_key in series_by_key:
            raise ReferenceInputError("reference series keys must be non-empty and unique")
        _nonnegative(item.initial_on_hand, name="initial on-hand")
        _nonnegative(item.holding_rate, name="holding rate")
        _nonnegative(item.shortage_rate, name="shortage rate")
        series_by_key[item.series_key] = item
    if not series_by_key:
        raise ReferenceInputError("reference series must not be empty")
    expected_demand = {
        (series_key, period) for period in frozen_periods for series_key in series_by_key
    }
    if set(demand) != expected_demand:
        missing = sorted(expected_demand - set(demand))
        extra = sorted(set(demand) - expected_demand)
        raise ReferenceInputError(
            f"reference demand keys mismatch: missing={missing!r}, extra={extra!r}"
        )
    normalized_demand = {key: _nonnegative(value, name="demand") for key, value in demand.items()}
    period_indexes = {period: index for index, period in enumerate(frozen_periods)}
    arrivals: dict[tuple[str, int], list[float]] = {}
    if initial_arrivals is not None:
        if len(frozen_periods) < lead_time:
            raise ReferenceInputError(
                "reference trajectory omits an initial-arrival pipeline period"
            )
        expected_arrivals = {
            (series_key, period)
            for period in frozen_periods[:lead_time]
            for series_key in series_by_key
        }
        if set(initial_arrivals) != expected_arrivals:
            missing = sorted(expected_arrivals - set(initial_arrivals))
            extra = sorted(set(initial_arrivals) - expected_arrivals)
            raise ReferenceInputError(
                f"reference initial arrival keys mismatch: missing={missing!r}, extra={extra!r}"
            )
        for (series_key, period), value in initial_arrivals.items():
            quantity = _nonnegative(value, name="initial arrival")
            arrivals.setdefault((series_key, period_indexes[period]), []).append(quantity)
    for order in orders:
        if not isinstance(order, ReferenceOrder):
            raise TypeError("reference orders must contain ReferenceOrder values")
        if order.series_key not in series_by_key:
            raise ReferenceInputError("reference order names an unknown series")
        if isinstance(order.origin_index, bool) or not isinstance(order.origin_index, int):
            raise ReferenceInputError("reference order origin indexes must be integers")
        if order.origin_index < 0 or order.origin_index >= len(frozen_periods):
            raise ReferenceInputError("reference order origin index lies outside the trajectory")
        quantity = _nonnegative(order.quantity, name="order quantity")
        arrival_index = order.origin_index + lead_time
        if arrival_index >= len(frozen_periods):
            raise ReferenceInputError("reference trajectory omits an order-arrival drain period")
        arrivals.setdefault((order.series_key, arrival_index), []).append(quantity)

    positions = {key: item.initial_on_hand for key, item in series_by_key.items()}
    rows: list[ReferenceRow] = []
    costs: dict[str, float] = {}
    for period_index, period in enumerate(frozen_periods):
        period_costs: list[float] = []
        for series_key in sorted(series_by_key, key=str.encode):
            item = series_by_key[series_key]
            opening = positions[series_key]
            arrived = math.fsum(arrivals.get((series_key, period_index), ()))
            realized = normalized_demand[(series_key, period)]
            available = opening + arrived
            fulfilled = min(available, realized)
            closing = available - fulfilled
            shortage = realized - fulfilled
            holding_cost = item.holding_rate * closing
            shortage_cost = item.shortage_rate * shortage
            row = ReferenceRow(
                series_key=series_key,
                period=period,
                opening=opening,
                arrivals=arrived,
                demand=realized,
                fulfilled=fulfilled,
                closing=closing,
                shortage=shortage,
                holding_cost=holding_cost,
                shortage_cost=shortage_cost,
            )
            rows.append(row)
            positions[series_key] = closing
            period_costs.append(row.total_cost)
        costs[period] = math.fsum(period_costs)
    return ReferenceTrajectory(
        rows=tuple(rows),
        cost_by_period=MappingProxyType(dict(costs)),
        total_cost=math.fsum(costs[period] for period in frozen_periods),
    )


def _nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceInputError(f"reference {name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ReferenceInputError(f"reference {name} must be finite and nonnegative")
    return normalized
