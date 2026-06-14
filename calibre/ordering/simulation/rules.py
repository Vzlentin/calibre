"""Inventory-transition rules for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from calibre.ordering.simulation.results import PeriodResult
from calibre.ordering.simulation.state import ProductState


class InventoryRule(Protocol):
    """Strategy for advancing a single product's inventory state by one period.

    Implementations mutate ``state`` in place and return a ``PeriodResult``
    populated with inventory accounting (sales, end_inventory, ...). Cost
    components are filled later by the cost model.
    """

    def transition(
        self,
        state: ProductState,
        period: int,
        demand: float,
        order: float,
    ) -> PeriodResult: ...


@dataclass(frozen=True, slots=True)
class LostSalesRule:
    """Lost-sales inventory rule.

    Demand exceeding available inventory is permanently lost (no backorders).
    The pipeline shifts by one slot each period: leftmost slot becomes
    arrivals, the new order is appended on the right.
    """

    lead_time_depth: int

    def __post_init__(self) -> None:
        if self.lead_time_depth < 0:
            raise ValueError("lead_time_depth must be non-negative")

    def transition(
        self,
        state: ProductState,
        period: int,
        demand: float,
        order: float,
    ) -> PeriodResult:
        if state.lead_time_depth != self.lead_time_depth:
            raise ValueError(
                f"State pipeline depth {state.lead_time_depth} does not match rule "
                f"lead_time_depth {self.lead_time_depth}"
            )

        arrivals = float(state.pipeline.popleft()) if self.lead_time_depth > 0 else 0.0
        start_inventory = float(state.end_inventory) + arrivals
        demand = float(demand)
        sales = min(start_inventory, demand)
        missed_sales = demand - sales
        end_inventory = start_inventory - sales

        order = float(order)
        if self.lead_time_depth > 0:
            state.pipeline.append(order)
        elif order != 0.0:
            raise ValueError("Cannot place an order when lead_time_depth=0")
        state.end_inventory = end_inventory

        return PeriodResult(
            unique_id=state.unique_id,
            period=int(period),
            start_inventory=start_inventory,
            arrivals=arrivals,
            demand=demand,
            sales=sales,
            missed_sales=missed_sales,
            end_inventory=end_inventory,
            order=order,
        )
