from __future__ import annotations

import pandas as pd

from calibre.simulation.costs import CostModel
from calibre.simulation.results import PeriodResult
from calibre.simulation.rules import InventoryRule
from calibre.simulation.state import ProductState


class Simulator:
    """Generic discrete-period inventory simulator.

    The simulator orchestrates per-product transitions via an ``InventoryRule``
    and per-product cost accounting via a ``CostModel``. State is deep-copied
    on construction so the caller's originals are not mutated.
    """

    def __init__(
        self,
        states: dict[str, ProductState],
        rule: InventoryRule,
        cost_model: CostModel,
    ) -> None:
        self.states: dict[str, ProductState] = {uid: s.copy() for uid, s in states.items()}
        self.rule = rule
        self.cost_model = cost_model
        self.history: list[PeriodResult] = []

    def step(
        self,
        period: int,
        orders: dict[str, float],
        actual_demand: dict[str, float],
    ) -> list[PeriodResult]:
        """Advance the simulation by one period for every product.

        Args:
            period: Period number, used for labelling ``PeriodResult`` rows.
            orders: Mapping ``unique_id -> order_quantity`` for orders placed
                this period (defaults to 0 when missing).
            actual_demand: Mapping ``unique_id -> realised demand`` (defaults
                to 0 when missing).

        Returns:
            One ``PeriodResult`` per product, in iteration order of
            ``self.states``.
        """
        period_results: list[PeriodResult] = []

        for uid, state in self.states.items():
            result = self.rule.transition(
                state=state,
                period=period,
                demand=float(actual_demand.get(uid, 0.0)),
                order=float(orders.get(uid, 0.0)),
            )
            costs = self.cost_model.cost(state, result)
            result.costs = dict(costs)
            for component, value in costs.items():
                state.cumulative_costs[component] = state.cumulative_costs.get(
                    component, 0.0
                ) + float(value)
            self.history.append(result)
            period_results.append(result)

        return period_results

    def total_cost(self) -> float:
        """Sum of all cumulative cost components across all products."""
        return sum(state.total_cost for state in self.states.values())

    def cost_breakdown(self) -> dict[str, float]:
        """Per-component total cost summed across all products."""
        breakdown: dict[str, float] = {}
        for state in self.states.values():
            for component, value in state.cumulative_costs.items():
                breakdown[component] = breakdown.get(component, 0.0) + float(value)
        return breakdown

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the simulation history into a DataFrame with one row per (product, period)."""
        return pd.DataFrame([result.to_record() for result in self.history])
