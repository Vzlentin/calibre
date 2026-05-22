"""VN2 inventory simulator: thin adapter over ``calibre.ordering.simulation``.

Holds VN2-specific concerns (the two-slot in-transit pipeline, the linear
holding/shortage cost rates, the CSV I/O helpers) and delegates the generic
state transitions and cost accounting to ``calibre.ordering.simulation``.

Cost rules:
    - Holding cost: EUR 0.20 per unit of end-of-week inventory (no cost for in-transit)
    - Shortage cost: EUR 1.00 per unit of lost sales (no backorders)
    - Lead time: order placed end of week X arrives start of week X+3
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from calibre.execution.io import join_uri
from calibre.ordering.simulation import (
    LinearCostModel,
    LostSalesRule,
    PeriodResult,
    Simulator,
    make_pipeline,
)
from calibre.ordering.simulation import ProductState as GenericProductState

LEAD_TIME_DEPTH: int = 2
HOLDING_COST_RATE: float = 0.2
SHORTAGE_COST_RATE: float = 1.0


@dataclass
class ProductState:
    """VN2-shaped inventory state (two-slot in-transit pipeline)."""

    unique_id: str
    end_inventory: float
    in_transit_w1: float
    in_transit_w2: float
    cumulative_holding_cost: float = 0.0
    cumulative_shortage_cost: float = 0.0


@dataclass
class WeekResult:
    """VN2-shaped per-week per-product simulation result."""

    unique_id: str
    week: int
    start_inventory: float
    arrivals: float
    demand: float
    sales: float
    missed_sales: float
    end_inventory: float
    holding_cost: float
    shortage_cost: float


def _to_generic(state: ProductState) -> GenericProductState:
    return GenericProductState(
        unique_id=state.unique_id,
        end_inventory=float(state.end_inventory),
        pipeline=make_pipeline(
            [float(state.in_transit_w1), float(state.in_transit_w2)],
            LEAD_TIME_DEPTH,
        ),
        cumulative_costs={
            "holding": float(state.cumulative_holding_cost),
            "shortage": float(state.cumulative_shortage_cost),
        },
    )


def _from_generic(state: GenericProductState) -> ProductState:
    pipeline = list(state.pipeline)
    padded = pipeline + [0.0] * max(0, LEAD_TIME_DEPTH - len(pipeline))
    return ProductState(
        unique_id=state.unique_id,
        end_inventory=float(state.end_inventory),
        in_transit_w1=float(padded[0]) if padded else 0.0,
        in_transit_w2=float(padded[1]) if len(padded) > 1 else 0.0,
        cumulative_holding_cost=float(state.cumulative_costs.get("holding", 0.0)),
        cumulative_shortage_cost=float(state.cumulative_costs.get("shortage", 0.0)),
    )


def _to_week_result(result: PeriodResult) -> WeekResult:
    return WeekResult(
        unique_id=result.unique_id,
        week=result.period,
        start_inventory=result.start_inventory,
        arrivals=result.arrivals,
        demand=result.demand,
        sales=result.sales,
        missed_sales=result.missed_sales,
        end_inventory=result.end_inventory,
        holding_cost=result.costs.get("holding", 0.0),
        shortage_cost=result.costs.get("shortage", 0.0),
    )


class VN2Simulator:
    """VN2-specific facade over the generic ``Simulator``.

    Lead-time mechanics:
        - Orders placed at end of week X go into in_transit_w2.
        - Next step (week X+1): in_transit_w2 shifts to in_transit_w1.
        - Step after (week X+2): in_transit_w1 arrives as available inventory.
        - So an order placed end of week X is available for sales starting week X+3.
    """

    HOLDING_COST_RATE: float = HOLDING_COST_RATE
    SHORTAGE_COST_RATE: float = SHORTAGE_COST_RATE

    def __init__(self, states: Mapping[str, ProductState | GenericProductState]) -> None:
        self.states: dict[str, ProductState] = {}
        generic_states: dict[str, GenericProductState] = {}
        for uid, state in states.items():
            if isinstance(state, GenericProductState):
                generic = state.copy()
                self.states[uid] = _from_generic(generic)
            else:
                self.states[uid] = ProductState(**vars(state))
                generic = _to_generic(state)
            generic_states[uid] = generic
        self._sim = Simulator(
            states=generic_states,
            rule=LostSalesRule(lead_time_depth=LEAD_TIME_DEPTH),
            cost_model=LinearCostModel(
                rates={"holding": HOLDING_COST_RATE, "shortage": SHORTAGE_COST_RATE}
            ),
        )
        self.history: list[WeekResult] = []

    def step(
        self,
        week: int,
        orders: dict[str, float],
        actual_demand: dict[str, float],
    ) -> list[WeekResult]:
        period_results = self._sim.step(week, orders, actual_demand)
        week_results: list[WeekResult] = []
        for result in period_results:
            week_result = _to_week_result(result)
            week_results.append(week_result)
            self.history.append(week_result)
            generic = self._sim.states[result.unique_id]
            user = self.states[result.unique_id]
            user.end_inventory = generic.end_inventory
            user.in_transit_w1, user.in_transit_w2 = generic.pipeline
            user.cumulative_holding_cost = generic.cumulative_costs.get("holding", 0.0)
            user.cumulative_shortage_cost = generic.cumulative_costs.get("shortage", 0.0)
        return week_results

    def total_cost(self) -> float:
        return sum(
            s.cumulative_holding_cost + s.cumulative_shortage_cost for s in self.states.values()
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.history])


def load_initial_states(initial_state_path: str | Path) -> dict[str, ProductState]:
    """Read week_0_initial_state.csv and return ``ProductState`` per unique_id.

    The unique_id is constructed as ``f"{Store}_{Product}"``.

    Expected CSV columns:
        Store, Product, End Inventory, In Transit W+1, In Transit W+2
    """
    df = pd.read_csv(str(initial_state_path))

    states: dict[str, ProductState] = {}
    for _, row in df.iterrows():
        uid = f"{int(row['Store'])}_{int(row['Product'])}"
        states[uid] = ProductState(
            unique_id=uid,
            end_inventory=float(row["End Inventory"]),
            in_transit_w1=float(row["In Transit W+1"]),
            in_transit_w2=float(row["In Transit W+2"]),
        )
    return states


def extract_new_actuals(data_dir: str | Path, week: int) -> dict[str, float]:
    """Extract actual demand for week N by diffing progressive sales CSVs.

    Loads ``week_{week}_sales.csv`` and ``week_{week-1}_sales.csv``. The new
    column (present in week_N but not in week_{N-1}) contains that week's sales.
    """
    current_path = join_uri(data_dir, f"week_{week}_sales.csv")
    previous_path = join_uri(data_dir, f"week_{week - 1}_sales.csv")

    current_df = pd.read_csv(current_path)
    previous_df = pd.read_csv(previous_path)

    new_cols = set(current_df.columns) - set(previous_df.columns)
    if len(new_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 new column in week_{week} vs week_{week - 1}, "
            f"found {len(new_cols)}: {new_cols}"
        )

    new_col = next(iter(new_cols))
    unique_ids = (
        current_df["Store"].astype(int).astype(str)
        + "_"
        + current_df["Product"].astype(int).astype(str)
    )
    demand_series = current_df[new_col].fillna(0.0).astype(float)
    return dict(zip(unique_ids, demand_series, strict=False))
