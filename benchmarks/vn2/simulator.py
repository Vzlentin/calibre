"""VN2 inventory simulator implementing the competition cost rules.

Cost rules:
    - Holding cost: €0.20 per unit of end-of-week inventory (no cost for in-transit)
    - Shortage cost: €1.00 per unit of lost sales (no backorders)
    - Lead time: order placed end of week X → arrives start of week X+3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ProductState:
    """Inventory state for a single product at the start of a simulation period."""

    unique_id: str
    end_inventory: float
    in_transit_w1: float  # arrives at start of next week (week+1)
    in_transit_w2: float  # arrives at start of week+2
    cumulative_holding_cost: float = 0.0
    cumulative_shortage_cost: float = 0.0


@dataclass
class WeekResult:
    """Snapshot of what happened for one product during one simulation week."""

    unique_id: str
    week: int
    start_inventory: float  # end_inventory (prev) + arrivals
    arrivals: float  # in_transit_w1 from previous state
    demand: float
    sales: float  # min(start_inventory, demand)
    missed_sales: float  # demand - sales
    end_inventory: float  # start_inventory - sales
    holding_cost: float  # end_inventory * 0.2
    shortage_cost: float  # missed_sales * 1.0


class VN2Simulator:
    """Week-by-week VN2 inventory simulator.

    Lead-time mechanics:
        - Orders placed at end of week X go into in_transit_w2.
        - At next step (week X+1): in_transit_w2 → in_transit_w1.
        - At step after (week X+2): in_transit_w1 arrives as arrivals → available inventory.
        - So order placed end of week X is available for sales starting week X+3.
    """

    HOLDING_COST_RATE: float = 0.2
    SHORTAGE_COST_RATE: float = 1.0

    def __init__(self, states: dict[str, ProductState]) -> None:
        # Deep-copy states so the caller's originals are unaffected
        self.states: dict[str, ProductState] = {
            uid: ProductState(**vars(s)) for uid, s in states.items()
        }
        self.history: list[WeekResult] = []

    def step(
        self,
        week: int,
        orders: dict[str, float],
        actual_demand: dict[str, float],
    ) -> list[WeekResult]:
        """Advance simulation by one week for all products.

        Lead time: order placed this week → stored as in_transit_w2.
        Next week that shifts to in_transit_w1. The week after that it
        arrives at start of the week (available for sales).

        Args:
            week: Current week number (used for labelling WeekResult rows).
            orders: Dict[unique_id, order_quantity] — quantities ordered this week.
            actual_demand: Dict[unique_id, demand] — realised demand this week.

        Returns:
            List of WeekResult, one per product.
        """
        week_results: list[WeekResult] = []

        for uid, state in self.states.items():
            # Arrivals come from the pipeline that was loaded in_transit_w1
            arrivals = state.in_transit_w1
            start_inventory = state.end_inventory + arrivals

            demand = actual_demand.get(uid, 0.0)
            sales = min(start_inventory, demand)
            missed_sales = demand - sales
            end_inventory = start_inventory - sales

            holding_cost = end_inventory * self.HOLDING_COST_RATE
            shortage_cost = missed_sales * self.SHORTAGE_COST_RATE

            result = WeekResult(
                unique_id=uid,
                week=week,
                start_inventory=start_inventory,
                arrivals=arrivals,
                demand=demand,
                sales=sales,
                missed_sales=missed_sales,
                end_inventory=end_inventory,
                holding_cost=holding_cost,
                shortage_cost=shortage_cost,
            )
            week_results.append(result)
            self.history.append(result)

            # Update state for next week
            # Pipeline: new order → in_transit_w2
            #           old in_transit_w2 → in_transit_w1
            #           old in_transit_w1 became arrivals (consumed above)
            state.in_transit_w1 = state.in_transit_w2
            state.in_transit_w2 = float(orders.get(uid, 0.0))
            state.end_inventory = end_inventory
            state.cumulative_holding_cost += holding_cost
            state.cumulative_shortage_cost += shortage_cost

        return week_results

    def total_cost(self) -> float:
        """Sum of cumulative holding + shortage cost across all products."""
        return sum(
            s.cumulative_holding_cost + s.cumulative_shortage_cost for s in self.states.values()
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert simulation history to a DataFrame, one row per (product, week)."""
        return pd.DataFrame([vars(r) for r in self.history])


def load_initial_states(initial_state_path: str | Path) -> dict[str, ProductState]:
    """Read week_0_initial_state.csv and return a dict of ProductState keyed by unique_id.

    The unique_id is constructed as f"{Store}_{Product}" (both integers).

    Expected CSV columns:
        Store, Product, End Inventory, In Transit W+1, In Transit W+2
    """
    df = pd.read_csv(initial_state_path)

    states: dict[str, ProductState] = {}
    for _, row in df.iterrows():
        uid = f"{int(row['Store'])}_{int(row['Product'])}"
        state = ProductState(
            unique_id=uid,
            end_inventory=float(row["End Inventory"]),
            in_transit_w1=float(row["In Transit W+1"]),
            in_transit_w2=float(row["In Transit W+2"]),
        )
        states[uid] = state

    return states


def extract_new_actuals(data_dir: str | Path, week: int) -> dict[str, float]:
    """Extract actual demand for week N by comparing progressive sales CSVs.

    Loads week_{week}_sales.csv and week_{week-1}_sales.csv. The new column
    (present in week_N but not in week_{N-1}) contains the week's sales.

    Args:
        data_dir: Directory containing the weekly sales CSV files.
        week: The week number to extract actuals for (must be >= 1).

    Returns:
        Dict mapping unique_id (f"{Store}_{Product}") → demand for that week.

    Raises:
        ValueError: If no new column is found or multiple new columns are found.
        FileNotFoundError: If either sales CSV is missing.
    """
    data_dir = Path(data_dir)

    current_path = data_dir / f"week_{week}_sales.csv"
    previous_path = data_dir / f"week_{week - 1}_sales.csv"

    current_df = pd.read_csv(current_path)
    previous_df = pd.read_csv(previous_path)

    current_cols = set(current_df.columns)
    previous_cols = set(previous_df.columns)
    new_cols = current_cols - previous_cols

    if len(new_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 new column in week_{week} vs week_{week - 1}, "
            f"found {len(new_cols)}: {new_cols}"
        )

    new_col = next(iter(new_cols))

    # Build unique_id from Store/Product columns
    unique_ids = (
        current_df["Store"].astype(int).astype(str)
        + "_"
        + current_df["Product"].astype(int).astype(str)
    )

    demand_series = current_df[new_col].fillna(0.0).astype(float)

    return dict(zip(unique_ids, demand_series, strict=False))
