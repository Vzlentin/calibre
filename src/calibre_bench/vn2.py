"""VN2 inventory planning benchmark: 599 store-product series, weekly, six decisions.

The rules: a two-week lead time and a one-week review, so each order protects three
weeks of demand. Holding costs 0.2 per unit left at the end of a week, shortage 1.0
per unit of lost sales. Six weekly orders, then eight settled weeks.

`load` reads the files of the VN2 release from a directory. `play` turns one upper
bound per decision into orders and settles them. The external results quoted for VN2
are self-reported and their settlement is not reconciled with this one, so compare
methods with each other here, not with those figures.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibre.hierarchy import Hierarchy
from calibre.panel import Panel
from calibre_bench.inventory import Week, order_up_to, settle

HOLDING = 0.2
SHORTAGE = 1.0
LEAD_TIME = 2
PROTECTION = LEAD_TIME + 1
DECISIONS = 6
SETTLED_WEEKS = DECISIONS + LEAD_TIME
FIRST_SETTLED_WEEK = pd.Timestamp("2024-04-15")
LEVELS = ["Store", "ProductGroup", "Division", "Department", "DepartmentGroup"]


@dataclass(frozen=True)
class VN2:
    """The benchmark data.

    `panel` holds weekly sales through the last revealed week. Its `censored` mask is
    True where the product was out of stock, known only for the weeks before the first
    decision. `decisions` are the six origins, each the last observed week before an
    order. `on_hand` and `in_transit` `[B, 2]` are the stock at the first decision.
    """

    panel: Panel
    hierarchy: Hierarchy
    attributes: pd.DataFrame
    decisions: np.ndarray
    on_hand: np.ndarray
    in_transit: np.ndarray

    def demand(self) -> np.ndarray:
        """Sales of the eight settled weeks `[B, 8]`, used as demand."""
        first = int(self.decisions[0]) + 1
        return self.panel.values[:, first : first + SETTLED_WEEKS]


@dataclass(frozen=True)
class Played:
    """Settled weeks of one policy, with costs."""

    orders: np.ndarray
    weeks: list[Week]

    @property
    def holding_cost(self) -> float:
        return float(sum(week.holding_cost.sum() for week in self.weeks))

    @property
    def shortage_cost(self) -> float:
        return float(sum(week.shortage_cost.sum() for week in self.weeks))

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost


def load(path: str | Path, levels: list | None = None) -> VN2:
    """Read `week_8_sales.csv`, `week_0_in_stock.csv`, `week_0_master.csv`, and
    `week_0_initial_state.csv` from `path`. `levels` go to `Hierarchy.from_attributes`.
    """
    path = Path(path)
    sales = pd.read_csv(path / "week_8_sales.csv")
    series = _series(sales)
    periods = pd.DatetimeIndex(pd.to_datetime(sales.columns[2:]))
    # A product not yet listed has no sales value: it sold nothing.
    values = sales.iloc[:, 2:].fillna(0).to_numpy(dtype=np.float32)

    in_stock = pd.read_csv(path / "week_0_in_stock.csv")
    in_stock.index = _series(in_stock)
    stocked = in_stock.iloc[:, 2:]
    stocked.columns = pd.to_datetime(stocked.columns)
    # Weeks after the in-stock file are unknown and count as in stock.
    stocked = stocked.reindex(index=series, columns=periods).fillna(True).astype(bool)
    censored = ~stocked.to_numpy()

    master = pd.read_csv(path / "week_0_master.csv")
    master.index = _series(master)
    attributes = master.loc[series].drop(columns=["Product"])
    hierarchy = Hierarchy.from_attributes(series, attributes, levels or LEVELS)

    state = pd.read_csv(path / "week_0_initial_state.csv")
    state.index = _series(state)
    state = state.loc[series]
    first = periods.get_loc(FIRST_SETTLED_WEEK) - 1
    return VN2(
        panel=Panel(series, periods, values, "W-MON", censored),
        hierarchy=hierarchy,
        attributes=attributes,
        decisions=np.arange(first, first + DECISIONS),
        on_hand=state["End Inventory"].to_numpy(dtype=np.float32),
        in_transit=state[["In Transit W+1", "In Transit W+2"]].to_numpy(dtype=np.float32),
    )


def play(data: VN2, bounds: np.ndarray) -> Played:
    """Order up to `bounds` `[6, B]` at each decision and settle the eight weeks.

    Each bound is an upper bound on the demand of the three protected weeks after its
    decision, for example `Calibrated.upper[:, :B, 0]` of a `WindowSum(3)` score.
    """
    n_series = len(data.panel.series)
    if bounds.shape != (DECISIONS, n_series):
        raise ValueError(f"bounds shape {bounds.shape} != {(DECISIONS, n_series)}")
    if not np.isfinite(bounds).all():
        raise ValueError("bounds must be finite: an infinite bound means not ready")
    demand = data.demand()
    on_hand, in_transit = data.on_hand.copy(), data.in_transit.copy()
    orders, weeks = np.zeros((DECISIONS, n_series), dtype=np.float32), []
    for week in range(SETTLED_WEEKS):
        order = np.zeros(n_series, dtype=np.float32)
        if week < DECISIONS:
            order = orders[week] = order_up_to(bounds[week], on_hand, in_transit)
        on_hand, in_transit, settled = settle(
            on_hand, in_transit, demand[:, week], order, HOLDING, SHORTAGE
        )
        weeks.append(settled)
    return Played(orders, weeks)


def _series(frame: pd.DataFrame) -> np.ndarray:
    return (frame["Store"].astype(str) + "_" + frame["Product"].astype(str)).to_numpy()
