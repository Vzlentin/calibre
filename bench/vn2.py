"""VN2 inventory planning benchmark: 599 store-product series, weekly, six decisions.

The rules: a two-week lead time and a one-week review, so each order protects three
weeks of demand. Holding costs 0.2 per unit left at the end of a week, shortage 1.0
per unit of lost sales. Six weekly orders, then eight settled weeks.

`read` gives the raw arrays of the VN2 release from a directory, and `play` turns one
upper bound per decision into orders and settles them. This file imports no `calibre`:
it loads the frozen decision files by path, so the gate checker can compute the cost in a
process where `calibre` never runs. The external results quoted for VN2 are self-reported
and their settlement is not reconciled with this one, so compare methods with each other
here, not with those figures.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOLDING = 0.2
SHORTAGE = 1.0
LEAD_TIME = 2
PROTECTION = LEAD_TIME + 1
DECISIONS = 6
SETTLED_WEEKS = DECISIONS + LEAD_TIME
FIRST_SETTLED_WEEK = pd.Timestamp("2024-04-15")
FREQ = "W-MON"
LEVELS = ["Store", "ProductGroup", "Division", "Department", "DepartmentGroup"]
DECISION_DIR = Path(__file__).resolve().parents[1] / "src" / "calibre" / "decision"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"vn2_decision_{name}", DECISION_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}.py from {DECISION_DIR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _load("policy")
lost_sales = _load("lost_sales")


@dataclass(frozen=True)
class VN2:
    """The benchmark data, as 0.1 read it.

    `values` is weekly sales `[B, T]` over `weeks`, through the last revealed week. A
    product that is not yet listed sold zero: those zeros are data. `censored` is True
    where the product was out of stock, known only for the weeks before the first
    decision. `decisions` are the six week indexes, each the last observed week before an
    order. `on_hand` and `in_transit` `[B, 2]` are the stock at the first decision.
    """

    series: np.ndarray
    weeks: pd.DatetimeIndex
    values: np.ndarray
    censored: np.ndarray
    attributes: pd.DataFrame
    decisions: np.ndarray
    on_hand: np.ndarray
    in_transit: np.ndarray

    def demand(self) -> np.ndarray:
        """Sales of the eight settled weeks `[B, 8]`, used as demand."""
        first = int(self.decisions[0]) + 1
        return self.values[:, first : first + SETTLED_WEEKS]


@dataclass(frozen=True)
class Played:
    """Settled weeks of one policy, with costs."""

    orders: np.ndarray
    holding_cost: float
    shortage_cost: float

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost


def read(path: str | Path) -> VN2:
    """Read `week_8_sales.csv`, `week_0_in_stock.csv`, `week_0_master.csv`, and
    `week_0_initial_state.csv` from `path`.
    """
    path = Path(path)
    sales = pd.read_csv(path / "week_8_sales.csv")
    series = _series(sales)
    weeks = pd.DatetimeIndex(pd.to_datetime(sales.columns[2:]))
    values = sales.iloc[:, 2:].fillna(0).to_numpy(dtype=np.float32)

    in_stock = pd.read_csv(path / "week_0_in_stock.csv")
    in_stock.index = _series(in_stock)
    stocked = in_stock.iloc[:, 2:]
    stocked.columns = pd.to_datetime(stocked.columns)
    # Weeks after the in-stock file are unknown and count as in stock.
    stocked = stocked.reindex(index=series, columns=weeks).fillna(True).astype(bool)

    master = pd.read_csv(path / "week_0_master.csv")
    master.index = _series(master)
    state = pd.read_csv(path / "week_0_initial_state.csv")
    state.index = _series(state)
    state = state.loc[series]
    first = weeks.get_loc(FIRST_SETTLED_WEEK) - 1
    return VN2(
        series=series,
        weeks=weeks,
        values=values,
        censored=~stocked.to_numpy(),
        attributes=master.loc[series].drop(columns=["Product"]),
        decisions=np.arange(first, first + DECISIONS),
        on_hand=state["End Inventory"].to_numpy(dtype=np.float32),
        in_transit=state[["In Transit W+1", "In Transit W+2"]].to_numpy(dtype=np.float32),
    )


def play(data: VN2, bounds: np.ndarray) -> Played:
    """Order up to `bounds` `[6, B]` at each decision and settle the eight weeks.

    Each bound is an upper bound on the demand of the three protected weeks after its
    decision.
    """
    n_series = len(data.series)
    if bounds.shape != (DECISIONS, n_series):
        raise ValueError(f"bounds shape {bounds.shape} != {(DECISIONS, n_series)}")
    if not np.isfinite(bounds).all():
        raise ValueError("bounds must be finite: an infinite bound means not ready")
    demand = data.demand()
    on_hand, in_transit = data.on_hand.copy(), data.in_transit.copy()
    orders = np.zeros((DECISIONS, n_series), dtype=np.float32)
    holding = shortage = 0.0
    for week in range(SETTLED_WEEKS):
        order = np.zeros(n_series, dtype=np.float32)
        if week < DECISIONS:
            order = orders[week] = policy.order_up_to(bounds[week], on_hand, in_transit)
        on_hand, in_transit, settled = lost_sales.settle(
            on_hand, in_transit, demand[:, week], order, HOLDING, SHORTAGE
        )
        holding += float(settled.holding_cost.sum())
        shortage += float(settled.shortage_cost.sum())
    return Played(orders, holding, shortage)


def _series(frame: pd.DataFrame) -> np.ndarray:
    return (frame["Store"].astype(str) + "_" + frame["Product"].astype(str)).to_numpy(dtype=str)
