"""M5 forecasting benchmark: 30,490 Walmart item-store series, daily, 12 levels.

`load` reads `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv` from
a directory. With the default levels the hierarchy has the 42,840 standard series.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibre.forecast import Covariate
from calibre.hierarchy import Hierarchy
from calibre.panel import Panel

LEVELS = [
    "state_id",
    "store_id",
    "cat_id",
    "dept_id",
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    "item_id",
    ["item_id", "state_id"],
]
ATTRIBUTES = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass(frozen=True)
class M5:
    """Daily unit sales, the hierarchy, and the calendar of the full M5 period.

    `calendar` has one row per day of the release, including the 28 days after the
    last sales day.
    """

    panel: Panel
    hierarchy: Hierarchy
    attributes: pd.DataFrame
    calendar: pd.DataFrame
    path: Path

    def prices(self, horizon: int) -> Covariate:
        """Daily sell price `[B, T + horizon]`, known ahead, mean over aggregate nodes.

        A price before the first listed week takes the first listed price, so the
        values stay finite.
        """
        n_days = len(self.panel.periods) + horizon
        if n_days > len(self.calendar):
            raise ValueError(f"the calendar has {len(self.calendar)} days, needs {n_days}")
        weeks = self.calendar["wm_yr_wk"].to_numpy()[:n_days]
        prices = pd.read_csv(self.path / "sell_prices.csv")
        wide = prices.pivot_table(
            index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price"
        )
        keys = pd.MultiIndex.from_frame(self.attributes[["store_id", "item_id"]])
        unique_weeks, column = np.unique(weeks, return_inverse=True)
        daily = wide.reindex(index=keys, columns=unique_weeks).to_numpy(np.float32)[:, column]
        daily = pd.DataFrame(daily).bfill(axis=1).ffill(axis=1).to_numpy(np.float32)
        return Covariate(daily, known_ahead=True, aggregate="mean")


def load(path: str | Path, levels: list | None = None) -> M5:
    """Read the M5 release from `path`. `levels` go to `Hierarchy.from_attributes`."""
    path = Path(path)
    sales = pd.read_csv(path / "sales_train_evaluation.csv")
    calendar = pd.read_csv(path / "calendar.csv", parse_dates=["date"])
    days = [column for column in sales.columns if column.startswith("d_")]
    series = (sales["item_id"] + "_" + sales["store_id"]).to_numpy()
    panel = Panel(series, calendar["date"][: len(days)], sales[days].to_numpy(np.float32), "D")
    attributes = sales[ATTRIBUTES].set_index(series)
    hierarchy = Hierarchy.from_attributes(series, attributes, levels or LEVELS)
    return M5(panel, hierarchy, attributes, calendar, path)
