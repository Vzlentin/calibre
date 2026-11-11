"""M5 forecasting benchmark: 30,490 Walmart item-store series, daily, 12 levels.

`read` gives the raw arrays of `sales_train_evaluation.csv` and `calendar.csv` from a
directory. It imports no `calibre`, so the gate checker can read the same data. Each
bottom series runs over all 1,941 days, as 0.1 read it. `LEVELS` with
`Hierarchy.from_attributes` gives the 42,840 standard series.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Daily unit sales of the bottom series.

    `series` is `item_id` and `store_id` joined by `_`, in file order. `values` is
    `[B, T]` over `dates`. `attributes` is indexed by series, with the columns of
    `ATTRIBUTES`.
    """

    series: np.ndarray
    dates: pd.DatetimeIndex
    values: np.ndarray
    attributes: pd.DataFrame


def read(path: str | Path) -> M5:
    """Read the M5 release from `path`."""
    path = Path(path)
    sales = pd.read_csv(path / "sales_train_evaluation.csv")
    calendar = pd.read_csv(path / "calendar.csv", parse_dates=["date"])
    days = [column for column in sales.columns if column.startswith("d_")]
    series = (sales["item_id"] + "_" + sales["store_id"]).to_numpy(dtype=str)
    return M5(
        series=series,
        dates=pd.DatetimeIndex(calendar["date"][: len(days)]),
        values=sales[days].to_numpy(np.float32),
        attributes=sales[ATTRIBUTES].set_index(series),
    )
