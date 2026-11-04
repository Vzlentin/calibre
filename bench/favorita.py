"""Favorita Store Sales: daily store-family sales, fractional quantities preserved.

The hierarchy is a constructed subset, not an official scoring hierarchy. Future
Kaggle test days are unlabelled and excluded; no future covariates are supplied.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibre.data import Hierarchy, Panel
from imputation import forward_fill
from sources import Source

BASE_URL = (
    "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/"
    "resolve/1ac138dfdfafc520bc29831e73373954b1b7bc05"
)
SOURCES = (
    Source(
        "train.csv",
        f"{BASE_URL}/train.csv",
        "99a1b7f4241821df10fa71f23abdd71c634f71f901e3cab1400c0f0c583f862c",
    ),
    Source(
        "stores.csv",
        f"{BASE_URL}/stores.csv",
        "af503b2bce11d7906d249f81cc0598f10e2addcc9f6c59aa2d95c9f2652c296b",
    ),
)
ATTRIBUTES = ["store_nbr", "family", "state", "city", "type", "cluster"]
LEVELS = ["state", ["state", "city"], "store_nbr", "family", ["state", "family"]]
UNLABELLED_TEST_START = pd.Timestamp("2017-08-16")
UNLABELLED_TEST_END = pd.Timestamp("2017-08-31")


@dataclass(frozen=True)
class Favorita:
    """Sales, static hierarchy attributes, and original-target mask [B, T]."""

    panel: Panel
    hierarchy: Hierarchy
    attributes: pd.DataFrame
    observed: np.ndarray
    warmup_periods_dropped: int


def load(path: str | Path) -> Favorita:
    """Verify raw files and causally forward-fill missing dates, including Christmas.

    Trim leading periods without a first observation in every series; never replace
    missing sales with zero. Exclude False observed cells from calibration/evaluation.
    """
    path = Path(path)
    for source in SOURCES:
        source.verify(path / source.name)
    frame = pd.read_csv(path / "train.csv", usecols=["date", "store_nbr", "family", "sales"])
    frame["series"] = frame["store_nbr"].astype(str) + ":" + frame["family"]
    frame["period"] = pd.to_datetime(frame["date"])
    frame = frame.merge(pd.read_csv(path / "stores.csv"), on="store_nbr", how="left")
    attributes = frame[["series", *ATTRIBUTES]].drop_duplicates().set_index("series")
    wide = frame.pivot(index="period", columns="series", values="sales")
    wide, observed, dropped = forward_fill(wide, "D")
    panel = Panel(
        wide.columns.to_numpy(), pd.DatetimeIndex(wide.index), wide.to_numpy(np.float32).T, "D"
    )
    attributes = attributes.loc[panel.series]
    hierarchy = Hierarchy.from_attributes(panel.series, attributes, LEVELS)
    return Favorita(panel, hierarchy, attributes, observed, dropped)
