"""FreshRetailNet-50K: normalized daily store-product sales, not physical demand.

Censoring is stock_hour6_22_cnt > 0, the annotated 06:00–22:00 stockout window.
It marks observed sales as lower bounds, not recovered demand. The constructed
hierarchy is not an official scoring hierarchy. No future covariates are supplied.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibre.data import Hierarchy, Panel
from sources import Source

BASE_URL = (
    "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/"
    "resolve/c374e85975c82816c495acfa904cc55ab14a47d3"
)
SOURCES = (
    Source(
        "train.parquet",
        f"{BASE_URL}/data/train.parquet",
        "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    ),
    Source(
        "eval.parquet",
        f"{BASE_URL}/data/eval.parquet",
        "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
    ),
)
ATTRIBUTES = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
LEVELS = [*ATTRIBUTES, ["city_id", "product_id"]]
TRAIN_START = pd.Timestamp("2024-03-28")
TRAIN_END = pd.Timestamp("2024-06-25")
EVAL_START = pd.Timestamp("2024-06-26")
EVAL_END = pd.Timestamp("2024-07-02")


@dataclass(frozen=True)
class FreshRetail:
    """Sales, censoring in panel, static attributes, and original-target mask [B, T]."""

    panel: Panel
    hierarchy: Hierarchy
    attributes: pd.DataFrame
    observed: np.ndarray


def load(path: str | Path) -> FreshRetail:
    """Verify and join the 90 train and 7 eval days without imputation or fitting.

    Needs pyarrow (uv run --with pyarrow). Respect EVAL_START when choosing origins;
    never train on eval to predict train. Read raw Parquet on every call, no cache.
    """
    path = Path(path)
    for source in SOURCES:
        source.verify(path / source.name)
    columns = [*ATTRIBUTES, "dt", "sale_amount", "stock_hour6_22_cnt"]
    frame = pd.concat(
        [
            pd.read_parquet(path / source.name, columns=columns, engine="pyarrow")
            for source in SOURCES
        ],
        ignore_index=True,
    )
    frame["series"] = frame["store_id"].astype(str) + ":" + frame["product_id"].astype(str)
    frame["period"] = pd.to_datetime(frame["dt"])
    attributes = frame[["series", *ATTRIBUTES]].drop_duplicates().set_index("series")
    wide = frame.pivot(index="period", columns="series", values="sale_amount").sort_index()
    frame["censored"] = frame["stock_hour6_22_cnt"] > 0
    censored = frame.pivot(index="period", columns="series", values="censored")
    panel = Panel(
        wide.columns.to_numpy(),
        pd.DatetimeIndex(wide.index),
        wide.to_numpy(np.float32).T,
        "D",
        censored.loc[wide.index, wide.columns].to_numpy(dtype=bool).T,
    )
    attributes = attributes.loc[panel.series]
    hierarchy = Hierarchy.from_attributes(panel.series, attributes, LEVELS)
    return FreshRetail(panel, hierarchy, attributes, np.ones(panel.values.shape, dtype=bool))
