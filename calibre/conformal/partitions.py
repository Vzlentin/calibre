from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from calibre.contracts.forecast_frame import UNIQUE_ID

GLOBAL_PARTITION = "__global__"


def global_partition(row: pd.Series) -> str:
    del row
    return GLOBAL_PARTITION


def series_partition(row: pd.Series) -> Hashable:
    return row[UNIQUE_ID]


def category_partition(col: str):
    def _partition(row: pd.Series) -> Hashable:
        return row[col]

    return _partition


def regime_partition(row: pd.Series) -> str:
    return str(row.get("regime_id", GLOBAL_PARTITION))
