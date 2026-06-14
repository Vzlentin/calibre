"""Partition-key functions that group rows for per-partition calibration."""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID

GLOBAL_PARTITION = "__global__"


def global_partition(row: pd.Series) -> str:
    """Map every row to the single shared global partition."""
    del row
    return GLOBAL_PARTITION


def series_partition(row: pd.Series) -> Hashable:
    """Partition each row by its ``unique_id`` (one partition per series)."""
    return row[UNIQUE_ID]


def category_partition(col: str):
    """Build a partition-key function that groups rows by ``row[col]``."""

    def _partition(row: pd.Series) -> Hashable:
        return row[col]

    return _partition


def regime_partition(row: pd.Series) -> str:
    """Partition each row by its ``regime_id``, falling back to global."""
    return str(row.get("regime_id", GLOBAL_PARTITION))
