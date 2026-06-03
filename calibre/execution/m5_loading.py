"""Loaders for the M5 hierarchical retail dataset."""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y


def _m5_unique_id(frame: pd.DataFrame) -> pd.Series:
    return frame["item_id"].astype(str).str.cat(frame["store_id"].astype(str), sep="_")


def _day_columns(columns: list[str]) -> list[str]:
    return sorted(
        [col for col in columns if str(col).startswith("d_")],
        key=lambda col: int(str(col).split("_", 1)[1]),
    )


def melt_m5_sales(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide M5 sales into long-format ``[unique_id, ds, y]``.

    ``sales`` and ``calendar`` are already-read frames so the (large) sales file
    is parsed once by the adapter and shared with :func:`build_m5_hierarchy`.
    """
    day_cols = _day_columns(list(sales.columns))
    if not day_cols:
        raise ValueError("M5 sales frame has no d_* day columns")

    if "d" not in calendar.columns or "date" not in calendar.columns:
        raise ValueError("M5 calendar frame missing required 'd'/'date' columns")
    day_to_date = dict(
        zip(
            calendar["d"].astype(str),
            pd.to_datetime(calendar["date"]),
            strict=True,
        )
    )

    melted = sales[day_cols].copy()
    melted.insert(0, UNIQUE_ID, _m5_unique_id(sales))

    long = melted.melt(id_vars=[UNIQUE_ID], var_name="d", value_name=Y)
    long["d"] = long["d"].astype(str)
    long[DS] = long["d"].map(day_to_date)
    if long[DS].isna().any():
        missing = sorted(long.loc[long[DS].isna(), "d"].unique())
        raise ValueError(f"calendar missing dates for day columns: {missing}")
    long[Y] = long[Y].astype("float64")
    long = long.drop(columns="d")
    return long[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)


def build_m5_hierarchy(sales: pd.DataFrame) -> pd.DataFrame:
    """Return one product/location attribute row per bottom-level M5 series."""
    attr_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing = [col for col in attr_cols if col not in sales.columns]
    if missing:
        raise ValueError(f"M5 sales frame missing hierarchy columns: {missing}")

    frame = sales[attr_cols].copy()
    frame[UNIQUE_ID] = _m5_unique_id(sales)
    frame = frame.drop_duplicates(UNIQUE_ID).reset_index(drop=True)
    return frame[[UNIQUE_ID, *attr_cols]]
