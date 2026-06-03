"""Loaders for the M5 hierarchical retail dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y


def _m5_unique_id(frame: pd.DataFrame) -> pd.Series:
    return frame["item_id"].astype(str).str.cat(frame["store_id"].astype(str), sep="_")


def _day_columns(columns: list[str]) -> list[str]:
    return sorted(
        [col for col in columns if str(col).startswith("d_")],
        key=lambda col: int(str(col).split("_", 1)[1]),
    )


def melt_m5_sales(sales_path: str | Path, calendar_path: str | Path) -> pd.DataFrame:
    """Read M5 wide sales, return long-format ``[unique_id, ds, y]``."""
    raw = pd.read_csv(str(sales_path))
    day_cols = _day_columns(list(raw.columns))
    if not day_cols:
        raise ValueError(f"No d_* columns found in {sales_path}")

    calendar = pd.read_csv(str(calendar_path))
    if "d" not in calendar.columns or "date" not in calendar.columns:
        raise ValueError(f"calendar missing d/date columns: {calendar_path}")
    day_to_date = dict(
        zip(
            calendar["d"].astype(str),
            pd.to_datetime(calendar["date"]),
            strict=True,
        )
    )

    id_frame = raw[["item_id", "store_id"]].copy()
    id_frame[UNIQUE_ID] = _m5_unique_id(raw)

    melted = raw[day_cols].copy()
    melted.insert(0, UNIQUE_ID, id_frame[UNIQUE_ID])

    long = melted.melt(id_vars=[UNIQUE_ID], var_name="d", value_name=Y)
    long["d"] = long["d"].astype(str)
    long[DS] = long["d"].map(day_to_date)
    if long[DS].isna().any():
        missing = sorted(long.loc[long[DS].isna(), "d"].unique())
        raise ValueError(f"calendar missing dates for day columns: {missing}")
    long[Y] = pd.to_numeric(long[Y], errors="coerce").astype("float64")
    long = long.drop(columns="d")
    return long[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)


def build_m5_hierarchy(sales_path: str | Path) -> pd.DataFrame:
    """Return one attribute row per bottom-level M5 series."""
    raw = pd.read_csv(str(sales_path))
    attr_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing = [col for col in attr_cols if col not in raw.columns]
    if missing:
        raise ValueError(f"sales file missing hierarchy columns: {missing}")

    frame = raw[attr_cols].copy()
    frame[UNIQUE_ID] = _m5_unique_id(raw)
    frame = frame.drop_duplicates(UNIQUE_ID).reset_index(drop=True)
    return frame[[UNIQUE_ID, *attr_cols]]
