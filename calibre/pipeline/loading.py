"""Functions for loading and reshaping raw data files into long format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y

IN_STOCK = "in_stock"


def _detect_date_columns(columns: list[str]) -> list[str]:
    """Return column names that can be parsed as dates."""
    date_cols = []
    for col in columns:
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except (ValueError, TypeError):
            pass
    return date_cols


def _make_unique_id(df: pd.DataFrame) -> pd.Series:
    """Return a Series of unique_id strings formatted as '{Store}_{Product}'."""
    return df["Store"].astype(int).astype(str) + "_" + df["Product"].astype(int).astype(str)


def melt_wide_sales(path: str | Path) -> pd.DataFrame:
    """Read a wide-format sales CSV, return long-format DataFrame with columns [unique_id, ds, y].

    - unique_id is f"{Store}_{Product}" (both are integers in the CSV)
    - ds is pd.Timestamp (datetime64[ns]), parsed from column names
    - y is float64
    - Detect date columns by trying pd.to_datetime on each column name
    - Return sorted by unique_id, then ds
    """
    raw = pd.read_csv(path)
    date_cols = _detect_date_columns(list(raw.columns))

    id_df = raw[["Store", "Product"]].copy()
    id_df[UNIQUE_ID] = _make_unique_id(raw)

    melted = raw[date_cols].copy()
    melted.insert(0, UNIQUE_ID, id_df[UNIQUE_ID])

    long = melted.melt(id_vars=[UNIQUE_ID], var_name=DS, value_name=Y)
    long[DS] = pd.to_datetime(long[DS])
    long[Y] = long[Y].astype("float64")

    long = long.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    return long[[UNIQUE_ID, DS, Y]]


def load_master(path: str | Path) -> pd.DataFrame:
    """Read Master CSV, add unique_id column = f"{Store}_{Product}". Return all columns."""
    df = pd.read_csv(path)
    df[UNIQUE_ID] = _make_unique_id(df)
    return df


def melt_wide_instock(path: str | Path) -> pd.DataFrame:
    """Read In Stock CSV, return long-format [unique_id, ds, in_stock].

    - in_stock is boolean
    - Same date column detection as melt_wide_sales
    """
    raw = pd.read_csv(path)
    date_cols = _detect_date_columns(list(raw.columns))

    id_col = _make_unique_id(raw)
    melted = raw[date_cols].copy()
    melted.insert(0, UNIQUE_ID, id_col)

    long = melted.melt(id_vars=[UNIQUE_ID], var_name=DS, value_name=IN_STOCK)
    long[DS] = pd.to_datetime(long[DS])
    long[IN_STOCK] = long[IN_STOCK].astype(bool)

    long = long.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    return long[[UNIQUE_ID, DS, IN_STOCK]]


def load_period(data_dir: str | Path, period: int) -> pd.DataFrame:
    """Find 'week_{period}_sales.csv' in data_dir, return melt_wide_sales result."""
    data_dir = Path(data_dir)
    path = data_dir / f"week_{period}_sales.csv"
    if not path.exists():
        raise FileNotFoundError(f"No Sales file found for period {period} in {data_dir}")
    return melt_wide_sales(path)
