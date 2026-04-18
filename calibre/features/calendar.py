"""Calendar features derived from the date column."""

from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import DS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``week_of_year``, ``month``, ``quarter`` columns from ``ds``."""
    ds = pd.to_datetime(df[DS])
    df["week_of_year"] = ds.dt.isocalendar().week.astype(int)
    df["month"] = ds.dt.month
    df["quarter"] = ds.dt.quarter
    return df
