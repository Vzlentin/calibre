"""Per-series scaling features."""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID
from calibre.forecasting.features.panel import sort_panel


def add_series_scaling(
    df: pd.DataFrame,
    target_col: str = "y_uncensored",
) -> pd.DataFrame:
    """Add per-series expanding scaling statistics and a scaled target.

    Per-series dynamic scaling normalises sales so a global model focuses on
    temporal patterns rather than absolute volume levels. This reduces
    magnitude imbalance across heterogeneous series.

    Adds columns:
      - ``series_mean``: expanding mean of ``target_col``
      - ``series_std``: expanding std of ``target_col`` (floored at 1.0)
      - ``y_scaled``: ``(target_col - series_mean) / series_std``
    """
    df = sort_panel(df)
    grouped = df.groupby(UNIQUE_ID, sort=False)[target_col]

    df["series_mean"] = grouped.transform(lambda x: x.expanding(min_periods=1).mean())
    raw_std = grouped.transform(lambda x: x.expanding(min_periods=1).std())
    df["series_std"] = raw_std.clip(lower=1.0).fillna(1.0)
    df["y_scaled"] = (df[target_col] - df["series_mean"]) / df["series_std"]

    return df
