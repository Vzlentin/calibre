"""Lag and rolling statistics features."""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID
from calibre.forecasting.features.panel import _sort_panel


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    target_col: str = "y_uncensored",
) -> pd.DataFrame:
    """Add lag features per series.

    Default lags cover recent weeks (1-4), quarterly (13), semi-annual (26),
    and annual (52) seasonality.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 13, 26, 52]

    df = _sort_panel(df)
    grouped = df.groupby(UNIQUE_ID, sort=False)[target_col]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    target_col: str = "y_uncensored",
) -> pd.DataFrame:
    """Add rolling mean and std features per series.

    Uses a 1-step shift so the window is strictly causal at prediction time.
    Default windows: 4 (monthly), 13 (quarterly), 26 (semi-annual).
    """
    if windows is None:
        windows = [4, 13, 26]

    df = _sort_panel(df)
    grouped = df.groupby(UNIQUE_ID, sort=False)[target_col]

    for w in windows:
        df[f"rolling_mean_{w}"] = grouped.transform(
            lambda x, w=w: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grouped.transform(
            lambda x, w=w: x.shift(1).rolling(window=w, min_periods=1).std().fillna(0.0)
        )

    return df
