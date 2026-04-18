"""Composer that chains the standard feature-engineering steps."""

from __future__ import annotations

import pandas as pd

from calibre.features.calendar import add_calendar_features
from calibre.features.censoring import add_stockout_features
from calibre.features.lags import add_lag_features, add_rolling_features
from calibre.features.scaling import add_series_scaling
from calibre.features.static import add_static_features
from calibre.features.weights import add_time_weights


def build_training_frame(
    sales: pd.DataFrame,
    instock: pd.DataFrame | None = None,
    static: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    half_life_weeks: int = 52,
) -> pd.DataFrame:
    """Compose the standard feature-engineering chain for a global model.

    Returns a DataFrame ready for tabular ML with both ``y_uncensored`` and
    ``y_scaled`` targets, lag/rolling/calendar/static features, and a
    ``sample_weight`` column.
    """
    df = add_stockout_features(sales, instock)
    df = add_series_scaling(df)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_calendar_features(df)
    df = add_static_features(df, static)
    df = add_time_weights(df, half_life_weeks=half_life_weeks)

    return df
