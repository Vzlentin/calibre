"""Feature engineering for the VN2 winning approach.

Implements the key feature engineering techniques from the VN2 winner report:
  1. Stockout-aware features — flag censored demand periods via in_stock data
  2. Per-series dynamic scaling — normalize sales by series-level statistics
  3. Lag features & rolling statistics — capture demand patterns
  4. Calendar features — week-of-year seasonality
  5. Static categorical features — store/product metadata from master data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y
from calibre.pipeline.loading import IN_STOCK


def add_stockout_features(
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge in-stock data and create stockout-aware features.

    When a product is out of stock, observed sales are censored (truncated at 0
    or at the remaining inventory). This biases forecasts downward. We add:
      - ``in_stock``: binary availability flag
      - ``y_uncensored``: replaces OOS periods with a local demand estimate
        (rolling median of recent in-stock sales) to reduce censoring bias.
    """
    df = sales.copy()

    if instock is None or instock.empty:
        df[IN_STOCK] = True
        df["y_uncensored"] = df[Y]
        return df

    df = df.merge(instock, on=[UNIQUE_ID, DS], how="left")
    df[IN_STOCK] = df[IN_STOCK].fillna(True)

    # For OOS periods, impute demand from recent in-stock sales
    df["y_uncensored"] = df[Y].copy()
    df = df.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)

    for _uid, group in df.groupby(UNIQUE_ID, sort=False):
        in_stock_mask = group[IN_STOCK]
        oos_mask = ~in_stock_mask

        if not oos_mask.any():
            continue

        # Rolling median of in-stock sales (window=8 weeks) as demand proxy
        in_stock_sales = group[Y].where(in_stock_mask)
        rolling_demand = in_stock_sales.expanding(min_periods=1).median().ffill()

        # Replace OOS observations: use max(observed, estimated) since observed
        # sales during OOS are a lower bound on true demand
        idx = group.index[oos_mask]
        df.loc[idx, "y_uncensored"] = np.maximum(
            df.loc[idx, Y].values,
            rolling_demand.loc[idx].values,
        )

    return df


def add_series_scaling(
    df: pd.DataFrame,
    target_col: str = "y_uncensored",
) -> pd.DataFrame:
    """Add per-series scaling statistics and scaled target.

    Per-series dynamic scaling normalises sales so the global model focuses on
    temporal patterns rather than absolute volume levels. This reduces magnitude
    imbalance across 599 heterogeneous series.

    Adds columns:
      - ``series_mean``: rolling expanding mean of target
      - ``series_std``: rolling expanding std of target (floored at 1.0)
      - ``y_scaled``: (target - series_mean) / series_std
    """
    df = df.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    grouped = df.groupby(UNIQUE_ID, sort=False)[target_col]

    df["series_mean"] = grouped.transform(lambda x: x.expanding(min_periods=1).mean())
    raw_std = grouped.transform(lambda x: x.expanding(min_periods=1).std())
    df["series_std"] = raw_std.clip(lower=1.0).fillna(1.0)
    df["y_scaled"] = (df[target_col] - df["series_mean"]) / df["series_std"]

    return df


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

    df = df.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
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

    Default windows: 4 (monthly), 13 (quarterly), 26 (semi-annual).
    """
    if windows is None:
        windows = [4, 13, 26]

    df = df.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    grouped = df.groupby(UNIQUE_ID, sort=False)[target_col]

    for w in windows:
        _w = w  # bind for lambda closure
        df[f"rolling_mean_{w}"] = grouped.transform(
            lambda x, _w=_w: x.shift(1).rolling(window=_w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grouped.transform(
            lambda x, _w=_w: x.shift(1).rolling(window=_w, min_periods=1).std().fillna(0.0)
        )

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from the date column."""
    ds = pd.to_datetime(df[DS])
    df["week_of_year"] = ds.dt.isocalendar().week.astype(int)
    df["month"] = ds.dt.month
    df["quarter"] = ds.dt.quarter
    return df


def add_master_features(
    df: pd.DataFrame,
    master: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge static categorical features from the master data file.

    Master columns (ProductGroup, Division, Department, etc.) become categorical
    features for the global model, allowing it to learn group-level patterns.
    """
    if master is None or master.empty:
        return df

    cat_cols = [c for c in master.columns if c not in ("Store", "Product", UNIQUE_ID)]
    if not cat_cols:
        return df

    merge_cols = [UNIQUE_ID] + cat_cols
    master_subset = master[merge_cols].drop_duplicates(subset=[UNIQUE_ID])

    df = df.merge(master_subset, on=UNIQUE_ID, how="left")

    # Encode as pandas categoricals for LightGBM native handling
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


def add_time_weights(
    df: pd.DataFrame,
    half_life_weeks: int = 52,
) -> pd.DataFrame:
    """Add time-decayed observation weights.

    More recent observations receive higher weight during training. Uses
    exponential decay with configurable half-life (default: 52 weeks / 1 year).
    """
    df = df.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)

    # Compute weeks from most recent observation per series
    max_ds = df.groupby(UNIQUE_ID, sort=False)[DS].transform("max")
    weeks_ago = (max_ds - df[DS]).dt.days / 7.0

    decay_rate = np.log(2) / half_life_weeks
    df["sample_weight"] = np.exp(-decay_rate * weeks_ago)

    return df


def build_training_frame(
    sales: pd.DataFrame,
    instock: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    half_life_weeks: int = 52,
) -> pd.DataFrame:
    """Build the full feature-engineered training frame for a global model.

    Chains all feature engineering steps in order:
      1. Stockout-aware imputation
      2. Per-series scaling
      3. Lag features
      4. Rolling statistics
      5. Calendar features
      6. Master data (categorical) features
      7. Time-decay weights

    Returns a DataFrame ready for tabular ML with columns:
      - Target: ``y_uncensored`` (original) or ``y_scaled`` (normalised)
      - Features: lag_*, rolling_*, week_of_year, month, quarter, series_mean,
                  series_std, in_stock, plus any master data columns
      - Weight: ``sample_weight``
      - Identifiers: unique_id, ds, y (original sales)
    """
    df = add_stockout_features(sales, instock)
    df = add_series_scaling(df)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_calendar_features(df)
    df = add_master_features(df, master)
    df = add_time_weights(df, half_life_weeks=half_life_weeks)

    return df
