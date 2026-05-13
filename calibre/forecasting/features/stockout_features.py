"""Censored-demand handling for stockout periods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import DS, IN_STOCK, UNIQUE_ID, Y
from calibre.forecasting.features.panel import _sort_panel


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

    df["y_uncensored"] = df[Y].copy()
    df = _sort_panel(df)

    for _uid, group in df.groupby(UNIQUE_ID, sort=False):
        in_stock_mask = group[IN_STOCK]
        oos_mask = ~in_stock_mask

        if not oos_mask.any():
            continue

        in_stock_sales = group[Y].where(in_stock_mask)
        rolling_demand = in_stock_sales.expanding(min_periods=1).median().ffill()

        # Observed sales during OOS are a lower bound on true demand.
        # ``fmax`` falls back to the non-NaN value when the rolling estimate
        # is undefined (e.g. leading OOS rows with no in-stock history yet).
        idx = group.index[oos_mask]
        observed = df[Y].loc[idx].to_numpy()
        imputed = rolling_demand.loc[idx].to_numpy()
        df.loc[idx, "y_uncensored"] = np.fmax(observed, imputed)

    return df
