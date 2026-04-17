"""Median ensemble aggregator for multi-model forecast ledgers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    validate_forecast_frame,
)

_GROUP_KEYS = [UNIQUE_ID, FORECAST_ORIGIN, DS, H]


def ensemble_median(
    ledger_df: pd.DataFrame,
    name: str = "ensemble_median",
) -> pd.DataFrame:
    """Aggregate a multi-model forecast ledger by taking the median y_hat.

    Groups by (unique_id, forecast_origin, ds, h) and computes the median
    y_hat across all models present in the ledger.

    Args:
        ledger_df: A valid forecast-frame DataFrame potentially containing
            predictions from multiple models.
        name: The model_name to assign to the resulting ensemble rows.

    Returns:
        A valid forecast-frame DataFrame with model_name = name and y = NaN.
    """
    if ledger_df.empty:
        result = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result[Y] = result[Y].astype("float64")
        result[Y_HAT] = result[Y_HAT].astype("float64")
        result[H] = result[H].astype("int64")
        return result

    validate_forecast_frame(ledger_df)

    grouped = ledger_df.groupby(_GROUP_KEYS, sort=False)[Y_HAT].median().reset_index()

    grouped[MODEL_NAME] = name
    grouped[Y] = np.nan

    # Ensure correct dtypes
    grouped[DS] = pd.to_datetime(grouped[DS])
    grouped[FORECAST_ORIGIN] = pd.to_datetime(grouped[FORECAST_ORIGIN])
    grouped[Y_HAT] = grouped[Y_HAT].astype("float64")
    grouped[Y] = grouped[Y].astype("float64")
    grouped[H] = grouped[H].astype("int64")
    grouped[UNIQUE_ID] = grouped[UNIQUE_ID].astype("object")
    grouped[MODEL_NAME] = grouped[MODEL_NAME].astype("object")

    result = grouped[REQUIRED_COLUMNS].reset_index(drop=True)
    validate_forecast_frame(result)
    return result
