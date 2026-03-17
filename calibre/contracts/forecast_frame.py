from __future__ import annotations

import pandas as pd

UNIQUE_ID = "unique_id"
DS = "ds"
Y = "y"
Y_HAT = "y_hat"
H = "h"
FORECAST_ORIGIN = "forecast_origin"
MODEL_NAME = "model_name"

REQUIRED_COLUMNS = [UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]

_EXPECTED_DTYPES = {
    UNIQUE_ID: "object",
    DS: "datetime64[ns]",
    Y: "float64",
    Y_HAT: "float64",
    H: "int64",
    FORECAST_ORIGIN: "datetime64[ns]",
    MODEL_NAME: "object",
}


def validate_forecast_frame(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the forecast-frame contract.

    Raises ValueError if validation fails.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col, expected in _EXPECTED_DTYPES.items():
        actual = str(df[col].dtype)
        if expected == "datetime64[ns]":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise ValueError(f"Column '{col}' expected datetime64, got {actual}")
        elif actual != expected:
            raise ValueError(f"Column '{col}' expected {expected}, got {actual}")
