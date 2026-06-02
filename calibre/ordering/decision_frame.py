from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    interval_column_names,
)


def decision_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in (UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME) if column in frame.columns
    ]


def validate_interval_columns(frame: pd.DataFrame, coverage: float) -> tuple[str, str]:
    lower_col, upper_col = interval_column_names(coverage)
    missing_columns = [column for column in (lower_col, upper_col) if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Missing conformal interval columns for coverage {coverage}: {missing_columns}"
        )
    return lower_col, upper_col
