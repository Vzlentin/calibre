import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID,
    DS,
    Y,
    Y_HAT,
    H,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    validate_forecast_frame,
)


def _make_valid_frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"] * n,
            DS: pd.date_range("2024-01-07", periods=n, freq="W"),
            Y: np.nan,
            Y_HAT: [10.0, 20.0, 30.0][:n],
            H: list(range(1, n + 1)),
            FORECAST_ORIGIN: pd.Timestamp("2024-01-01"),
            MODEL_NAME: ["SeasonalNaive"] * n,
        }
    )


def test_valid_frame_passes():
    df = _make_valid_frame()
    validate_forecast_frame(df)


def test_missing_column_raises():
    df = _make_valid_frame().drop(columns=[Y_HAT])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_forecast_frame(df)


def test_wrong_dtype_raises():
    df = _make_valid_frame()
    df[H] = df[H].astype(float)
    with pytest.raises(ValueError, match="Column 'h'"):
        validate_forecast_frame(df)


def test_y_allows_nan():
    df = _make_valid_frame()
    df[Y] = np.nan
    validate_forecast_frame(df)


def test_constants_are_strings():
    assert UNIQUE_ID == "unique_id"
    assert DS == "ds"
    assert Y == "y"
    assert Y_HAT == "y_hat"
    assert H == "h"
    assert FORECAST_ORIGIN == "forecast_origin"
    assert MODEL_NAME == "model_name"
    assert len(REQUIRED_COLUMNS) == 7
