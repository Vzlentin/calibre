import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    CALIBRATION_STATE,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
    lower_interval_column,
    quantile_column,
    upper_interval_column,
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


def test_interval_column_helpers_use_coverage_suffix():
    assert lower_interval_column(0.9) == "lo_0p9"
    assert upper_interval_column(0.95) == "hi_0p95"
    assert interval_column_names(0.9) == ("lo_0p9", "hi_0p9")


def test_conformal_optional_columns_validate_when_present():
    lower_col, upper_col = interval_column_names(0.9)
    df = _make_valid_frame()
    df[lower_col] = np.array([9.0, 19.0, 29.0], dtype=float)
    df[upper_col] = np.array([11.0, 21.0, 31.0], dtype=float)
    df[NONCONFORMITY_SCORE] = np.array([np.nan, 1.0, 2.0], dtype=float)
    df[CONFORMAL_ALPHA] = np.array([0.1, 0.1, 0.1], dtype=float)
    df[CALIBRATION_STATE] = ["{}", "{}", "{}"]
    df[CONFORMAL_METHOD] = ["mscp", "mscp", "mscp"]
    validate_forecast_frame(df)


def test_bad_conformal_interval_dtype_raises():
    lower_col, _ = interval_column_names(0.9)
    df = _make_valid_frame()
    df[lower_col] = ["bad", "bad", "bad"]
    with pytest.raises(ValueError, match=lower_col):
        validate_forecast_frame(df)


def test_quantile_column_helper_uses_coverage_suffix():
    assert quantile_column(0.5) == "q_0p5"
    assert quantile_column(0.833) == "q_0p833"


def test_quantile_columns_validate_when_present():
    df = _make_valid_frame()
    df[quantile_column(0.5)] = np.array([10.0, 20.0, 30.0], dtype=float)
    df[quantile_column(0.833)] = np.array([12.0, 22.0, 32.0], dtype=float)
    validate_forecast_frame(df)


def test_bad_quantile_dtype_raises():
    df = _make_valid_frame()
    df[quantile_column(0.5)] = ["bad", "bad", "bad"]
    with pytest.raises(ValueError, match="q_0p5"):
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
