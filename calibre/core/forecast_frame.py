from __future__ import annotations

import pandas as pd

UNIQUE_ID = "unique_id"
DS = "ds"
Y = "y"
Y_HAT = "y_hat"
H = "h"
FORECAST_ORIGIN = "forecast_origin"
MODEL_NAME = "model_name"
FITTED_Y_HAT = "fitted_y_hat"
NONCONFORMITY_SCORE = "nonconformity_score"
CALIBRATION_STATE = "calibration_state"
CALIBRATION_STATE_REF = "calibration_state_ref"
CONFORMAL_PARTITION = "conformal_partition"
CONFORMAL_METHOD = "conformal_method"
CONFORMAL_ALPHA = "conformal_alpha"
CONFORMAL_MODE = "conformal_mode"
IN_STOCK = "in_stock"

REQUIRED_COLUMNS = [UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]
FITTED_VALUE_COLUMNS = [UNIQUE_ID, DS, Y, MODEL_NAME, FITTED_Y_HAT]

_RESERVED_HISTORY_COLS = frozenset({UNIQUE_ID, DS, Y})


def exogenous_columns(df: pd.DataFrame) -> list[str]:
    """Columns of ``df`` that are not ``{unique_id, ds, y}`` — i.e. regressors.

    Every column in ``history`` that is not one of the three reserved columns
    (``unique_id``, ``ds``, ``y``) is treated as an exogenous regressor and
    forwarded to the library's ``fit`` call.  Callers are responsible for
    ensuring that any extra columns in ``history`` are genuine numeric
    regressors; metadata columns (e.g. ``"category"``, ``"store_id"``) will be
    passed through and may cause errors inside the underlying library.
    """
    return [c for c in df.columns if c not in _RESERVED_HISTORY_COLS]


_EXPECTED_DTYPES = {
    UNIQUE_ID: "object",
    DS: "datetime64[ns]",
    Y: "float64",
    Y_HAT: "float64",
    H: "int64",
    FORECAST_ORIGIN: "datetime64[ns]",
    MODEL_NAME: "object",
}

_OPTIONAL_DTYPES = {
    NONCONFORMITY_SCORE: "float64",
    CALIBRATION_STATE: "object",
    CALIBRATION_STATE_REF: "object",
    CONFORMAL_PARTITION: "object",
    CONFORMAL_METHOD: "object",
    CONFORMAL_ALPHA: "float64",
    CONFORMAL_MODE: "object",
}


def _format_coverage_suffix(coverage: float) -> str:
    value = f"{float(coverage):.12g}"
    return value.replace(".", "p")


def lower_interval_column(coverage: float) -> str:
    return f"lo_{_format_coverage_suffix(coverage)}"


def upper_interval_column(coverage: float) -> str:
    return f"hi_{_format_coverage_suffix(coverage)}"


def interval_column_names(coverage: float) -> tuple[str, str]:
    return lower_interval_column(coverage), upper_interval_column(coverage)


def quantile_column(quantile: float) -> str:
    """Column name for a predicted quantile (e.g. 0.833 -> ``q_0p833``)."""
    return f"q_{_format_coverage_suffix(quantile)}"


def _is_interval_column(column: str) -> bool:
    return column.startswith("lo_") or column.startswith("hi_")


def is_quantile_column(column: str) -> bool:
    """Return True iff ``column`` is a per-quantile prediction column (``q_*``)."""
    return column.startswith("q_")


def _validate_dtype(df: pd.DataFrame, col: str, expected: str) -> None:
    actual = str(df[col].dtype)
    if expected == "datetime64[ns]":
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"Column '{col}' expected datetime64, got {actual}")
        return
    if actual != expected:
        raise ValueError(f"Column '{col}' expected {expected}, got {actual}")


def validate_forecast_frame(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the forecast-frame contract.

    Raises ValueError if validation fails.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col, expected in _EXPECTED_DTYPES.items():
        _validate_dtype(df, col, expected)

    for col, expected in _OPTIONAL_DTYPES.items():
        if col in df.columns:
            _validate_dtype(df, col, expected)

    for col in df.columns:
        if _is_interval_column(col) and not pd.api.types.is_numeric_dtype(df[col]):
            actual = str(df[col].dtype)
            raise ValueError(f"Column '{col}' expected numeric interval bounds, got {actual}")
        if is_quantile_column(col) and not pd.api.types.is_numeric_dtype(df[col]):
            actual = str(df[col].dtype)
            raise ValueError(f"Column '{col}' expected numeric quantile values, got {actual}")


def validate_actuals_frame(df: pd.DataFrame) -> None:
    missing = {UNIQUE_ID, DS, Y} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required actuals columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df[DS]):
        raise ValueError(f"Column '{DS}' expected datetime64, got {df[DS].dtype}")
    if not pd.api.types.is_numeric_dtype(df[Y]):
        raise ValueError(f"Column '{Y}' expected numeric, got {df[Y].dtype}")
    if df[[UNIQUE_ID, DS]].isna().any().any():
        raise ValueError("Actuals frame has null values in key columns")


def validate_fitted_values_frame(df: pd.DataFrame) -> None:
    """Validate the in-sample fitted-value sidecar contract."""
    missing = set(FITTED_VALUE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing fitted-value columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df[DS]):
        raise ValueError(f"Column '{DS}' expected datetime64, got {df[DS].dtype}")
    if not pd.api.types.is_numeric_dtype(df[Y]):
        raise ValueError(f"Column '{Y}' expected numeric, got {df[Y].dtype}")
    if not pd.api.types.is_numeric_dtype(df[FITTED_Y_HAT]):
        raise ValueError(f"Column '{FITTED_Y_HAT}' expected numeric, got {df[FITTED_Y_HAT].dtype}")
    if df[[UNIQUE_ID, DS, MODEL_NAME]].isna().any().any():
        raise ValueError("Fitted-value frame has null values in key columns")
    if df[[Y, FITTED_Y_HAT]].isna().any().any():
        raise ValueError("Fitted-value frame has null values in y/fitted_y_hat")
    duplicates = df[df.duplicated([UNIQUE_ID, DS, MODEL_NAME], keep=False)]
    if not duplicates.empty:
        keys = (
            duplicates[[UNIQUE_ID, DS, MODEL_NAME]]
            .drop_duplicates()
            .sort_values([UNIQUE_ID, DS, MODEL_NAME], kind="stable")
        )
        values = [tuple(row) for row in keys.itertuples(index=False, name=None)]
        raise ValueError(f"Duplicate fitted-value rows for keys: {values}")
