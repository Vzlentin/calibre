"""Define and validate the first-brick forecast-frame schema.

The U1 column vocabulary is deliberately literal: required columns use the
names declared below, while intervals and quantiles use canonical decimal
suffixes such as ``lower_0.9`` and ``quantile_0.5``. Value columns normalize
integer inputs to ``float64``; other dtype families are validated without
coercion.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from newcalibre.domain.calendar import Calendar, CalendarError

SERIES_KEY: Final = "series_key"
TARGET_TIMESTAMP: Final = "target_timestamp"
ACTUAL_VALUE: Final = "actual_value"
POINT_FORECAST: Final = "point_forecast"
HORIZON_STEP: Final = "horizon_step"
ORIGIN: Final = "origin"
MODEL_NAME: Final = "model_name"

REQUIRED_FRAME_COLUMNS: Final = (
    SERIES_KEY,
    TARGET_TIMESTAMP,
    ACTUAL_VALUE,
    POINT_FORECAST,
    HORIZON_STEP,
    ORIGIN,
    MODEL_NAME,
)
FRAME_KEY_COLUMNS: Final = (SERIES_KEY, ORIGIN, HORIZON_STEP, MODEL_NAME)

_LOWER_PREFIX = "lower_"
_UPPER_PREFIX = "upper_"
_QUANTILE_PREFIX = "quantile_"
_OPTIONAL_PREFIXES = (_LOWER_PREFIX, _UPPER_PREFIX, _QUANTILE_PREFIX)
_OPTIONAL_STEMS = ("lower", "upper", "quantile")
_FLOAT64 = np.dtype("float64")


class ForecastFrameError(ValueError):
    """Report a forecast frame that fails the chapter 02 schema."""


def interval_columns(level: object) -> tuple[str, str]:
    """Return the canonical lower and upper column names for a coverage level."""
    suffix = _canonical_level(level)
    return f"{_LOWER_PREFIX}{suffix}", f"{_UPPER_PREFIX}{suffix}"


def quantile_column(level: object) -> str:
    """Return the canonical column name for a native quantile level."""
    return f"{_QUANTILE_PREFIX}{_canonical_level(level)}"


def target_timestamp(
    origin: pd.Timestamp,
    horizon_step: int,
    *,
    calendar: Calendar,
) -> pd.Timestamp:
    """Derive a row target as origin advanced ``horizon_step - 1`` periods."""
    if not isinstance(horizon_step, Integral) or isinstance(horizon_step, bool) or horizon_step < 1:
        raise ForecastFrameError("horizon step must be a positive integer")
    if not isinstance(calendar, Calendar):
        raise ForecastFrameError("calendar must be a Calendar")
    try:
        return calendar.advance(origin, int(horizon_step) - 1)
    except CalendarError as error:
        raise ForecastFrameError(str(error)) from error


def validate_forecast_frame(
    frame: pd.DataFrame,
    *,
    calendar: Calendar,
) -> pd.DataFrame:
    """Validate a frame atomically and return its normalized copy.

    Integer forecast-value columns are the sole coercion: they are copied and
    upcast to ``float64``. All other required dtypes must already be exact.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ForecastFrameError("forecast frame must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ForecastFrameError("forecast frame has duplicate column labels")

    missing = [column for column in REQUIRED_FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ForecastFrameError(f"missing required columns: {', '.join(missing)}")

    optional_value_columns = _optional_value_columns(frame.columns)
    normalized = frame.copy(deep=True)
    _require_string(normalized, SERIES_KEY)
    _require_naive_datetime64(normalized, TARGET_TIMESTAMP)
    _normalize_float64(normalized, ACTUAL_VALUE)
    _normalize_float64(normalized, POINT_FORECAST)
    _require_integer(normalized, HORIZON_STEP)
    _require_naive_datetime64(normalized, ORIGIN)
    _require_string(normalized, MODEL_NAME)
    for column in optional_value_columns:
        _normalize_float64(normalized, column)

    _validate_row_identity(normalized)
    _validate_target_timestamps(normalized, calendar)
    return normalized


def _canonical_level(level: object) -> str:
    if isinstance(level, bool):
        raise ForecastFrameError("forecast level must be a valid decimal")
    try:
        decimal_level = Decimal(str(level))
    except (InvalidOperation, ValueError) as error:
        raise ForecastFrameError("forecast level must be a valid decimal") from error
    if not decimal_level.is_finite():
        raise ForecastFrameError("forecast level must be a finite decimal")
    if decimal_level == 0:
        return "0"
    return format(decimal_level.normalize(), "f")


def _optional_value_columns(columns: pd.Index) -> list[str]:
    lower_levels: set[str] = set()
    upper_levels: set[str] = set()
    optional: list[str] = []

    for column in columns:
        if not isinstance(column, str):
            continue
        prefix = next(
            (candidate for candidate in _OPTIONAL_PREFIXES if column.startswith(candidate)),
            None,
        )
        if prefix is None:
            if _looks_like_malformed_optional_name(column):
                raise ForecastFrameError(
                    f"optional forecast column {column!r} has a malformed reserved name"
                )
            continue
        suffix = column.removeprefix(prefix)
        canonical = _canonical_level(suffix)
        if suffix != canonical:
            raise ForecastFrameError(
                f"optional forecast column {column!r} must use canonical level {canonical!r}"
            )
        optional.append(column)
        if prefix == _LOWER_PREFIX:
            lower_levels.add(suffix)
        elif prefix == _UPPER_PREFIX:
            upper_levels.add(suffix)

    if lower_levels != upper_levels:
        missing_upper = sorted(lower_levels - upper_levels)
        missing_lower = sorted(upper_levels - lower_levels)
        details: list[str] = []
        if missing_lower:
            details.append(f"missing lower for {missing_lower}")
        if missing_upper:
            details.append(f"missing upper for {missing_upper}")
        raise ForecastFrameError(
            "interval columns require complete lower/upper pairs: " + "; ".join(details)
        )
    return optional


def _looks_like_malformed_optional_name(column: str) -> bool:
    for stem in _OPTIONAL_STEMS:
        if column == stem:
            return True
        if column.startswith(stem):
            following = column[len(stem) : len(stem) + 1]
            return not following.isalpha()
    return False


def _require_string(frame: pd.DataFrame, column: str) -> None:
    if not isinstance(frame[column].dtype, pd.StringDtype):
        raise ForecastFrameError(f"column {column!r} must have pandas string dtype")


def _require_naive_datetime64(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, np.dtype) or dtype.kind != "M":
        raise ForecastFrameError(
            f"column {column!r} must have a timezone-naive numpy datetime64 dtype"
        )


def _require_integer(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if not is_integer_dtype(dtype) or is_bool_dtype(dtype):
        raise ForecastFrameError(f"column {column!r} must have an integer dtype")


def _normalize_float64(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if dtype == _FLOAT64:
        return
    if is_integer_dtype(dtype) and not is_bool_dtype(dtype):
        try:
            frame[column] = frame[column].astype(_FLOAT64)
        except (TypeError, ValueError) as error:
            raise ForecastFrameError(f"column {column!r} could not be upcast to float64") from error
        return
    raise ForecastFrameError(f"column {column!r} must have exact float64 or integer dtype")


def _validate_row_identity(frame: pd.DataFrame) -> None:
    if frame[list(FRAME_KEY_COLUMNS)].isna().any(axis=None):
        raise ForecastFrameError("full row key cannot contain missing values")
    if (frame[SERIES_KEY].str.len() == 0).any():
        raise ForecastFrameError("series key must be non-empty")
    if (frame[MODEL_NAME].str.len() == 0).any():
        raise ForecastFrameError("model name must be non-empty")
    if (frame[HORIZON_STEP] < 1).any():
        raise ForecastFrameError("horizon step must be a positive integer")
    if frame.duplicated(subset=list(FRAME_KEY_COLUMNS)).any():
        raise ForecastFrameError("forecast frame contains a duplicate full row key")


def _validate_target_timestamps(frame: pd.DataFrame, calendar: Calendar) -> None:
    if not isinstance(calendar, Calendar):
        raise ForecastFrameError("calendar must be a Calendar")
    expected: list[pd.Timestamp] = []
    for origin, horizon_step in zip(frame[ORIGIN], frame[HORIZON_STEP], strict=True):
        if pd.isna(origin):
            raise ForecastFrameError("origin cannot be missing")
        try:
            expected.append(calendar.advance(pd.Timestamp(origin), int(horizon_step) - 1))
        except CalendarError as error:
            raise ForecastFrameError(str(error)) from error

    actual_targets = pd.DatetimeIndex(frame[TARGET_TIMESTAMP])
    expected_targets = pd.DatetimeIndex(expected)
    mismatch = actual_targets != expected_targets
    if mismatch.any():
        first = int(np.flatnonzero(mismatch)[0])
        raise ForecastFrameError(
            "target timestamp must equal origin advanced horizon_step - 1 periods; "
            f"row {first} has {actual_targets[first]!s}, expected {expected_targets[first]!s}"
        )
