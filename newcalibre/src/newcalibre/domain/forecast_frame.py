"""Define and validate the first-brick forecast-frame schema.

The U1 column vocabulary is deliberately literal: required columns use the
names declared below, while intervals and quantiles use canonical decimal
suffixes such as ``lower_0.9`` and ``quantile_0.5``. Value columns normalize
accepted dense, nullable, Arrow-backed, or sparse real numerics to ``float64``.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
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
_FITTED_VALUE_SIDECAR_COLUMN = "fitted_value"
_FLOAT64 = np.dtype("float64")
_DATETIME_UNITS = frozenset({"s", "ms", "us", "ns"})
_TRANSPORT_STRING_DTYPE = pd.StringDtype(storage="pyarrow")
_SUPPORTED_EXTENSION_DTYPE_NUMS = frozenset(
    np.dtype(name).num
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
)
_INTEGER_DTYPE_NUMS = frozenset(
    np.dtype(name).num
    for name in (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    )
)
_SPARSE_REAL_DTYPE_NUMS = _INTEGER_DTYPE_NUMS | frozenset(
    np.dtype(name).num for name in ("float16", "float32", "float64")
)
_NULLABLE_REAL_DTYPE_NAMES = frozenset(
    {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
    }
)
_ARROW_REAL_TYPES = frozenset(
    {
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
    }
)


class ForecastFrameError(ValueError):
    """Report a forecast frame that fails the chapter 02 schema."""


def interval_columns(level: object) -> tuple[str, str]:
    """Return the canonical lower and upper column names for a coverage level."""
    suffix = _canonical_level(level)
    return f"{_LOWER_PREFIX}{suffix}", f"{_UPPER_PREFIX}{suffix}"


def quantile_column(level: object) -> str:
    """Return the canonical column name for a native quantile level."""
    return f"{_QUANTILE_PREFIX}{_canonical_level(level)}"


def forecast_bound_groups(columns: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Return every canonical quantile or interval group in stable column order.

    Quantiles are singleton groups. Interval lower/upper columns are one
    two-column group even when the two columns are not adjacent in the frame.
    """
    if isinstance(columns, (str, bytes)):
        raise ForecastFrameError("forecast columns must be an iterable of column names")
    try:
        labels = pd.Index(tuple(columns))
    except TypeError as error:
        raise ForecastFrameError("forecast columns must be iterable") from error
    if labels.has_duplicates:
        raise ForecastFrameError("forecast columns contain duplicate labels")
    _require_column_labels(pd.DataFrame(columns=labels))

    optional = _optional_value_columns(labels)
    groups: list[tuple[str, ...]] = []
    grouped_interval_levels: set[str] = set()
    for column in optional:
        if column.startswith(_QUANTILE_PREFIX):
            groups.append((column,))
            continue
        prefix = _LOWER_PREFIX if column.startswith(_LOWER_PREFIX) else _UPPER_PREFIX
        level = column.removeprefix(prefix)
        if level not in grouped_interval_levels:
            groups.append(interval_columns(level))
            grouped_interval_levels.add(level)
    return tuple(groups)


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

    Accepted real-numeric value columns are copied and normalized to
    ``float64``. All other required dtypes must already be exact.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ForecastFrameError("forecast frame must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ForecastFrameError("forecast frame has duplicate column labels")
    _require_column_labels(frame)

    missing = [column for column in REQUIRED_FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ForecastFrameError(f"missing required columns: {', '.join(missing)}")

    optional_value_columns = _optional_value_columns(frame.columns)
    if _FITTED_VALUE_SIDECAR_COLUMN in frame.columns:
        raise ForecastFrameError("fitted values belong in the separate fitted-values sidecar")

    normalized = frame.copy(deep=True).set_flags(allows_duplicate_labels=True)
    normalized.attrs = {}
    normalized.index.name = None
    normalized.columns.name = None
    _require_string(normalized, SERIES_KEY)
    _require_naive_datetime64(normalized, TARGET_TIMESTAMP)
    _normalize_float64(normalized, ACTUAL_VALUE)
    _normalize_float64(normalized, POINT_FORECAST)
    _require_integer(normalized, HORIZON_STEP)
    _require_naive_datetime64(normalized, ORIGIN)
    _require_string(normalized, MODEL_NAME)
    for column in optional_value_columns:
        _normalize_float64(normalized, column)

    owned_columns = {*REQUIRED_FRAME_COLUMNS, *optional_value_columns}
    for column in normalized.columns:
        if column not in owned_columns:
            _canonicalize_extension(normalized, column)

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
    try:
        frame[column] = frame[column].astype(_TRANSPORT_STRING_DTYPE)
    except (TypeError, UnicodeError, ValueError) as error:
        raise ForecastFrameError(
            f"column {column!r} cannot normalize to Arrow-backed string dtype"
        ) from error


def _require_column_labels(frame: pd.DataFrame) -> None:
    if any(not isinstance(column, str) for column in frame.columns):
        raise ForecastFrameError("forecast frame column labels must be strings")
    try:
        for column in frame.columns:
            column.encode("utf-8")
    except UnicodeError as error:
        raise ForecastFrameError(
            "forecast frame column labels must be valid UTF-8 strings"
        ) from error


def _canonicalize_extension(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if isinstance(dtype, pd.StringDtype):
        _require_string(frame, column)
        return
    if isinstance(dtype, pd.ArrowDtype) and dtype.pyarrow_dtype in _ARROW_REAL_TYPES:
        chunked = frame[column].array.__arrow_array__()
        array = (
            pa.concat_arrays(chunked.chunks)
            if chunked.num_chunks
            else pa.array([], type=dtype.pyarrow_dtype)
        )
        mask = array.is_null().to_numpy(zero_copy_only=False)
        values = array.fill_null(0).to_numpy(zero_copy_only=False)
        if pa.types.is_floating(dtype.pyarrow_dtype):
            mask |= np.isnan(values)
        canonical = pa.array(values, mask=mask, type=dtype.pyarrow_dtype, from_pandas=False)
        frame[column] = pd.Series(canonical, index=frame.index, name=column, dtype=dtype)
        return
    if isinstance(dtype, pd.SparseDtype):
        try:
            frame[column] = frame[column].sparse.to_dense()
        except (TypeError, ValueError) as error:
            raise ForecastFrameError(
                f"extension column {column!r} could not densify its sparse values"
            ) from error
        _canonicalize_extension(frame, column)
        return
    nullable_name = str(dtype)
    if nullable_name in _NULLABLE_REAL_DTYPE_NAMES:
        expected = pd.api.types.pandas_dtype(nullable_name)
        if type(dtype) is type(expected):
            return
    if not isinstance(dtype, np.dtype) or not dtype.isnative:
        raise ForecastFrameError(
            f"extension column {column!r} must have a flat transport-safe primitive dtype"
        )
    if dtype.kind == "M":
        _require_naive_datetime64(frame, column)
        return
    if dtype.num not in _SUPPORTED_EXTENSION_DTYPE_NUMS:
        raise ForecastFrameError(
            f"extension column {column!r} must have a flat transport-safe primitive dtype"
        )
    if dtype.kind == "f" and frame[column].isna().any():
        frame.loc[frame[column].isna(), column] = np.nan


def _require_naive_datetime64(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, np.dtype) or dtype.kind != "M":
        raise ForecastFrameError(
            f"column {column!r} must have a timezone-naive numpy datetime64 dtype"
        )
    if str(dtype) not in {f"datetime64[{unit}]" for unit in _DATETIME_UNITS}:
        raise ForecastFrameError(f"column {column!r} must use datetime64[s], [ms], [us], or [ns]")


def _require_integer(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, np.dtype) or not is_integer_dtype(dtype) or is_bool_dtype(dtype):
        raise ForecastFrameError(f"column {column!r} must have an integer dtype")


def _normalize_float64(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if isinstance(dtype, pd.ArrowDtype) and dtype.pyarrow_dtype in _ARROW_REAL_TYPES:
        try:
            values = frame[column].array.to_numpy(
                dtype=_FLOAT64,
                na_value=np.nan,
                copy=True,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ForecastFrameError(
                f"column {column!r} could not normalize Arrow-backed values to float64"
            ) from error
        frame[column] = pd.Series(values, index=frame.index, name=column)
        return
    if isinstance(dtype, pd.SparseDtype):
        try:
            dense = frame[column].sparse.to_dense()
        except (TypeError, ValueError) as error:
            raise ForecastFrameError(
                f"column {column!r} could not densify its sparse real values"
            ) from error
        dense_dtype = dense.dtype
        if (
            not isinstance(dense_dtype, np.dtype)
            or not dense_dtype.isnative
            or dense_dtype.num not in _SPARSE_REAL_DTYPE_NUMS
        ):
            raise ForecastFrameError(f"column {column!r} must have exact float64 or integer dtype")
        try:
            frame[column] = dense.astype(_FLOAT64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ForecastFrameError(
                f"column {column!r} could not normalize sparse values to float64"
            ) from error
        return
    nullable_name = str(dtype)
    if nullable_name in _NULLABLE_REAL_DTYPE_NAMES:
        expected = pd.api.types.pandas_dtype(nullable_name)
        if type(dtype) is type(expected):
            values = frame[column].array.to_numpy(
                dtype=_FLOAT64,
                na_value=np.nan,
                copy=True,
            )
            frame[column] = pd.Series(values, index=frame.index, name=column)
            return
    if not isinstance(dtype, np.dtype) or not dtype.isnative:
        raise ForecastFrameError(f"column {column!r} must have exact float64 or integer dtype")
    if dtype.num == _FLOAT64.num:
        return
    if dtype.num in _INTEGER_DTYPE_NUMS:
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
