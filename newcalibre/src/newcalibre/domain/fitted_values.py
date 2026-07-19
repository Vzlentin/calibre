"""Define the immutable in-sample fitted-values sidecar."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from newcalibre.domain.forecast_frame import ACTUAL_VALUE, MODEL_NAME, SERIES_KEY
from newcalibre.domain.panel import TIMESTAMP

FITTED_VALUE: Final = "fitted_value"
REQUIRED_FITTED_VALUE_COLUMNS: Final = (
    SERIES_KEY,
    TIMESTAMP,
    ACTUAL_VALUE,
    FITTED_VALUE,
    MODEL_NAME,
)
FITTED_VALUE_KEY_COLUMNS: Final = (SERIES_KEY, TIMESTAMP, MODEL_NAME)
_FLOAT64 = np.dtype("float64")
_DATETIME_UNITS = frozenset({"s", "ms", "us", "ns"})
_TRANSPORT_STRING_DTYPE = pd.StringDtype(storage="pyarrow")
_SUPPORTED_INPUT_DTYPE_NUMS = frozenset(
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
        "float16",
        "float32",
        "float64",
    )
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


class FittedValuesError(ValueError):
    """Report a malformed fitted-values sidecar."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FittedValues:
    """Own a canonical defensive snapshot distinct from forecast rows."""

    _frame: pd.DataFrame = field(repr=False)
    _series_residual_periods: tuple[tuple[str, str, int], ...] = field(repr=False)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> FittedValues:
        """Validate, normalize, and snapshot one fitted-values sidecar."""
        if not isinstance(frame, pd.DataFrame):
            raise FittedValuesError("fitted values must be a pandas DataFrame")
        if frame.columns.has_duplicates:
            raise FittedValuesError("fitted values have duplicate column labels")
        if set(frame.columns) != set(REQUIRED_FITTED_VALUE_COLUMNS):
            missing = [
                column for column in REQUIRED_FITTED_VALUE_COLUMNS if column not in frame.columns
            ]
            unexpected = [
                column for column in frame.columns if column not in REQUIRED_FITTED_VALUE_COLUMNS
            ]
            raise FittedValuesError(
                "fitted values require the exact schema; "
                f"missing={missing}, unexpected={unexpected}"
            )

        normalized = (
            frame.loc[:, list(REQUIRED_FITTED_VALUE_COLUMNS)]
            .copy(deep=True)
            .set_flags(allows_duplicate_labels=True)
        )
        normalized.attrs = {}
        normalized.index.name = None
        normalized.columns.name = None
        for column in (SERIES_KEY, MODEL_NAME):
            if not isinstance(normalized[column].dtype, pd.StringDtype):
                raise FittedValuesError(
                    f"fitted-values column {column!r} must have pandas string dtype"
                )
            if normalized[column].isna().any() or (normalized[column].str.len() == 0).any():
                raise FittedValuesError(
                    f"fitted-values column {column!r} cannot be missing or empty"
                )
            try:
                normalized[column] = normalized[column].astype(_TRANSPORT_STRING_DTYPE)
            except (TypeError, UnicodeError, ValueError) as error:
                raise FittedValuesError(
                    f"fitted-values column {column!r} cannot normalize to Arrow-backed string dtype"
                ) from error
        timestamp_dtype = normalized[TIMESTAMP].dtype
        if not isinstance(timestamp_dtype, np.dtype) or timestamp_dtype.kind != "M":
            raise FittedValuesError(
                f"fitted-values column {TIMESTAMP!r} must have timezone-naive datetime64 dtype"
            )
        if normalized[TIMESTAMP].isna().any():
            raise FittedValuesError("fitted-values timestamps cannot be missing")
        if str(timestamp_dtype) not in {f"datetime64[{unit}]" for unit in _DATETIME_UNITS}:
            raise FittedValuesError(
                "fitted-values timestamps must use datetime64[s], [ms], [us], or [ns]"
            )

        for column in (ACTUAL_VALUE, FITTED_VALUE):
            normalized[column] = _normalize_real_to_float64(normalized[column], column=column)

        normalized = normalized.dropna(subset=[ACTUAL_VALUE, FITTED_VALUE]).reset_index(drop=True)
        if normalized.duplicated(subset=list(FITTED_VALUE_KEY_COLUMNS)).any():
            raise FittedValuesError("fitted values contain a duplicate full key")
        order = sorted(
            range(len(normalized)),
            key=lambda index: (
                str(normalized.iloc[index][SERIES_KEY]).encode(),
                pd.Timestamp(normalized.iloc[index][TIMESTAMP]),
                str(normalized.iloc[index][MODEL_NAME]).encode(),
            ),
        )
        normalized = (
            normalized.iloc[order].reset_index(drop=True).set_flags(allows_duplicate_labels=True)
        )
        normalized.attrs = {}
        normalized.index.name = None
        normalized.columns.name = None
        residual_period_counts = normalized.groupby(
            [MODEL_NAME, SERIES_KEY],
            sort=True,
            observed=True,
            dropna=False,
        ).size()
        series_residual_periods = tuple(
            (
                str(cast(tuple[str, str], key)[0]),
                str(cast(tuple[str, str], key)[1]),
                int(periods),
            )
            for key, periods in residual_period_counts.items()
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_frame", normalized)
        object.__setattr__(instance, "_series_residual_periods", series_residual_periods)
        return instance

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy in canonical key order."""
        return self._frame.copy(deep=True)

    def residual_periods_for(
        self,
        model_name: str,
        series_keys: Iterable[str],
    ) -> int | None:
        """Return an applicable model subset's period count without copying row data."""
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("fitted-values model name must be a non-empty string")
        selected_keys = _validated_series_keys(series_keys)
        model_counts = {
            series_key: periods
            for candidate, series_key, periods in self._series_residual_periods
            if candidate == model_name
        }
        if not model_counts:
            return None
        return max((model_counts[key] for key in selected_keys if key in model_counts), default=0)

    def select_model_series(
        self,
        model_name: str,
        series_keys: Iterable[str],
    ) -> pd.DataFrame:
        """Return a defensive copy filtered to one model and series subset."""
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("fitted-values model name must be a non-empty string")
        selected_keys = _validated_series_keys(series_keys)
        selected = self._frame.loc[
            (self._frame[MODEL_NAME] == model_name) & self._frame[SERIES_KEY].isin(selected_keys)
        ]
        return selected.copy(deep=True).reset_index(drop=True)


def _validated_series_keys(series_keys: Iterable[str]) -> tuple[str, ...]:
    if isinstance(series_keys, (str, bytes)):
        raise TypeError("fitted-values series keys must be an iterable of labels")
    try:
        selected_keys = tuple(series_keys)
    except TypeError as error:
        raise TypeError("fitted-values series keys must be iterable") from error
    if any(not isinstance(key, str) or not key for key in selected_keys):
        raise ValueError("fitted-values series keys must be non-empty strings")
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError("fitted-values series keys must be unique")
    return selected_keys


def _normalize_real_to_float64(series: pd.Series, *, column: str) -> pd.Series:
    dtype = series.dtype
    if isinstance(dtype, pd.ArrowDtype) and dtype.pyarrow_dtype in _ARROW_REAL_TYPES:
        try:
            values = series.array.to_numpy(
                dtype=_FLOAT64,
                na_value=np.nan,
                copy=True,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise FittedValuesError(
                f"fitted-values column {column!r} cannot normalize to float64"
            ) from error
        return pd.Series(values, index=series.index, name=column, dtype=_FLOAT64)
    if isinstance(dtype, pd.SparseDtype):
        try:
            dense = series.sparse.to_dense()
        except (TypeError, ValueError) as error:
            raise FittedValuesError(
                f"fitted-values column {column!r} cannot densify sparse real values"
            ) from error
        return _normalize_real_to_float64(dense, column=column)

    nullable_name = str(dtype)
    if nullable_name in _NULLABLE_REAL_DTYPE_NAMES:
        expected = pd.api.types.pandas_dtype(nullable_name)
        if type(dtype) is type(expected):
            try:
                values = series.array.to_numpy(
                    dtype=_FLOAT64,
                    na_value=np.nan,
                    copy=True,
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise FittedValuesError(
                    f"fitted-values column {column!r} cannot normalize to float64"
                ) from error
            return pd.Series(values, index=series.index, name=column, dtype=_FLOAT64)

    if (
        not isinstance(dtype, np.dtype)
        or not dtype.isnative
        or dtype.num not in _SUPPORTED_INPUT_DTYPE_NUMS
    ):
        raise FittedValuesError(f"fitted-values column {column!r} must be numeric")
    try:
        return pd.Series(
            series.to_numpy(dtype=_FLOAT64, copy=True),
            index=series.index,
            name=column,
            dtype=_FLOAT64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise FittedValuesError(
            f"fitted-values column {column!r} cannot normalize to float64"
        ) from error
