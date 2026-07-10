"""Define the immutable in-sample fitted-values sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

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


class FittedValuesError(ValueError):
    """Report a malformed fitted-values sidecar."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FittedValues:
    """Own a canonical defensive snapshot distinct from forecast rows."""

    _frame: pd.DataFrame = field(repr=False)

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
            dtype = normalized[column].dtype
            if (
                not isinstance(dtype, np.dtype)
                or not dtype.isnative
                or dtype.num not in _SUPPORTED_INPUT_DTYPE_NUMS
            ):
                raise FittedValuesError(f"fitted-values column {column!r} must be numeric")
            try:
                normalized[column] = normalized[column].astype(_FLOAT64)
            except (TypeError, ValueError, OverflowError) as error:
                raise FittedValuesError(
                    f"fitted-values column {column!r} cannot normalize to float64"
                ) from error

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
        instance = object.__new__(cls)
        object.__setattr__(instance, "_frame", normalized)
        return instance

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy in canonical key order."""
        return self._frame.copy(deep=True)
