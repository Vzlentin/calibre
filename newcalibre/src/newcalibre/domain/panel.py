"""Validate a panel once and partition it into forecast tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
import pyarrow as pa

from newcalibre.domain.calendar import Calendar, CalendarError
from newcalibre.domain.forecast_frame import SERIES_KEY

if TYPE_CHECKING:
    from newcalibre.domain.forecast_task import ForecastTask

TIMESTAMP: Final = "timestamp"
OBSERVED_VALUE: Final = "value"
CENSOR_STATUS: Final = "censor_status"
AVAILABILITY_BOUND: Final = "availability_bound"
KNOWN_AT: Final = "known_at"

REQUIRED_PANEL_COLUMNS: Final = (SERIES_KEY, TIMESTAMP, OBSERVED_VALUE)
PANEL_KEY_COLUMNS: Final = (SERIES_KEY, TIMESTAMP)
UNDECLARED_CENSORING: Final = "undeclared"
_DATETIME_UNITS: Final = frozenset({"s", "ms", "us", "ns"})
_TRANSPORT_STRING_DTYPE: Final = pd.StringDtype(storage="pyarrow")
_SUPPORTED_NUMERIC_DTYPE_NUMS: Final = frozenset(
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
_NULLABLE_NUMERIC_DTYPE_NAMES: Final = frozenset(
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
_ARROW_NUMERIC_TYPES = {
    str(arrow_type): arrow_type
    for arrow_type in (
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
    )
}


class PanelError(ValueError):
    """Report a panel or task-partitioning contract violation."""


class Scope(StrEnum):
    """Name the task partition selected once by a panel."""

    LOCAL = "local"
    GLOBAL = "global"


class CensoringAssertion(StrEnum):
    """Name the two assertions a dataset can make about one observation."""

    CENSORED = "censored"
    UNCENSORED = "uncensored"


class TargetSupport(StrEnum):
    """Declare the mathematical support of a panel's forecast target."""

    REAL = "real"
    NONNEGATIVE = "nonnegative"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Panel:
    """Own a canonical defensive snapshot of a whole panel and its calendar."""

    _frame: pd.DataFrame = field(repr=False)
    _calendar: Calendar
    _series_keys: tuple[str, ...]
    _target_support: TargetSupport

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        calendar: Calendar,
        target_support: TargetSupport,
    ) -> Panel:
        """Validate and snapshot a non-empty long-format panel."""
        if not isinstance(target_support, TargetSupport):
            raise PanelError(
                "target support must be TargetSupport.REAL or TargetSupport.NONNEGATIVE"
            )
        normalized, bound_calendar = _canonicalize_panel_frame(
            frame,
            calendar=calendar,
            allow_empty=False,
            bind_calendar=True,
        )
        series_keys = tuple(sorted(normalized[SERIES_KEY].unique(), key=str.encode))
        instance = object.__new__(cls)
        object.__setattr__(instance, "_frame", normalized)
        object.__setattr__(instance, "_calendar", bound_calendar)
        object.__setattr__(instance, "_series_keys", series_keys)
        object.__setattr__(instance, "_target_support", target_support)
        return instance

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy in canonical row and column order."""
        return self._frame.copy(deep=True)

    @property
    def calendar(self) -> Calendar:
        """Return the panel-wide calendar value."""
        return self._calendar

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return exact opaque keys in deterministic UTF-8 byte order."""
        return self._series_keys

    @property
    def target_support(self) -> TargetSupport:
        """Return the declared mathematical support for forecast targets."""
        return self._target_support

    @property
    def has_censoring_facts(self) -> bool:
        """Return whether the source declared either censoring metadata field."""
        return CENSOR_STATUS in self._frame.columns

    def forecast_tasks(
        self,
        *,
        origin: pd.Timestamp,
        horizon: int,
        scope: Scope,
        model_config: Mapping[str, object],
        future_exogenous: pd.DataFrame | None = None,
    ) -> tuple[ForecastTask, ...]:
        """Resolve scope once and construct immutable pre-origin tasks.

        Local scope returns one one-series task per panel key. Global scope
        returns one task carrying the whole panel. Adapters need not inspect
        or branch on scope.
        """
        from newcalibre.domain.forecast_task import ForecastTask

        if not isinstance(scope, Scope):
            raise PanelError("scope must be Scope.LOCAL or Scope.GLOBAL")
        try:
            self._calendar.require_member(origin, name="origin")
        except CalendarError as error:
            raise PanelError(str(error)) from error
        ForecastTask._require_horizon(horizon)

        history = self._frame[self._frame[TIMESTAMP] < origin].reset_index(drop=True)
        future = _canonicalize_future_exogenous(
            future_exogenous,
            calendar=self._calendar,
            origin=origin,
            horizon=int(horizon),
            series_keys=self._series_keys,
        )

        if scope is Scope.GLOBAL:
            return (
                ForecastTask._from_components(
                    history=history,
                    future_exogenous=future,
                    horizon=int(horizon),
                    origin=origin,
                    calendar=self._calendar,
                    model_config=model_config,
                    scope=scope,
                    series_keys=self._series_keys,
                ),
            )

        tasks: list[ForecastTask] = []
        for series_key in self._series_keys:
            local_history = history[history[SERIES_KEY] == series_key].reset_index(drop=True)
            local_future = None
            if future is not None:
                local_future = future[future[SERIES_KEY] == series_key].reset_index(drop=True)
            tasks.append(
                ForecastTask._from_components(
                    history=local_history,
                    future_exogenous=local_future,
                    horizon=int(horizon),
                    origin=origin,
                    calendar=self._calendar,
                    model_config=model_config,
                    scope=scope,
                    series_keys=(series_key,),
                )
            )
        return tuple(tasks)


def _canonicalize_panel_frame(
    frame: pd.DataFrame,
    *,
    calendar: Calendar,
    allow_empty: bool,
    bind_calendar: bool = False,
) -> tuple[pd.DataFrame, Calendar]:
    if not isinstance(calendar, Calendar):
        raise PanelError("calendar must be a Calendar")
    if not isinstance(frame, pd.DataFrame):
        raise PanelError("panel must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise PanelError("panel has duplicate column labels")
    _require_column_labels(frame, surface="panel")
    missing = [column for column in REQUIRED_PANEL_COLUMNS if column not in frame.columns]
    if missing:
        raise PanelError(f"panel is missing required columns: {', '.join(missing)}")
    if not allow_empty and frame.empty:
        raise PanelError("panel must contain at least one observation row")

    normalized = _data_only_copy(frame)
    _require_string(normalized, SERIES_KEY, surface="panel")
    _require_naive_datetime(normalized, TIMESTAMP, surface="panel")
    _require_numeric(normalized, OBSERVED_VALUE, surface="panel")

    if normalized[SERIES_KEY].isna().any() or (normalized[SERIES_KEY].str.len() == 0).any():
        raise PanelError("panel series keys must be non-missing, non-empty strings")
    if normalized[TIMESTAMP].isna().any():
        raise PanelError("panel timestamps cannot be missing")
    if normalized.duplicated(subset=list(PANEL_KEY_COLUMNS)).any():
        raise PanelError("panel contains a duplicate series/timestamp key")
    effective_calendar = calendar
    if bind_calendar:
        try:
            effective_calendar = calendar.bind(pd.Timestamp(normalized[TIMESTAMP].min()))
        except CalendarError as error:
            raise PanelError(f"panel {error}") from error
    last_timestamp_by_series: dict[str, pd.Timestamp] = {}
    timestamps_out_of_order = False
    for series_key, raw_timestamp in zip(
        normalized[SERIES_KEY], normalized[TIMESTAMP], strict=True
    ):
        timestamp = pd.Timestamp(raw_timestamp)
        try:
            effective_calendar.require_member(timestamp)
        except CalendarError as error:
            raise PanelError(f"panel {error}") from error
        previous_timestamp = last_timestamp_by_series.get(series_key)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            timestamps_out_of_order = True
        last_timestamp_by_series[series_key] = timestamp
    if timestamps_out_of_order:
        raise PanelError("panel timestamps must be strictly increasing within each series")

    metadata_columns: list[str] = []
    has_censoring_facts = (
        CENSOR_STATUS in normalized.columns or AVAILABILITY_BOUND in normalized.columns
    )
    if has_censoring_facts:
        if CENSOR_STATUS not in normalized.columns:
            normalized[CENSOR_STATUS] = pd.Series(
                UNDECLARED_CENSORING,
                index=normalized.index,
                dtype=_TRANSPORT_STRING_DTYPE,
            )
        else:
            _require_string(normalized, CENSOR_STATUS, surface="panel")
            normalized[CENSOR_STATUS] = normalized[CENSOR_STATUS].fillna(UNDECLARED_CENSORING)
        assertions = set(normalized[CENSOR_STATUS].tolist())
        allowed = {assertion.value for assertion in CensoringAssertion} | {UNDECLARED_CENSORING}
        if not assertions <= allowed:
            invalid = sorted(assertions - allowed)
            raise PanelError(
                "censor_status must be 'censored', 'uncensored', or the "
                f"'undeclared' sentinel; invalid values: {invalid}"
            )
        metadata_columns.append(CENSOR_STATUS)
    if AVAILABILITY_BOUND in normalized.columns:
        _require_numeric(normalized, AVAILABILITY_BOUND, surface="panel")
        metadata_columns.append(AVAILABILITY_BOUND)

    declared = {*REQUIRED_PANEL_COLUMNS, CENSOR_STATUS, AVAILABILITY_BOUND}
    exogenous = sorted(
        (column for column in normalized.columns if column not in declared), key=str.encode
    )
    for column in exogenous:
        _require_numeric(normalized, column, surface="panel exogenous")

    columns = [*REQUIRED_PANEL_COLUMNS, *metadata_columns, *exogenous]
    normalized = normalized.loc[:, columns]
    normalized = normalized.sort_values(
        list(PANEL_KEY_COLUMNS),
        kind="mergesort",
        ignore_index=True,
    )
    normalized = _clear_pandas_metadata(normalized)
    return normalized, effective_calendar


def _canonicalize_future_exogenous(
    frame: pd.DataFrame | None,
    *,
    calendar: Calendar,
    origin: pd.Timestamp,
    horizon: int,
    series_keys: tuple[str, ...],
) -> pd.DataFrame | None:
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise PanelError("future exogenous data must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise PanelError("future exogenous data has duplicate column labels")
    _require_column_labels(frame, surface="future exogenous")
    required = (SERIES_KEY, TIMESTAMP, KNOWN_AT)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PanelError(f"future exogenous data is missing columns: {', '.join(missing)}")
    regressors = sorted(
        (column for column in frame.columns if column not in required), key=str.encode
    )
    if not regressors:
        raise PanelError("future exogenous data requires at least one numeric regressor")

    normalized = _data_only_copy(frame)
    _require_string(normalized, SERIES_KEY, surface="future exogenous")
    _require_naive_datetime(normalized, TIMESTAMP, surface="future exogenous")
    _require_naive_datetime(normalized, KNOWN_AT, surface="future exogenous")
    for column in regressors:
        _require_numeric(normalized, column, surface="future exogenous")
    if normalized[list(required)].isna().any(axis=None):
        raise PanelError("future exogenous keys and known_at cannot be missing")
    if (normalized[SERIES_KEY].str.len() == 0).any():
        raise PanelError("future exogenous series keys must be non-empty")
    if normalized[regressors].isna().any(axis=None):
        raise PanelError("future exogenous regressor values must be known at the origin")
    if normalized.duplicated(subset=[SERIES_KEY, TIMESTAMP]).any():
        raise PanelError("future exogenous data contains a duplicate series/timestamp key")

    allowed_keys = set(series_keys)
    unknown = sorted(set(normalized[SERIES_KEY]) - allowed_keys, key=str.encode)
    if unknown:
        raise PanelError(f"future exogenous data contains unknown series keys: {unknown}")
    if not normalized[KNOWN_AT].le(origin).all():
        raise PanelError("every future exogenous value must be known at or before origin")
    for timestamp in normalized[TIMESTAMP]:
        try:
            calendar.require_member(pd.Timestamp(timestamp), name="future exogenous timestamp")
        except CalendarError as error:
            raise PanelError(str(error)) from error
    targets = {calendar.advance(origin, step) for step in range(horizon)}
    if not normalized[TIMESTAMP].map(pd.Timestamp).isin(targets).all():
        raise PanelError("future exogenous timestamps must lie within the task horizon")

    normalized = normalized.loc[:, [*required, *regressors]]
    order = sorted(
        range(len(normalized)),
        key=lambda index: (
            str(normalized.iloc[index][SERIES_KEY]).encode(),
            pd.Timestamp(normalized.iloc[index][TIMESTAMP]),
        ),
    )
    normalized = normalized.iloc[order].reset_index(drop=True)
    normalized = _clear_pandas_metadata(normalized)
    return normalized


def _require_string(frame: pd.DataFrame, column: str, *, surface: str) -> None:
    if not isinstance(frame[column].dtype, pd.StringDtype):
        raise PanelError(f"{surface} column {column!r} must have pandas string dtype")
    try:
        frame[column] = frame[column].astype(_TRANSPORT_STRING_DTYPE)
    except (TypeError, UnicodeError, ValueError) as error:
        raise PanelError(
            f"{surface} column {column!r} cannot normalize to Arrow-backed string dtype"
        ) from error


def _require_naive_datetime(frame: pd.DataFrame, column: str, *, surface: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, np.dtype) or dtype.kind != "M":
        raise PanelError(
            f"{surface} column {column!r} must have a timezone-naive numpy datetime64 dtype"
        )
    if str(dtype) not in {f"datetime64[{unit}]" for unit in _DATETIME_UNITS}:
        raise PanelError(f"{surface} column {column!r} must use datetime64[s], [ms], [us], or [ns]")


def _require_numeric(frame: pd.DataFrame, column: str, *, surface: str) -> None:
    dtype = frame[column].dtype
    arrow_type = _arrow_numeric_type(dtype)
    if arrow_type is not None:
        _canonicalize_arrow_numeric(frame, column, arrow_type=arrow_type)
        return
    if isinstance(dtype, pd.SparseDtype):
        try:
            frame[column] = frame[column].sparse.to_dense()
        except (TypeError, ValueError) as error:
            raise PanelError(
                f"{surface} column {column!r} cannot densify its sparse numeric values"
            ) from error
        _require_numeric(frame, column, surface=surface)
        return
    nullable_name = _nullable_numeric_dtype_name(dtype)
    if nullable_name is not None:
        _canonicalize_nullable_numeric(frame, column, dtype_name=nullable_name)
        return
    if (
        not isinstance(dtype, np.dtype)
        or not dtype.isnative
        or dtype.num not in _SUPPORTED_NUMERIC_DTYPE_NUMS
    ):
        raise PanelError(
            f"{surface} column {column!r} must use a native NumPy integer, unsigned, "
            "or floating dtype supported by Arrow"
        )
    if dtype.kind == "f" and frame[column].isna().any():
        frame.loc[frame[column].isna(), column] = np.nan


def _nullable_numeric_dtype_name(dtype: object) -> str | None:
    name = str(dtype)
    if name not in _NULLABLE_NUMERIC_DTYPE_NAMES:
        return None
    expected = pd.api.types.pandas_dtype(name)
    return name if type(dtype) is type(expected) else None


def _arrow_numeric_type(dtype: object) -> pa.DataType | None:
    if not isinstance(dtype, pd.ArrowDtype):
        return None
    arrow_type = dtype.pyarrow_dtype
    expected = _ARROW_NUMERIC_TYPES.get(str(arrow_type))
    return expected if expected == arrow_type else None


def _canonicalize_arrow_numeric(
    frame: pd.DataFrame, column: str, *, arrow_type: pa.DataType
) -> None:
    chunked = frame[column].array.__arrow_array__()
    array = (
        pa.concat_arrays(chunked.chunks) if chunked.num_chunks else pa.array([], type=arrow_type)
    )
    mask = array.is_null().to_numpy(zero_copy_only=False)
    values = array.fill_null(0).to_numpy(zero_copy_only=False)
    if pa.types.is_floating(arrow_type):
        mask |= np.isnan(values)
    canonical = pa.array(values, mask=mask, type=arrow_type, from_pandas=False)
    frame[column] = pd.Series(
        canonical,
        index=frame.index,
        name=column,
        dtype=pd.ArrowDtype(arrow_type),
    )


def _canonicalize_nullable_numeric(frame: pd.DataFrame, column: str, *, dtype_name: str) -> None:
    numpy_dtype = np.dtype(dtype_name.lower())
    extension = frame[column].array
    mask = np.asarray(extension.isna(), dtype=bool)
    values = extension.to_numpy(dtype=numpy_dtype, na_value=0, copy=True)
    if numpy_dtype.kind == "f":
        mask |= np.isnan(values)
        values[mask] = 0
        canonical = pd.arrays.FloatingArray(values, mask)
    else:
        canonical = pd.arrays.IntegerArray(values, mask)
    frame[column] = pd.Series(canonical, index=frame.index, name=column)


def _data_only_copy(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    return _clear_pandas_metadata(normalized)


def _require_column_labels(frame: pd.DataFrame, *, surface: str) -> None:
    if any(not isinstance(column, str) for column in frame.columns):
        raise PanelError(f"{surface} column labels must be strings")
    try:
        for column in frame.columns:
            column.encode("utf-8")
    except UnicodeError as error:
        raise PanelError(f"{surface} column labels must be valid UTF-8 strings") from error


def _clear_pandas_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.set_flags(allows_duplicate_labels=True)
    normalized.attrs = {}
    normalized.index.name = None
    normalized.columns.name = None
    return normalized
