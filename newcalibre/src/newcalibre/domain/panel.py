"""Validate a panel once and partition it into forecast tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

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
_CENSOR_ASSERTIONS: Final = frozenset({"censored", "uncensored"})


class PanelError(ValueError):
    """Report a panel or task-partitioning contract violation."""


class Scope(StrEnum):
    """Name the task partition selected once by a panel."""

    LOCAL = "local"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Panel:
    """Own a canonical defensive snapshot of a whole panel and its calendar."""

    _frame: pd.DataFrame = field(repr=False)
    _calendar: Calendar
    _series_keys: tuple[str, ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, calendar: Calendar) -> Panel:
        """Validate and snapshot a non-empty long-format panel."""
        normalized = _canonicalize_panel_frame(frame, calendar=calendar, allow_empty=False)
        series_keys = tuple(sorted(normalized[SERIES_KEY].unique(), key=str.encode))
        instance = object.__new__(cls)
        object.__setattr__(instance, "_frame", normalized)
        object.__setattr__(instance, "_calendar", calendar)
        object.__setattr__(instance, "_series_keys", series_keys)
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
    frame: pd.DataFrame, *, calendar: Calendar, allow_empty: bool
) -> pd.DataFrame:
    if not isinstance(calendar, Calendar):
        raise PanelError("calendar must be a Calendar")
    if not isinstance(frame, pd.DataFrame):
        raise PanelError("panel must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise PanelError("panel has duplicate column labels")
    if any(not isinstance(column, str) for column in frame.columns):
        raise PanelError("panel column labels must be strings")
    missing = [column for column in REQUIRED_PANEL_COLUMNS if column not in frame.columns]
    if missing:
        raise PanelError(f"panel is missing required columns: {', '.join(missing)}")
    if not allow_empty and frame.empty:
        raise PanelError("panel must contain at least one observation row")

    normalized = frame.copy(deep=True)
    _require_string(normalized, SERIES_KEY, surface="panel")
    _require_naive_datetime(normalized, TIMESTAMP, surface="panel")
    _require_numeric(normalized, OBSERVED_VALUE, surface="panel")

    if normalized[SERIES_KEY].isna().any() or (normalized[SERIES_KEY].str.len() == 0).any():
        raise PanelError("panel series keys must be non-missing, non-empty strings")
    if normalized[TIMESTAMP].isna().any():
        raise PanelError("panel timestamps cannot be missing")
    if normalized.duplicated(subset=list(PANEL_KEY_COLUMNS)).any():
        raise PanelError("panel contains a duplicate series/timestamp key")
    for timestamp in normalized[TIMESTAMP]:
        try:
            calendar.require_member(pd.Timestamp(timestamp))
        except CalendarError as error:
            raise PanelError(f"panel {error}") from error

    metadata_columns = [CENSOR_STATUS]
    if CENSOR_STATUS not in normalized.columns:
        normalized[CENSOR_STATUS] = pd.Series(pd.NA, index=normalized.index, dtype="string")
    else:
        _require_string(normalized, CENSOR_STATUS, surface="panel")
        assertions = set(normalized[CENSOR_STATUS].dropna().tolist())
        if not assertions <= _CENSOR_ASSERTIONS:
            invalid = sorted(assertions - _CENSOR_ASSERTIONS)
            raise PanelError(
                "censor_status assertions must be 'censored' or 'uncensored'; "
                f"invalid values: {invalid}"
            )
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
    order = sorted(
        range(len(normalized)),
        key=lambda index: (
            str(normalized.iloc[index][SERIES_KEY]).encode(),
            pd.Timestamp(normalized.iloc[index][TIMESTAMP]),
        ),
    )
    return normalized.iloc[order].reset_index(drop=True)


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
    if any(not isinstance(column, str) for column in frame.columns):
        raise PanelError("future exogenous column labels must be strings")
    required = (SERIES_KEY, TIMESTAMP, KNOWN_AT)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PanelError(f"future exogenous data is missing columns: {', '.join(missing)}")
    regressors = sorted(
        (column for column in frame.columns if column not in required), key=str.encode
    )
    if not regressors:
        raise PanelError("future exogenous data requires at least one numeric regressor")

    normalized = frame.copy(deep=True)
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
    return normalized.iloc[order].reset_index(drop=True)


def _require_string(frame: pd.DataFrame, column: str, *, surface: str) -> None:
    if not isinstance(frame[column].dtype, pd.StringDtype):
        raise PanelError(f"{surface} column {column!r} must have pandas string dtype")


def _require_naive_datetime(frame: pd.DataFrame, column: str, *, surface: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, np.dtype) or dtype.kind != "M":
        raise PanelError(
            f"{surface} column {column!r} must have a timezone-naive numpy datetime64 dtype"
        )


def _require_numeric(frame: pd.DataFrame, column: str, *, surface: str) -> None:
    dtype = frame[column].dtype
    if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
        raise PanelError(f"{surface} column {column!r} must be numeric and not boolean")
