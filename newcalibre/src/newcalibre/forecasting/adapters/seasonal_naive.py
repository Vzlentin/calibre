"""Forecast from deterministic incrementally maintained seasonal state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from numbers import Integral
from typing import Final, Never

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from newcalibre.domain import (
    ACTUAL_VALUE,
    HISTORY_TIMESTAMP,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    Calendar,
    FittedValues,
    ForecastTask,
    HistoryCursor,
    HistoryDelta,
    target_timestamp,
    validate_forecast_frame,
)
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.forecasting.protocol import (
    AdapterCapability,
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterDataError,
    AdapterLifecycleError,
)

SEASONAL_NAIVE_BACKEND: Final = "seasonal-naive"
HISTORY_VALUE: Final = "value"
_STATE_SCHEMA: Final = "newcalibre.seasonal-naive-state/v1"

_RetainedSeason = tuple[tuple[pd.Timestamp, float], ...]


@dataclass(frozen=True, slots=True)
class _EffectiveConfig:
    backend: str
    season_length: int
    model_name: str
    requested_capabilities: frozenset[AdapterCapability]


class SeasonalNaiveAdapter:
    """Repeat phases from the latest incrementally maintained complete season."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        config = self._effective_config(model_config)
        self._config = config
        self._season_length = config.season_length
        self._model_name = config.model_name
        self._fit_calendar: Calendar | None = None
        self._cursor: HistoryCursor | None = None
        self._series_keys: tuple[str, ...] | None = None
        self._season_by_series: dict[str, _RetainedSeason] | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        """Declare deterministic incremental update and native persistence."""
        return frozenset(
            {
                AdapterCapability.ARTIFACT_PERSISTENCE,
                AdapterCapability.INCREMENTAL_UPDATE,
            }
        )

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        """Return immutable capability requests derived from configuration."""
        return self._config.requested_capabilities

    def fit(self, task: ForecastTask) -> None:
        """Retain only the final pre-origin season for each task series."""
        self._require_matching_config(task)
        unsupported = self.requested_capabilities - self.capabilities
        if unsupported:
            self._raise_unsupported(min(unsupported, key=lambda capability: capability.value))
        history = task.history.materialize()
        self._validate_history_surface(history, task.series_keys)
        self._fit_calendar = task.calendar
        self._cursor = task.cursor
        self._series_keys = task.series_keys
        self._season_by_series = self._retain_at_origin(
            history,
            series_keys=task.series_keys,
            origin=task.origin,
            previous=None,
        )

    def update(self, delta: HistoryDelta) -> None:
        """Advance retained state using only the supplied contiguous delta."""
        retained = self._require_fitted()
        if not isinstance(delta, HistoryDelta):
            raise TypeError("seasonal update requires a HistoryDelta")
        if self._cursor != delta.start_cursor:
            raise AdapterDataError("update delta must begin at the fitted history cursor")
        if self._series_keys != delta.series_keys:
            raise AdapterDataError("update delta series must match the fitted series")
        calendar = self._fit_calendar
        if calendar is None or calendar.phase is None:
            raise AdapterLifecycleError("update requires a fitted bound calendar")
        frame = delta.materialize()
        assert self._series_keys is not None
        self._validate_history_surface(frame, self._series_keys)
        origin = calendar.advance(calendar.phase, delta.end_cursor.time_bound)
        self._season_by_series = self._retain_at_origin(
            frame,
            series_keys=self._series_keys,
            origin=origin,
            previous=retained,
        )
        self._cursor = delta.end_cursor

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        """Emit a deterministic validated frame from retained seasonal phases."""
        retained = self._require_fitted()
        self._require_matching_config(task)
        if task.cursor != self._cursor or task.series_keys != self._series_keys:
            raise AdapterDataError("predict task must match the fitted history cursor and series")
        if self._fit_calendar is None or not task.calendar.shares_grid_with(self._fit_calendar):
            raise AdapterDataError("predict task calendar must match the fitted task")

        seasonal_timestamps = self._seasonal_timestamps(task)
        values_by_series: dict[str, dict[pd.Timestamp, float]] = {}
        for series_key in task.series_keys:
            values = dict(retained[series_key])
            missing = [timestamp for timestamp in seasonal_timestamps if timestamp not in values]
            if missing:
                missing_text = ", ".join(str(timestamp) for timestamp in missing)
                raise AdapterDataError(
                    f"series {series_key!r} requires one complete season of "
                    f"{self._season_length} observations; missing: {missing_text}"
                )
            values_by_series[series_key] = values

        output_series: list[str] = []
        output_targets: list[pd.Timestamp] = []
        output_points: list[float] = []
        output_steps: list[int] = []
        for series_key in task.series_keys:
            for horizon_step in range(1, task.horizon + 1):
                phase = (horizon_step - 1) % self._season_length
                output_series.append(series_key)
                output_targets.append(
                    target_timestamp(task.origin, horizon_step, calendar=task.calendar)
                )
                output_points.append(values_by_series[series_key][seasonal_timestamps[phase]])
                output_steps.append(horizon_step)

        row_count = len(output_series)
        frame = pd.DataFrame(
            {
                SERIES_KEY: pd.Series(output_series, dtype="string"),
                TARGET_TIMESTAMP: pd.Series(pd.to_datetime(output_targets)),
                ACTUAL_VALUE: pd.Series(np.full(row_count, np.nan), dtype="float64"),
                POINT_FORECAST: pd.Series(output_points, dtype="float64"),
                HORIZON_STEP: pd.Series(output_steps, dtype="int64"),
                ORIGIN: pd.Series(pd.to_datetime([task.origin] * row_count)),
                MODEL_NAME: pd.Series([self._model_name] * row_count, dtype="string"),
            }
        )
        return validate_forecast_frame(frame, calendar=task.calendar)

    def fitted_values(self) -> FittedValues:
        """Reject fitted-value collection because this adapter does not declare it."""
        self._raise_unsupported(AdapterCapability.FITTED_VALUES)

    def dump_state(self) -> bytes:
        """Serialize minimal fitted state through a deterministic native envelope."""
        retained = self._require_fitted()
        cursor = self._cursor
        calendar = self._fit_calendar
        series_keys = self._series_keys
        assert cursor is not None and calendar is not None and calendar.phase is not None
        assert series_keys is not None
        payload = {
            "calendar": {
                "frequency": calendar.frequency,
                "phase": _timestamp_record(calendar.phase),
            },
            "cursor": {
                "panel_identity": cursor.panel_identity,
                "series_start": cursor.series_start,
                "series_stop": cursor.series_stop,
                "time_bound": cursor.time_bound,
            },
            "model_name": self._model_name,
            "schema": _STATE_SCHEMA,
            "season_by_series": {
                key: [
                    {"timestamp": _timestamp_record(timestamp), "value": value}
                    for timestamp, value in retained[key]
                ]
                for key in series_keys
            },
            "season_length": self._season_length,
            "series_keys": list(series_keys),
        }
        try:
            return canonical_json_bytes(payload, path="seasonal-naive state")
        except CanonicalJsonError as error:
            raise AdapterDataError("seasonal state must contain finite values") from error

    def load_state(self, state: bytes) -> None:
        """Restore and validate minimal fitted state from native bytes."""
        if not isinstance(state, bytes):
            raise TypeError("seasonal state must be bytes")
        try:
            payload = json.loads(state)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterDataError("seasonal state is not canonical JSON") from error
        if (
            not isinstance(payload, dict)
            or canonical_json_bytes(
                payload,
                path="seasonal-naive state",
            )
            != state
        ):
            raise AdapterDataError("seasonal state is not canonical JSON")
        try:
            if payload.get("schema") != _STATE_SCHEMA:
                raise ValueError("schema")
            if payload.get("season_length") != self._season_length:
                raise ValueError("season length")
            if payload.get("model_name") != self._model_name:
                raise ValueError("model name")
            raw_calendar = _record(
                payload["calendar"],
                keys={"frequency", "phase"},
                name="calendar",
            )
            frequency = raw_calendar["frequency"]
            if not isinstance(frequency, str):
                raise ValueError("calendar frequency")
            calendar = Calendar(frequency).bind(_timestamp_from_record(raw_calendar["phase"]))
            raw_cursor = _record(
                payload["cursor"],
                keys={"panel_identity", "series_start", "series_stop", "time_bound"},
                name="cursor",
            )
            panel_identity = raw_cursor["panel_identity"]
            if not isinstance(panel_identity, str):
                raise ValueError("cursor panel identity")
            cursor_bounds = tuple(
                _integer(raw_cursor[name], name=f"cursor {name}")
                for name in ("series_start", "series_stop", "time_bound")
            )
            cursor = HistoryCursor(
                panel_identity,
                cursor_bounds[0],
                cursor_bounds[1],
                cursor_bounds[2],
            )
            raw_keys = payload["series_keys"]
            if not isinstance(raw_keys, list) or any(not isinstance(key, str) for key in raw_keys):
                raise ValueError("series keys")
            series_keys = tuple(raw_keys)
            raw_seasons = _record(
                payload["season_by_series"],
                keys=set(series_keys),
                name="season series",
            )
            if set(raw_seasons) != set(series_keys):
                raise ValueError("season series")
            retained = {key: _retained_season(raw_seasons[key]) for key in series_keys}
        except (KeyError, TypeError, ValueError) as error:
            raise AdapterDataError(f"seasonal state metadata is invalid: {error}") from error
        if cursor.series_stop - cursor.series_start != len(series_keys):
            raise AdapterDataError("seasonal state cursor does not match its series")
        self._fit_calendar = calendar
        self._cursor = cursor
        self._series_keys = series_keys
        self._season_by_series = retained

    @staticmethod
    def _requests_native_quantiles(model_config: Mapping[str, object]) -> bool:
        levels = model_config.get("quantile_levels")
        if levels is None:
            return False
        if isinstance(levels, (str, bytes)):
            return bool(levels)
        if isinstance(levels, Sized):
            return len(levels) > 0
        return True

    def _require_matching_config(self, task: ForecastTask) -> None:
        if self._effective_config(task.model_config) != self._config:
            raise AdapterConfigurationError(
                "task model configuration must match adapter construction configuration"
            )

    @classmethod
    def _effective_config(cls, model_config: Mapping[str, object]) -> _EffectiveConfig:
        if not isinstance(model_config, Mapping):
            raise AdapterConfigurationError("model configuration must be a mapping")
        backend = model_config.get("backend")
        if not isinstance(backend, str) or backend != SEASONAL_NAIVE_BACKEND:
            raise AdapterConfigurationError(
                f"model configuration backend must be {SEASONAL_NAIVE_BACKEND!r}"
            )
        season_length = model_config.get("m")
        if (
            not isinstance(season_length, Integral)
            or isinstance(season_length, bool)
            or season_length < 1
        ):
            raise AdapterConfigurationError("season length 'm' must be a positive integer")
        model_name = model_config.get("model_name", SEASONAL_NAIVE_BACKEND)
        if not isinstance(model_name, str) or not model_name:
            raise AdapterConfigurationError("model name must be a non-empty string")
        censoring_aware = model_config.get("censoring_aware", False)
        if not isinstance(censoring_aware, bool):
            raise AdapterConfigurationError("censoring_aware must be a boolean")
        refit_cadence = model_config.get("refit_cadence")
        if refit_cadence is not None and (
            not isinstance(refit_cadence, Integral)
            or isinstance(refit_cadence, bool)
            or refit_cadence < 1
        ):
            raise AdapterConfigurationError("refit cadence must be a positive integer")
        requested_capabilities: set[AdapterCapability] = set()
        if cls._requests_native_quantiles(model_config):
            requested_capabilities.add(AdapterCapability.NATIVE_QUANTILES)
        if censoring_aware:
            requested_capabilities.add(AdapterCapability.CENSORING_AWARE_FIT)
        return _EffectiveConfig(
            backend=backend,
            season_length=int(season_length),
            model_name=model_name,
            requested_capabilities=frozenset(requested_capabilities),
        )

    def _retain_at_origin(
        self,
        frame: pd.DataFrame,
        *,
        series_keys: tuple[str, ...],
        origin: pd.Timestamp,
        previous: dict[str, _RetainedSeason] | None,
    ) -> dict[str, _RetainedSeason]:
        season_start = self._calendar_for_retention().retreat(origin, self._season_length)
        updates: dict[str, dict[pd.Timestamp, float]] = {key: {} for key in series_keys}
        for raw_key, raw_timestamp, raw_value in zip(
            frame[SERIES_KEY],
            frame[HISTORY_TIMESTAMP],
            frame[HISTORY_VALUE],
            strict=True,
        ):
            if pd.isna(raw_value):
                continue
            timestamp = pd.Timestamp(raw_timestamp)
            if season_start <= timestamp < origin:
                updates[str(raw_key)][timestamp] = float(raw_value)
        retained: dict[str, _RetainedSeason] = {}
        for series_key in series_keys:
            values = {} if previous is None else dict(previous[series_key])
            values.update(updates[series_key])
            retained[series_key] = tuple(
                sorted(
                    (
                        (timestamp, value)
                        for timestamp, value in values.items()
                        if season_start <= timestamp < origin
                    ),
                    key=lambda item: item[0],
                )
            )
        return retained

    def _calendar_for_retention(self) -> Calendar:
        if self._fit_calendar is None:
            raise AdapterLifecycleError("seasonal retention requires a bound calendar")
        return self._fit_calendar

    @staticmethod
    def _validate_history_surface(
        history: pd.DataFrame,
        task_series_keys: tuple[str, ...],
    ) -> None:
        missing = [
            column
            for column in (SERIES_KEY, HISTORY_TIMESTAMP, HISTORY_VALUE)
            if column not in history.columns
        ]
        if missing:
            raise AdapterDataError(f"history is missing required columns: {', '.join(missing)}")
        raw_keys = history[SERIES_KEY].tolist()
        if any(not isinstance(key, str) or not key for key in raw_keys):
            raise AdapterDataError("history series keys must be non-empty strings")
        if not set(raw_keys) <= set(task_series_keys):
            raise AdapterDataError("history contains a series outside task.series_keys")
        value_dtype = history[HISTORY_VALUE].dtype
        if not is_numeric_dtype(value_dtype) or is_bool_dtype(value_dtype):
            raise AdapterDataError("history values must be numeric")

    def _seasonal_timestamps(self, task: ForecastTask) -> tuple[pd.Timestamp, ...]:
        season_start = task.calendar.retreat(task.origin, self._season_length)
        return tuple(
            task.calendar.advance(season_start, phase) for phase in range(self._season_length)
        )

    def _require_fitted(self) -> dict[str, _RetainedSeason]:
        if self._season_by_series is None:
            raise AdapterLifecycleError("operation requires a successful fit or load first")
        return self._season_by_series

    def _raise_unsupported(self, capability: AdapterCapability) -> Never:
        raise AdapterCapabilityError(
            f"backend {SEASONAL_NAIVE_BACKEND!r} does not declare capability {capability.value!r}"
        )


def _timestamp_record(timestamp: pd.Timestamp) -> dict[str, object]:
    return {
        "unit": timestamp.unit,
        "value": int(timestamp.asm8.astype("int64")),
    }


def _timestamp_from_record(value: object) -> pd.Timestamp:
    record = _record(value, keys={"unit", "value"}, name="timestamp")
    unit = record["unit"]
    raw = record["value"]
    if (
        not isinstance(unit, str)
        or unit not in {"s", "ms", "us", "ns"}
        or not isinstance(raw, int)
        or isinstance(raw, bool)
    ):
        raise ValueError("timestamp")
    return pd.Timestamp(raw, unit=unit)


def _record(value: object, *, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(name)
    record: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(name)
        record[key] = item
    return record


def _retained_season(value: object) -> _RetainedSeason:
    if not isinstance(value, list):
        raise ValueError("season rows")
    retained: list[tuple[pd.Timestamp, float]] = []
    for raw_record in value:
        record = _record(
            raw_record,
            keys={"timestamp", "value"},
            name="season row",
        )
        raw_value = record["value"]
        if (
            not isinstance(raw_value, (int, float))
            or isinstance(raw_value, bool)
            or not math.isfinite(raw_value)
        ):
            raise ValueError("season value")
        retained.append((_timestamp_from_record(record["timestamp"]), float(raw_value)))
    return tuple(retained)


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(name)
    return value


__all__ = ["HISTORY_VALUE", "SEASONAL_NAIVE_BACKEND", "SeasonalNaiveAdapter"]
