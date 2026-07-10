"""Forecast from the most recent complete seasonal phase lookup."""

from __future__ import annotations

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
    target_timestamp,
    validate_forecast_frame,
)
from newcalibre.forecasting.protocol import (
    AdapterCapability,
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterDataError,
    AdapterLifecycleError,
)

SEASONAL_NAIVE_BACKEND: Final = "seasonal-naive"
HISTORY_VALUE: Final = "value"

_RetainedSeason = tuple[tuple[pd.Timestamp, float], ...]


@dataclass(frozen=True, slots=True)
class _EffectiveConfig:
    backend: str
    season_length: int
    model_name: str
    requests_native_quantiles: bool
    requests_censoring_aware_fit: bool


class SeasonalNaiveAdapter:
    """Repeat the latest pre-origin seasonal phase for every horizon step.

    Fit retention is intentionally exact and small: for each series, retain
    only non-missing observations from the final ``m`` calendar periods before
    the fitted origin. Earlier history, task objects, and forecast rows are not
    retained. Missing phase observations therefore remain visible and fail
    loudly when prediction requires the complete season.
    """

    def __init__(self, model_config: Mapping[str, object]) -> None:
        config = self._effective_config(model_config)
        self._config = config
        self._season_length = config.season_length
        self._model_name = config.model_name
        self._fit_origin: pd.Timestamp | None = None
        self._fit_calendar: Calendar | None = None
        self._season_by_series: dict[str, _RetainedSeason] | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        """Declare that this library-free brick has no optional capabilities."""
        return frozenset()

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        """Retain only the final pre-origin season for each task series."""
        self._require_matching_config(task)
        if not isinstance(collect_fitted_values, bool):
            raise AdapterConfigurationError("collect_fitted_values must be a boolean")
        if collect_fitted_values:
            self._raise_unsupported(AdapterCapability.FITTED_VALUES)
        if self._config.requests_native_quantiles:
            self._raise_unsupported(AdapterCapability.NATIVE_QUANTILES)
        if self._config.requests_censoring_aware_fit:
            self._raise_unsupported(AdapterCapability.CENSORING_AWARE_FIT)

        retained = self._extract_retained_season(task)
        self._fit_origin = task.origin
        self._fit_calendar = task.calendar
        self._season_by_series = retained

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        """Emit a deterministic validated frame from retained seasonal phases."""
        retained = self._require_fitted()
        self._require_matching_config(task)
        if task.origin != self._fit_origin or task.calendar != self._fit_calendar:
            raise AdapterDataError("predict task origin and calendar must match the fitted task")

        predict_season = self._extract_retained_season(task)
        if predict_season != retained:
            raise AdapterDataError("predict task's retained season must match the fitted task")
        task_series = list(predict_season)

        seasonal_timestamps = self._seasonal_timestamps(task)
        values_by_series: dict[str, dict[pd.Timestamp, float]] = {}
        for series_key in task_series:
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
        for series_key in task_series:
            for horizon_step in range(1, task.horizon + 1):
                phase = (horizon_step - 1) % self._season_length
                output_series.append(series_key)
                output_targets.append(
                    target_timestamp(
                        task.origin,
                        horizon_step,
                        calendar=task.calendar,
                    )
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

    def fitted_values(self, task: ForecastTask) -> FittedValues:
        """Reject fitted-value collection because this adapter does not declare it."""
        del task
        self._raise_unsupported(AdapterCapability.FITTED_VALUES)

    def dump_state(self) -> bytes:
        """Reject persistence because this adapter does not declare it."""
        self._raise_unsupported(AdapterCapability.ARTIFACT_PERSISTENCE)

    def load_state(self, state: bytes) -> None:
        """Reject persistence because this adapter does not declare it."""
        del state
        self._raise_unsupported(AdapterCapability.ARTIFACT_PERSISTENCE)

    def update(self, task: ForecastTask) -> None:
        """Reject incremental update because this adapter does not declare it."""
        del task
        self._raise_unsupported(AdapterCapability.INCREMENTAL_UPDATE)

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
        return _EffectiveConfig(
            backend=backend,
            season_length=int(season_length),
            model_name=model_name,
            requests_native_quantiles=cls._requests_native_quantiles(model_config),
            requests_censoring_aware_fit=censoring_aware,
        )

    def _extract_retained_season(self, task: ForecastTask) -> dict[str, _RetainedSeason]:
        history = task.history
        series_keys = self._validate_history_surface(history, task.series_keys)
        seasonal_timestamp_set = set(self._seasonal_timestamps(task))
        relevant = history[history[HISTORY_TIMESTAMP].isin(seasonal_timestamp_set)]

        duplicate_lookup = relevant.duplicated(subset=[SERIES_KEY, HISTORY_TIMESTAMP])
        if duplicate_lookup.any():
            raise AdapterDataError("seasonal lookup requires unique series/timestamp rows")

        retained: dict[str, _RetainedSeason] = {}
        for series_key in series_keys:
            rows = relevant[relevant[SERIES_KEY] == series_key]
            observations: list[tuple[pd.Timestamp, float]] = []
            for timestamp, value in zip(rows[HISTORY_TIMESTAMP], rows[HISTORY_VALUE], strict=True):
                if pd.isna(value):
                    continue
                observations.append((pd.Timestamp(timestamp), float(value)))
            retained[series_key] = tuple(sorted(observations, key=lambda item: item[0]))
        return retained

    def _validate_history_surface(
        self, history: pd.DataFrame, task_series_keys: tuple[str, ...]
    ) -> tuple[str, ...]:
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
        return task_series_keys

    def _seasonal_timestamps(self, task: ForecastTask) -> tuple[pd.Timestamp, ...]:
        season_start = task.calendar.retreat(task.origin, self._season_length)
        return tuple(
            task.calendar.advance(season_start, phase) for phase in range(self._season_length)
        )

    def _require_fitted(self) -> dict[str, _RetainedSeason]:
        if self._season_by_series is None:
            raise AdapterLifecycleError("predict requires a successful fit first")
        return self._season_by_series

    def _raise_unsupported(self, capability: AdapterCapability) -> Never:
        raise AdapterCapabilityError(
            f"backend {SEASONAL_NAIVE_BACKEND!r} does not declare capability {capability.value!r}"
        )
