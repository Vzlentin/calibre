"""Define immutable forecast tasks over opaque indexed-history views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import pandas as pd

from newcalibre.domain._canonical_json import (
    CanonicalJsonError,
    canonical_json,
    canonical_json_bytes,
)
from newcalibre.domain.calendar import Calendar, CalendarError
from newcalibre.domain.history import HistoryCursor, HistoryDelta, HistoryView
from newcalibre.domain.panel import (
    TIMESTAMP,
    Scope,
    _canonicalize_future_exogenous,
)

HISTORY_TIMESTAMP = TIMESTAMP
_DATETIME_UNITS = frozenset({"s", "ms", "us", "ns"})


class ForecastTaskError(ValueError):
    """Report an invalid indexed forecast task."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ForecastTask:
    """Carry one immutable fit-and-predict request over staged history."""

    _history: HistoryView = field(repr=False)
    _delta: HistoryDelta = field(repr=False)
    _cursor: HistoryCursor
    _future_exogenous: pd.DataFrame | None = field(repr=False)
    _horizon: int
    _origin: pd.Timestamp
    _calendar: Calendar
    _model_config: dict[str, object] = field(repr=False)
    _scope: Scope
    _series_keys: tuple[str, ...]
    _identity: str

    @classmethod
    def _from_components(
        cls,
        *,
        history: HistoryView,
        delta: HistoryDelta,
        cursor: HistoryCursor,
        future_exogenous: pd.DataFrame | None,
        horizon: int,
        origin: pd.Timestamp,
        calendar: Calendar,
        model_config: Mapping[str, object],
        scope: Scope,
        series_keys: tuple[str, ...],
    ) -> ForecastTask:
        cls._require_horizon(horizon)
        if not isinstance(calendar, Calendar):
            raise ForecastTaskError("calendar must be a Calendar")
        try:
            calendar.require_member(origin, name="origin")
        except CalendarError as error:
            raise ForecastTaskError(str(error)) from error
        if origin.unit not in _DATETIME_UNITS:
            raise ForecastTaskError("origin must use timestamp resolution s, ms, us, or ns")
        if not isinstance(scope, Scope):
            raise ForecastTaskError("scope must be a Scope")
        _require_series_keys(series_keys)
        if not isinstance(history, HistoryView):
            raise TypeError("task history must be a HistoryView")
        if not isinstance(delta, HistoryDelta):
            raise TypeError("task delta must be a HistoryDelta")
        if not isinstance(cursor, HistoryCursor):
            raise TypeError("task cursor must be a HistoryCursor")
        if history.cursor != cursor or delta.end_cursor != cursor:
            raise ForecastTaskError("task history, delta, and cursor must share one end cursor")
        if history.series_keys != series_keys or delta.series_keys != series_keys:
            raise ForecastTaskError("task history and delta must exactly match task series keys")

        try:
            normalized_future = _canonicalize_future_exogenous(
                future_exogenous,
                calendar=calendar,
                origin=origin,
                horizon=int(horizon),
                series_keys=series_keys,
            )
        except ValueError as error:
            raise ForecastTaskError(str(error)) from error
        normalized_config = _canonical_model_config(model_config)

        instance = object.__new__(cls)
        object.__setattr__(instance, "_history", history)
        object.__setattr__(instance, "_delta", delta)
        object.__setattr__(instance, "_cursor", cursor)
        object.__setattr__(instance, "_future_exogenous", normalized_future)
        object.__setattr__(instance, "_horizon", int(horizon))
        object.__setattr__(instance, "_origin", origin)
        object.__setattr__(instance, "_calendar", calendar)
        object.__setattr__(instance, "_model_config", normalized_config)
        object.__setattr__(instance, "_scope", scope)
        object.__setattr__(instance, "_series_keys", series_keys)
        object.__setattr__(instance, "_identity", _task_identity(instance))
        return instance

    @staticmethod
    def _require_horizon(horizon: object) -> None:
        if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
            raise ForecastTaskError("horizon must be a positive integer")

    @property
    def history(self) -> HistoryView:
        """Return the opaque staged-history view."""
        return self._history

    @property
    def delta(self) -> HistoryDelta:
        """Return only history newly visible after the prior cursor."""
        return self._delta

    @property
    def cursor(self) -> HistoryCursor:
        """Return the exclusive staged-history cursor at this origin."""
        return self._cursor

    @property
    def identity(self) -> str:
        """Return the canonical task identity used for checkpoint binding."""
        return self._identity

    @property
    def future_exogenous(self) -> pd.DataFrame | None:
        """Return a defensive copy of facts proven known by the origin."""
        if self._future_exogenous is None:
            return None
        return self._future_exogenous.copy(deep=True)

    @property
    def horizon(self) -> int:
        """Return the positive forecast horizon."""
        return self._horizon

    @property
    def origin(self) -> pd.Timestamp:
        """Return the on-calendar forecast origin."""
        return self._origin

    @property
    def calendar(self) -> Calendar:
        """Return the task's panel-wide calendar."""
        return self._calendar

    @property
    def model_config(self) -> Mapping[str, object]:
        """Return isolated adapter configuration, excluding engine scope."""
        return json.loads(canonical_json(self._model_config))

    @property
    def scope(self) -> Scope:
        """Return the construction-time scope decision."""
        return self._scope

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return the fixed canonical series enumeration."""
        return self._series_keys


def _canonical_model_config(model_config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(model_config, Mapping):
        raise ForecastTaskError("model configuration must be a mapping")
    candidate = dict(model_config)
    if "scope" in candidate:
        raise ForecastTaskError(
            "scope is engine configuration and cannot appear in model configuration"
        )
    try:
        canonical = canonical_json_bytes(candidate, path="model configuration")
        return json.loads(canonical)
    except CanonicalJsonError as error:
        raise ForecastTaskError("model configuration must contain finite JSON values") from error


def _require_series_keys(series_keys: tuple[str, ...]) -> None:
    if (
        not isinstance(series_keys, tuple)
        or not series_keys
        or len(set(series_keys)) != len(series_keys)
        or any(not isinstance(key, str) or not key for key in series_keys)
        or tuple(sorted(series_keys, key=str.encode)) != series_keys
    ):
        raise ForecastTaskError(
            "task series keys must be unique non-empty strings in UTF-8 byte order"
        )


def _task_identity(task: ForecastTask) -> str:
    future_digest = None
    if task._future_exogenous is not None:
        digest = hashlib.sha256()
        digest.update(
            canonical_json_bytes(
                [
                    {"dtype": str(task._future_exogenous[column].dtype), "name": column}
                    for column in task._future_exogenous.columns
                ],
                path="forecast task future-exogenous schema",
            )
        )
        digest.update(
            pd.util.hash_pandas_object(
                task._future_exogenous,
                index=False,
                categorize=True,
            )
            .to_numpy(copy=False)
            .tobytes()
        )
        future_digest = digest.hexdigest()
    payload = canonical_json_bytes(
        {
            "calendar_frequency": task._calendar.frequency,
            "cursor": {
                "panel_identity": task._cursor.panel_identity,
                "series_start": task._cursor.series_start,
                "series_stop": task._cursor.series_stop,
                "time_bound": task._cursor.time_bound,
            },
            "future_exogenous_digest": future_digest,
            "horizon": task._horizon,
            "model_config": task._model_config,
            "origin": {
                "unit": task._origin.unit,
                "value": int(task._origin.asm8.astype("int64")),
            },
            "scope": task._scope.value,
            "series_keys": list(task._series_keys),
        },
        path="forecast task identity",
    )
    return hashlib.sha256(b"newcalibre.forecast-task/v2\0" + payload).hexdigest()


__all__ = ["HISTORY_TIMESTAMP", "ForecastTask", "ForecastTaskError"]
