"""Define immutable, exactly serializable panel-derived forecast tasks."""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa

from newcalibre.domain.calendar import Calendar, CalendarError
from newcalibre.domain.forecast_frame import SERIES_KEY
from newcalibre.domain.panel import (
    TIMESTAMP,
    Scope,
    _canonicalize_future_exogenous,
    _canonicalize_panel_frame,
)

HISTORY_TIMESTAMP: Final = TIMESTAMP

_MAGIC = b"NCFT"
_FORMAT_VERSION = 1
_PREFIX = struct.Struct(">4sBQQQ")
_DIGEST_SIZE = hashlib.sha256().digest_size


class ForecastTaskError(ValueError):
    """Report an invalid task or serialized-task envelope."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ForecastTask:
    """Carry one immutable, task-closed fit-and-predict request."""

    _history: pd.DataFrame = field(repr=False)
    _future_exogenous: pd.DataFrame | None = field(repr=False)
    _horizon: int
    _origin: pd.Timestamp
    _calendar: Calendar
    _model_config: dict[str, object] = field(repr=False)
    _scope: Scope
    _series_keys: tuple[str, ...]
    _serialized: bytes = field(repr=False)

    @classmethod
    def _from_components(
        cls,
        *,
        history: pd.DataFrame,
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
        if not isinstance(scope, Scope):
            raise ForecastTaskError("scope must be a Scope")
        if (
            not series_keys
            or len(set(series_keys)) != len(series_keys)
            or any(not isinstance(key, str) or not key for key in series_keys)
            or tuple(sorted(series_keys, key=str.encode)) != series_keys
        ):
            raise ForecastTaskError(
                "task series keys must be unique non-empty strings in UTF-8 byte order"
            )
        if scope is Scope.LOCAL and len(series_keys) != 1:
            raise ForecastTaskError("a local task must contain exactly one series")

        try:
            normalized_history = _canonicalize_panel_frame(
                history, calendar=calendar, allow_empty=True
            )
        except ValueError as error:
            raise ForecastTaskError(str(error)) from error
        history_keys = set(normalized_history[SERIES_KEY])
        if not history_keys <= set(series_keys):
            raise ForecastTaskError("task history contains a series outside task.series_keys")
        if not normalized_history[TIMESTAMP].lt(origin).all():
            raise ForecastTaskError("every history timestamp must be strictly before origin")

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
        object.__setattr__(instance, "_history", normalized_history)
        object.__setattr__(instance, "_future_exogenous", normalized_future)
        object.__setattr__(instance, "_horizon", int(horizon))
        object.__setattr__(instance, "_origin", origin)
        object.__setattr__(instance, "_calendar", calendar)
        object.__setattr__(instance, "_model_config", normalized_config)
        object.__setattr__(instance, "_scope", scope)
        object.__setattr__(instance, "_series_keys", series_keys)
        object.__setattr__(instance, "_serialized", _encode_task(instance))
        return instance

    @staticmethod
    def _require_horizon(horizon: object) -> None:
        if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
            raise ForecastTaskError("horizon must be a positive integer")

    @property
    def history(self) -> pd.DataFrame:
        """Return a defensive copy of canonical, strictly pre-origin history."""
        return self._history.copy(deep=True)

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
        """Return an isolated copy of the finite JSON model configuration."""
        return json.loads(_canonical_json(self._model_config))

    @property
    def scope(self) -> Scope:
        """Return the construction-time scope decision."""
        return self._scope

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return the fixed task enumeration in UTF-8 byte order."""
        return self._series_keys

    def to_bytes(self) -> bytes:
        """Return a versioned deterministic non-pickle task envelope."""
        return self._serialized

    @classmethod
    def from_bytes(cls, data: bytes) -> ForecastTask:
        """Materialize an exact task or reject the whole envelope."""
        if not isinstance(data, bytes):
            raise ForecastTaskError("serialized task must be bytes")
        minimum = _PREFIX.size + _DIGEST_SIZE
        if len(data) < minimum:
            raise ForecastTaskError("serialized task is truncated")
        try:
            magic, version, header_size, history_size, future_size = _PREFIX.unpack_from(data)
        except struct.error as error:
            raise ForecastTaskError("serialized task has an invalid prefix") from error
        if magic != _MAGIC:
            raise ForecastTaskError("serialized task has an invalid magic value")
        if version != _FORMAT_VERSION:
            raise ForecastTaskError(f"unsupported serialized task version: {version}")

        expected_size = minimum + header_size + history_size + future_size
        if len(data) != expected_size:
            relation = "trailing bytes" if len(data) > expected_size else "truncated"
            raise ForecastTaskError(f"serialized task contains {relation}")
        payload = data[:-_DIGEST_SIZE]
        if not hmac.compare_digest(hashlib.sha256(payload).digest(), data[-_DIGEST_SIZE:]):
            raise ForecastTaskError("serialized task failed its integrity digest")

        cursor = _PREFIX.size
        header_bytes = data[cursor : cursor + header_size]
        cursor += header_size
        history_bytes = data[cursor : cursor + history_size]
        cursor += history_size
        future_bytes = data[cursor : cursor + future_size]
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ForecastTaskError("serialized task header is not canonical JSON") from error
        if not isinstance(header, dict) or set(header) != {
            "calendar",
            "future_schema",
            "history_schema",
            "horizon",
            "model_config",
            "origin",
            "scope",
            "series_keys",
        }:
            raise ForecastTaskError("serialized task header schema is unsupported")
        if _canonical_json(header).encode() != header_bytes:
            raise ForecastTaskError("serialized task header is not canonical JSON")

        history = _arrow_to_frame(history_bytes, surface="history")
        future = None
        if future_size:
            future = _arrow_to_frame(future_bytes, surface="future exogenous")
        if _frame_schema(history) != header["history_schema"]:
            raise ForecastTaskError("serialized task history schema drifted")
        if (None if future is None else _frame_schema(future)) != header["future_schema"]:
            raise ForecastTaskError("serialized task future-exogenous schema drifted")
        try:
            raw_origin = header["origin"]
            if (
                not isinstance(raw_origin, dict)
                or set(raw_origin) != {"unit", "value"}
                or raw_origin["unit"] not in {"s", "ms", "us", "ns"}
                or not isinstance(raw_origin["value"], int)
                or isinstance(raw_origin["value"], bool)
            ):
                raise TypeError
            origin = pd.Timestamp(np.datetime64(raw_origin["value"], raw_origin["unit"]))
            calendar = Calendar(header["calendar"])
            scope = Scope(header["scope"])
            raw_keys = header["series_keys"]
            if not isinstance(raw_keys, list) or any(not isinstance(key, str) for key in raw_keys):
                raise TypeError
            task = cls._from_components(
                history=history,
                future_exogenous=future,
                horizon=header["horizon"],
                origin=origin,
                calendar=calendar,
                model_config=header["model_config"],
                scope=scope,
                series_keys=tuple(raw_keys),
            )
        except (CalendarError, ForecastTaskError, TypeError, ValueError) as error:
            raise ForecastTaskError(f"serialized task metadata is invalid: {error}") from error
        if task.to_bytes() != data:
            raise ForecastTaskError("serialized task is not in canonical exact form")
        return task


def _canonical_model_config(model_config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(model_config, Mapping):
        raise ForecastTaskError("model configuration must be a mapping")
    candidate = dict(model_config)
    _require_json_value(candidate, path="model configuration")
    try:
        return json.loads(_canonical_json(candidate))
    except (TypeError, ValueError) as error:
        raise ForecastTaskError("model configuration must contain finite JSON values") from error


def _require_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ForecastTaskError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ForecastTaskError(f"{path} contains a non-string object key")
            _require_json_value(item, path=f"{path}.{key}")
        return
    raise ForecastTaskError(f"{path} contains non-JSON value {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _frame_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"dtype": str(frame[column].dtype), "name": column} for column in frame.columns]


def _frame_to_arrow(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _arrow_to_frame(data: bytes, *, surface: str) -> pd.DataFrame:
    try:
        with pa.ipc.open_stream(data) as reader:
            return reader.read_all().to_pandas()
    except (pa.ArrowException, OSError, TypeError, ValueError) as error:
        raise ForecastTaskError(f"serialized task {surface} Arrow stream is invalid") from error


def _encode_task(task: ForecastTask) -> bytes:
    history = _frame_to_arrow(task._history)
    future = b"" if task._future_exogenous is None else _frame_to_arrow(task._future_exogenous)
    header = _canonical_json(
        {
            "calendar": task._calendar.frequency,
            "future_schema": (
                None if task._future_exogenous is None else _frame_schema(task._future_exogenous)
            ),
            "history_schema": _frame_schema(task._history),
            "horizon": task._horizon,
            "model_config": task._model_config,
            "origin": {
                "unit": task._origin.unit,
                "value": int(task._origin.asm8.astype("int64")),
            },
            "scope": task._scope.value,
            "series_keys": list(task._series_keys),
        }
    ).encode()
    prefix = _PREFIX.pack(_MAGIC, _FORMAT_VERSION, len(header), len(history), len(future))
    payload = prefix + header + history + future
    return payload + hashlib.sha256(payload).digest()
