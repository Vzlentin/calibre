"""Define immutable, exactly serializable panel-derived forecast tasks."""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from newcalibre.domain._canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
)
from newcalibre.domain._canonical_json import (
    canonical_json as _canonical_json,
)
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
_DATETIME_UNITS = frozenset({"s", "ms", "us", "ns"})
_STRING_DTYPE_TOKEN = "string[pyarrow]"
_TRANSPORT_STRING_DTYPE = pd.StringDtype(storage="pyarrow")
_SUPPORTED_NUMERIC_TOKENS = frozenset(
    {
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
    }
)
_NULLABLE_NUMERIC_DTYPES = {
    f"nullable:{name}": pd.api.types.pandas_dtype(name)
    for name in (
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
    )
}
_NULLABLE_NUMPY_DTYPES = {
    token: np.dtype(name.lower())
    for token, name in (
        ("nullable:Int8", "int8"),
        ("nullable:Int16", "int16"),
        ("nullable:Int32", "int32"),
        ("nullable:Int64", "int64"),
        ("nullable:UInt8", "uint8"),
        ("nullable:UInt16", "uint16"),
        ("nullable:UInt32", "uint32"),
        ("nullable:UInt64", "uint64"),
        ("nullable:Float32", "float32"),
        ("nullable:Float64", "float64"),
    )
}
_ARROW_NUMERIC_DTYPES = {
    f"arrow:{arrow_type!s}": pd.ArrowDtype(arrow_type)
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
        if origin.unit not in _DATETIME_UNITS:
            raise ForecastTaskError("origin must use timestamp resolution s, ms, us, or ns")
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
            normalized_history, _ = _canonicalize_panel_frame(
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
        """Return an isolated copy of adapter configuration, excluding engine scope."""
        return json.loads(_canonical_json(self._model_config))

    @property
    def scope(self) -> Scope:
        """Return the construction-time scope decision."""
        return self._scope

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return the fixed operational enumeration in UTF-8 byte order.

        Consumers that derive identity from the participating series must
        canonicalize the set independently. Presentation or processing order
        is never itself an identity hash input.
        """
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

        history = _arrow_to_frame(
            history_bytes,
            schema=header["history_schema"],
            surface="history",
        )
        future = None
        if future_size:
            future = _arrow_to_frame(
                future_bytes,
                schema=header["future_schema"],
                surface="future exogenous",
            )
        if _frame_schema(history) != header["history_schema"]:
            raise ForecastTaskError("serialized task history schema drifted")
        if (None if future is None else _frame_schema(future)) != header["future_schema"]:
            raise ForecastTaskError("serialized task future-exogenous schema drifted")
        try:
            origin = _timestamp_from_record(header["origin"], name="origin")
            raw_calendar = header["calendar"]
            if not isinstance(raw_calendar, dict) or set(raw_calendar) != {
                "frequency",
                "phase",
            }:
                raise TypeError
            calendar = Calendar(raw_calendar["frequency"])
            raw_phase = raw_calendar["phase"]
            if raw_phase is not None:
                calendar = calendar.bind(_timestamp_from_record(raw_phase, name="calendar phase"))
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
    if "scope" in candidate:
        raise ForecastTaskError(
            "scope is engine configuration and cannot appear in model configuration"
        )
    try:
        canonical = canonical_json_bytes(candidate, path="model configuration")
        return json.loads(canonical)
    except CanonicalJsonError as error:
        raise ForecastTaskError("model configuration must contain finite JSON values") from error


def _frame_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"dtype": _dtype_token(frame[column].dtype), "name": column} for column in frame.columns
    ]


def _dtype_token(dtype: object) -> str:
    if isinstance(dtype, pd.StringDtype) and dtype.storage == "pyarrow":
        return _STRING_DTYPE_TOKEN
    nullable_token = f"nullable:{dtype!s}"
    expected_nullable = _NULLABLE_NUMERIC_DTYPES.get(nullable_token)
    if expected_nullable is not None and type(dtype) is type(expected_nullable):
        return nullable_token
    if isinstance(dtype, pd.ArrowDtype):
        arrow_token = f"arrow:{dtype.pyarrow_dtype!s}"
        expected_arrow = _ARROW_NUMERIC_DTYPES.get(arrow_token)
        if expected_arrow is not None and expected_arrow == dtype:
            return arrow_token
    if isinstance(dtype, np.dtype):
        return str(dtype)
    raise ForecastTaskError(f"task frame contains unsupported dtype {dtype!s}")


def _frame_to_arrow(frame: pd.DataFrame) -> bytes:
    try:
        source = pa.Table.from_pandas(frame, preserve_index=False)
        arrays = [
            _canonical_arrow_array(
                pa.concat_arrays(column.chunks)
                if column.num_chunks
                else pa.array([], type=column.type)
            )
            for column in source.columns
        ]
        table = pa.Table.from_arrays(
            arrays,
            names=source.column_names,
        )
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()
    except (pa.ArrowException, OverflowError, TypeError, ValueError) as error:
        raise ForecastTaskError("task frame cannot be encoded as Arrow") from error


def _canonical_arrow_array(array: pa.Array) -> pa.Array:
    if not array.null_count:
        return array
    if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        mask = array.is_null().to_numpy(zero_copy_only=False)
        values = array.fill_null(0).to_numpy(zero_copy_only=False)
        return pa.array(values, mask=mask, type=array.type, from_pandas=False)
    return pa.array(array.to_pylist(), type=array.type)


def _arrow_to_frame(
    data: bytes,
    *,
    schema: object,
    surface: str,
) -> pd.DataFrame:
    manifest = _validate_frame_schema(schema, surface=surface)
    try:
        with pa.ipc.open_stream(data) as reader:
            table = reader.read_all()
    except (pa.ArrowException, OSError, TypeError, ValueError) as error:
        raise ForecastTaskError(f"serialized task {surface} Arrow stream is invalid") from error
    if table.schema.metadata is not None:
        raise ForecastTaskError(f"serialized task {surface} contains undeclared Arrow metadata")
    if table.column_names != [entry["name"] for entry in manifest]:
        raise ForecastTaskError(f"serialized task {surface} schema drifted")
    for arrow_field, entry in zip(table.schema, manifest, strict=True):
        if arrow_field.type != _arrow_type(entry["dtype"]):
            raise ForecastTaskError(f"serialized task {surface} schema drifted")
    try:
        columns = {
            entry["name"]: _arrow_column_to_series(
                table.column(entry["name"]),
                token=entry["dtype"],
            )
            for entry in manifest
        }
        frame = pd.DataFrame(columns)
    except (pa.ArrowException, OverflowError, TypeError, ValueError) as error:
        raise ForecastTaskError(f"serialized task {surface} schema cannot be restored") from error
    frame = frame.set_flags(allows_duplicate_labels=True)
    frame.attrs = {}
    frame.index.name = None
    frame.columns.name = None
    return frame


def _arrow_column_to_series(column: pa.ChunkedArray, *, token: str) -> pd.Series:
    array = pa.concat_arrays(column.chunks) if column.num_chunks else pa.array([], type=column.type)
    if token == _STRING_DTYPE_TOKEN:
        return pd.Series(array.to_pylist(), dtype=_TRANSPORT_STRING_DTYPE)
    arrow_dtype = _ARROW_NUMERIC_DTYPES.get(token)
    if arrow_dtype is not None:
        return pd.Series(array.to_pylist(), dtype=arrow_dtype)
    nullable_dtype = _NULLABLE_NUMERIC_DTYPES.get(token)
    if nullable_dtype is not None:
        numpy_dtype = _NULLABLE_NUMPY_DTYPES[token]
        mask = array.is_null().to_numpy(zero_copy_only=False)
        values = array.fill_null(0).to_numpy(zero_copy_only=False).astype(numpy_dtype)
        if numpy_dtype.kind == "f":
            valid_nan = np.isnan(values) & ~mask
            values[valid_nan] = np.nan
            extension = pd.arrays.FloatingArray(values, mask)
        else:
            extension = pd.arrays.IntegerArray(values, mask)
        return pd.Series(extension, dtype=nullable_dtype)
    dtype = np.dtype(token)
    if dtype.kind in "iu" and array.null_count:
        raise ForecastTaskError("serialized native integer column contains missing values")
    return pd.Series(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _validate_frame_schema(schema: object, *, surface: str) -> list[dict[str, str]]:
    if not isinstance(schema, list):
        raise ForecastTaskError(f"serialized task {surface} schema is invalid")
    manifest: list[dict[str, str]] = []
    names: set[str] = set()
    for entry in schema:
        if not isinstance(entry, dict):
            raise ForecastTaskError(f"serialized task {surface} schema is invalid")
        raw_entry = cast(dict[object, object], entry)
        dtype = raw_entry.get("dtype")
        name = raw_entry.get("name")
        if (
            set(raw_entry) != {"dtype", "name"}
            or not isinstance(dtype, str)
            or not isinstance(name, str)
            or name in names
        ):
            raise ForecastTaskError(f"serialized task {surface} schema is invalid")
        _arrow_type(dtype)
        names.add(name)
        manifest.append({"dtype": dtype, "name": name})
    return manifest


def _arrow_type(token: str) -> pa.DataType:
    if token == _STRING_DTYPE_TOKEN:
        return pa.large_string()
    if token.startswith("datetime64["):
        unit = token.removeprefix("datetime64[").removesuffix("]")
        if token != f"datetime64[{unit}]" or unit not in _DATETIME_UNITS:
            raise ForecastTaskError(f"serialized task schema dtype {token!r} is unsupported")
        return pa.timestamp(unit)
    nullable_numpy = _NULLABLE_NUMPY_DTYPES.get(token)
    if nullable_numpy is not None:
        return pa.from_numpy_dtype(nullable_numpy)
    arrow_dtype = _ARROW_NUMERIC_DTYPES.get(token)
    if arrow_dtype is not None:
        return arrow_dtype.pyarrow_dtype
    if token not in _SUPPORTED_NUMERIC_TOKENS:
        raise ForecastTaskError(f"serialized task schema dtype {token!r} is unsupported")
    try:
        dtype = np.dtype(token)
        if not dtype.isnative or str(dtype) != token:
            raise TypeError
        return pa.from_numpy_dtype(dtype)
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise ForecastTaskError(f"serialized task schema dtype {token!r} is unsupported") from error


def _timestamp_record(timestamp: pd.Timestamp) -> dict[str, int | str]:
    if timestamp.unit not in _DATETIME_UNITS:
        raise ForecastTaskError("timestamp resolution must be s, ms, us, or ns")
    return {
        "unit": timestamp.unit,
        "value": int(timestamp.asm8.astype("int64")),
    }


def _timestamp_from_record(record: object, *, name: str) -> pd.Timestamp:
    if not isinstance(record, dict):
        raise TypeError(f"{name} representation is invalid")
    raw_record = cast(dict[object, object], record)
    unit = raw_record.get("unit")
    value = raw_record.get("value")
    if (
        set(raw_record) != {"unit", "value"}
        or not isinstance(unit, str)
        or unit not in _DATETIME_UNITS
        or not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(f"{name} representation is invalid")
    try:
        return pd.Timestamp(value, unit=unit)
    except (OverflowError, ValueError) as error:
        raise TypeError(f"{name} representation is invalid") from error


def _encode_task(task: ForecastTask) -> bytes:
    history = _frame_to_arrow(task._history)
    future = b"" if task._future_exogenous is None else _frame_to_arrow(task._future_exogenous)
    header = _canonical_json(
        {
            "calendar": {
                "frequency": task._calendar.frequency,
                "phase": (
                    None
                    if task._calendar.phase is None
                    else _timestamp_record(task._calendar.phase)
                ),
            },
            "future_schema": (
                None if task._future_exogenous is None else _frame_schema(task._future_exogenous)
            ),
            "history_schema": _frame_schema(task._history),
            "horizon": task._horizon,
            "model_config": task._model_config,
            "origin": _timestamp_record(task._origin),
            "scope": task._scope.value,
            "series_keys": list(task._series_keys),
        }
    ).encode()
    prefix = _PREFIX.pack(_MAGIC, _FORMAT_VERSION, len(header), len(history), len(future))
    payload = prefix + header + history + future
    return payload + hashlib.sha256(payload).digest()
