"""Define bounded immutable reads over a closed logical forecast ledger."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

import pandas as pd

from newcalibre.domain import (
    CensoringAssertion,
    GuaranteeClaim,
    GuaranteeDescriptor,
    SessionIdentity,
)


class LedgerColumn(StrEnum):
    """Name every selectable logical forecast-ledger column."""

    SERIES_KEY = "series_key"
    ORIGIN = "origin"
    HORIZON_STEP = "horizon_step"
    MODEL_NAME = "model_name"
    TARGET_TIMESTAMP = "target_timestamp"
    POINT_FORECAST = "point_forecast"
    ISSUANCES = "issuances"
    RESOLUTION = "resolution"
    SCORES = "scores"


_COLUMN_NAMES = frozenset(column.value for column in LedgerColumn)


@dataclass(frozen=True, slots=True)
class LedgerForecastKey:
    """Identify one logical forecast item without exposing its stored row."""

    series_key: str
    origin: pd.Timestamp
    horizon_step: int
    model_name: str

    def __post_init__(self) -> None:
        _require_identifier(self.series_key, name="ledger forecast series key")
        _require_timestamp(self.origin, name="ledger forecast origin")
        step = _positive_integer(self.horizon_step, name="ledger forecast horizon step")
        _require_identifier(self.model_name, name="ledger forecast model name")
        object.__setattr__(self, "horizon_step", step)


@dataclass(frozen=True, slots=True)
class LedgerBoundIssuance:
    """Project one issued bound and its immutable forecast payload values."""

    bound_key: tuple[str, ...]
    bound_values: tuple[float | None, ...]
    descriptor: GuaranteeDescriptor
    guaranteed_side: str | None
    calibration_ready: bool
    bounds_finite: bool
    bounds_null_reason: str | None

    def __post_init__(self) -> None:
        key = _bound_key(self.bound_key)
        if isinstance(self.bound_values, (str, bytes)):
            raise TypeError("ledger issuance bound values must be a sequence")
        try:
            raw_values = tuple(self.bound_values)
        except TypeError as error:
            raise TypeError("ledger issuance bound values must be a sequence") from error
        if len(raw_values) != len(key):
            raise ValueError("ledger issuance bound values must align with its bound key")
        values = tuple(
            _optional_finite_real(value, name="issued bound value") for value in raw_values
        )
        if not isinstance(self.descriptor, GuaranteeDescriptor):
            raise TypeError("ledger issuance descriptor must be a GuaranteeDescriptor")
        side = self.guaranteed_side
        if side is not None and side not in {"lower", "upper"}:
            raise ValueError("ledger issuance guaranteed side must be lower, upper, or omitted")
        if not isinstance(self.calibration_ready, bool):
            raise TypeError("ledger issuance calibration readiness must be a boolean")
        if not isinstance(self.bounds_finite, bool):
            raise TypeError("ledger issuance bounds finiteness must be a boolean")
        if self.bounds_finite:
            if any(value is None for value in values):
                raise ValueError("finite ledger issuance bounds must contain finite values")
            if self.bounds_null_reason is not None:
                raise ValueError("finite ledger issuance bounds cannot have a null reason")
        else:
            if any(value is not None for value in values):
                raise ValueError("non-finite ledger issuance bounds must contain null values")
            _require_text(self.bounds_null_reason, name="ledger issuance bounds null reason")
        one_sided = self.descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
        if one_sided != (side is not None):
            raise ValueError("only one-sided ledger issuances must declare a guaranteed side")
        object.__setattr__(self, "bound_key", key)
        object.__setattr__(self, "bound_values", values)


@dataclass(frozen=True, slots=True)
class LedgerObservationAnnotation:
    """Project one persisted observe annotation without its repeated row key."""

    score: float | None
    exclusion_cause: str | None
    advanced_delivered_score: bool

    def __post_init__(self) -> None:
        has_score = self.score is not None
        has_cause = self.exclusion_cause is not None
        if has_score == has_cause:
            raise ValueError("a ledger annotation requires exactly one score or exclusion cause")
        if has_score:
            object.__setattr__(
                self,
                "score",
                _finite_real(self.score, name="ledger annotation score"),
            )
        else:
            _require_text(self.exclusion_cause, name="ledger annotation exclusion cause")
        if not isinstance(self.advanced_delivered_score, bool):
            raise TypeError("ledger annotation advancement must be a boolean")
        if self.advanced_delivered_score and not has_score:
            raise ValueError("only a scored ledger annotation can advance delivered score")


@dataclass(frozen=True, slots=True)
class LedgerResolution:
    """Project one persisted forecast resolution and its observe annotation."""

    target_timestamp: pd.Timestamp
    actual_value: float
    censoring_assertion: CensoringAssertion | None
    availability_bound: float | None
    annotation: LedgerObservationAnnotation | None

    def __post_init__(self) -> None:
        _require_timestamp(self.target_timestamp, name="ledger resolution target timestamp")
        object.__setattr__(
            self,
            "actual_value",
            _finite_real(self.actual_value, name="ledger resolution actual value"),
        )
        if self.censoring_assertion is not None and not isinstance(
            self.censoring_assertion,
            CensoringAssertion,
        ):
            raise TypeError(
                "ledger resolution censoring assertion must be a CensoringAssertion or omitted"
            )
        if self.availability_bound is not None:
            object.__setattr__(
                self,
                "availability_bound",
                _finite_real(
                    self.availability_bound,
                    name="ledger resolution availability bound",
                ),
            )
        if self.annotation is not None and not isinstance(
            self.annotation,
            LedgerObservationAnnotation,
        ):
            raise TypeError(
                "ledger resolution annotation must be a LedgerObservationAnnotation or omitted"
            )


@dataclass(frozen=True, slots=True)
class LedgerBoundScore:
    """Project one registered per-bound score outcome for a forecast item."""

    bound_key: tuple[str, ...]
    descriptor: GuaranteeDescriptor
    guaranteed_side: str | None
    resolved: bool
    scored: bool
    value: float | None
    covered: bool | None
    unscored_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bound_key", _bound_key(self.bound_key))
        if not isinstance(self.descriptor, GuaranteeDescriptor):
            raise TypeError("ledger score descriptor must be a GuaranteeDescriptor")
        if self.guaranteed_side is not None and self.guaranteed_side not in {"lower", "upper"}:
            raise ValueError("ledger score guaranteed side must be lower, upper, or omitted")
        if not isinstance(self.resolved, bool) or not isinstance(self.scored, bool):
            raise TypeError("ledger score state flags must be booleans")
        if not self.resolved:
            if self.scored or any(
                value is not None for value in (self.value, self.covered, self.unscored_reason)
            ):
                raise ValueError("pending ledger scores cannot contain scoring results")
            return
        if self.scored:
            object.__setattr__(
                self,
                "value",
                _finite_real(self.value, name="ledger score value"),
            )
            if self.covered is not None and not isinstance(self.covered, bool):
                raise TypeError("ledger score coverage must be a boolean or omitted")
            if self.unscored_reason is not None:
                raise ValueError("scored ledger outcomes cannot have an unscored reason")
            return
        if self.value is not None or self.covered is not None:
            raise ValueError("unscored ledger outcomes cannot contain predicate results")
        _require_text(self.unscored_reason, name="ledger score unscored reason")


@dataclass(frozen=True, slots=True, init=False)
class LedgerSelection:
    """Select one session, inclusive origin range, projection, and batch bound."""

    session: SessionIdentity
    columns: tuple[str, ...]
    batch_size: int
    origin_start: pd.Timestamp | None
    origin_end: pd.Timestamp | None

    def __init__(
        self,
        session: SessionIdentity,
        columns: Iterable[str | LedgerColumn],
        batch_size: int,
        *,
        origin_start: pd.Timestamp | None = None,
        origin_end: pd.Timestamp | None = None,
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("ledger selection session must be a SessionIdentity")
        normalized_columns = _column_sequence(columns, name="ledger selection columns")
        normalized_batch_size = _positive_integer(batch_size, name="ledger selection batch size")
        if origin_start is not None:
            _require_timestamp(origin_start, name="ledger selection origin start")
        if origin_end is not None:
            _require_timestamp(origin_end, name="ledger selection origin end")
        if origin_start is not None and origin_end is not None and origin_start > origin_end:
            raise ValueError("ledger selection origin start cannot follow its origin end")
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "columns", normalized_columns)
        object.__setattr__(self, "batch_size", normalized_batch_size)
        object.__setattr__(self, "origin_start", origin_start)
        object.__setattr__(self, "origin_end", origin_end)


@dataclass(frozen=True, slots=True, init=False)
class LedgerBatch:
    """Carry one row-aligned immutable column batch within a declared bound."""

    session: SessionIdentity
    keys: tuple[LedgerForecastKey, ...]
    columns: Mapping[str, tuple[object, ...]]
    batch_size: int
    row_count: int = field(init=False)

    def __init__(
        self,
        *,
        session: SessionIdentity,
        keys: Sequence[LedgerForecastKey],
        columns: Mapping[str | LedgerColumn, Iterable[object]],
        batch_size: int,
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("ledger batch session must be a SessionIdentity")
        if isinstance(keys, (str, bytes)):
            raise TypeError("ledger batch keys must be a sequence")
        try:
            frozen_keys = tuple(keys)
        except TypeError as error:
            raise TypeError("ledger batch keys must be a sequence") from error
        if any(not isinstance(key, LedgerForecastKey) for key in frozen_keys):
            raise TypeError("ledger batch keys must contain LedgerForecastKey values")
        if not isinstance(columns, Mapping):
            raise TypeError("ledger batch columns must be a mapping")
        if not columns:
            raise ValueError("ledger batch columns must not be empty")
        normalized_batch_size = _positive_integer(batch_size, name="ledger batch size")
        frozen_columns: dict[str, tuple[object, ...]] = {}
        for raw_name, values in columns.items():
            name = _column_name(raw_name)
            if name in frozen_columns:
                raise ValueError(f"ledger batch contains duplicate column {name!r}")
            frozen_columns[name] = _snapshot_column(name, values)
        lengths = {len(frozen_keys), *(len(values) for values in frozen_columns.values())}
        if len(lengths) != 1:
            raise ValueError("ledger batch keys and columns must have equal lengths")
        row_count = len(frozen_keys)
        if row_count > normalized_batch_size:
            raise ValueError("ledger batch row count cannot exceed its batch size")
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "keys", frozen_keys)
        object.__setattr__(self, "columns", MappingProxyType(frozen_columns))
        object.__setattr__(self, "batch_size", normalized_batch_size)
        object.__setattr__(self, "row_count", row_count)

    def __len__(self) -> int:
        """Return the number of row-aligned logical items in this batch."""
        return self.row_count


@dataclass(frozen=True, slots=True)
class LedgerSessionMetadata:
    """Identify one reader session and its complete canonical series set."""

    session: SessionIdentity
    series_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("ledger metadata session must be a SessionIdentity")
        if not isinstance(self.series_keys, tuple):
            raise TypeError("ledger metadata series keys must be a tuple")
        if self.series_keys != self.session.series_keys:
            raise ValueError("ledger metadata series keys must match the session identity")


@runtime_checkable
class LedgerReader(Protocol):
    """Stream a closed logical forecast ledger in bounded canonical batches."""

    @property
    def metadata(self) -> LedgerSessionMetadata:
        """Return immutable identity for the closed reader session."""
        ...

    def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
        """Return canonical immutable batches for ``selection``."""
        ...


def _column_sequence(
    columns: Iterable[str | LedgerColumn],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of column names")
    try:
        values = tuple(_column_name(column) for column in columns)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of column names") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicate columns")
    return values


def _column_name(value: object) -> str:
    name = value.value if isinstance(value, LedgerColumn) else value
    if not isinstance(name, str):
        raise TypeError("ledger column names must be strings or LedgerColumn values")
    if name not in _COLUMN_NAMES:
        available = ", ".join(sorted(_COLUMN_NAMES))
        raise ValueError(f"unsupported ledger column {name!r}; available columns: {available}")
    return name


def _snapshot_column(name: str, values: Iterable[object]) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"ledger batch column {name!r} must be an iterable")
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"ledger batch column {name!r} must be an iterable") from error
    return tuple(_snapshot_column_value(name, value) for value in raw_values)


def _snapshot_column_value(name: str, value: object) -> object:
    if name in {LedgerColumn.SERIES_KEY.value, LedgerColumn.MODEL_NAME.value}:
        _require_identifier(value, name=f"ledger batch {name}")
        return value
    if name in {LedgerColumn.ORIGIN.value, LedgerColumn.TARGET_TIMESTAMP.value}:
        _require_timestamp(value, name=f"ledger batch {name}")
        return value
    if name == LedgerColumn.HORIZON_STEP.value:
        return _positive_integer(value, name="ledger batch horizon step")
    if name == LedgerColumn.POINT_FORECAST.value:
        return _finite_real(value, name="ledger batch point forecast")
    if name == LedgerColumn.ISSUANCES.value:
        return _typed_tuple(value, LedgerBoundIssuance, name="ledger batch issuances")
    if name == LedgerColumn.RESOLUTION.value:
        if value is not None and not isinstance(value, LedgerResolution):
            raise TypeError("ledger batch resolutions must contain LedgerResolution or None")
        return value
    if name == LedgerColumn.SCORES.value:
        return _typed_tuple(value, LedgerBoundScore, name="ledger batch scores")
    raise AssertionError(f"unhandled ledger column {name!r}")


def _typed_tuple(value: object, item_type: type, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable")
    try:
        values = tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    if any(not isinstance(item, item_type) for item in values):
        raise TypeError(f"{name} must contain {item_type.__name__} values")
    return values


def _bound_key(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not 1 <= len(value) <= 2
        or any(not isinstance(column, str) or not column for column in value)
    ):
        raise ValueError("ledger bound key must contain one or two non-empty column names")
    if len(set(value)) != len(value):
        raise ValueError("ledger bound key cannot contain duplicate columns")
    return cast(tuple[str, ...], value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


def _optional_finite_real(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_real(value, name=name)


def _require_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error


def _require_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error


def _require_timestamp(value: object, *, name: str) -> None:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise TypeError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")


__all__ = [
    "LedgerBatch",
    "LedgerBoundIssuance",
    "LedgerBoundScore",
    "LedgerColumn",
    "LedgerForecastKey",
    "LedgerObservationAnnotation",
    "LedgerReader",
    "LedgerResolution",
    "LedgerSelection",
    "LedgerSessionMetadata",
]
