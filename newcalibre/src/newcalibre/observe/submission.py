"""Define validated atomic actual submissions for the observe loop."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd

from newcalibre.domain import CensoringAssertion

type ActualKey = tuple[str, pd.Timestamp]
type RecordedValue = int | float


class ObserveError(ValueError):
    """Report an invalid observe value, state snapshot, or loop operation."""


@dataclass(frozen=True, slots=True)
class ActualRecord:
    """Carry one posted bottom-series actual and its recorded censoring facts."""

    series_key: str
    timestamp: pd.Timestamp
    recorded_value: RecordedValue
    censoring_assertion: CensoringAssertion | None = None
    availability_bound: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.series_key, name="series key")
        _require_timestamp(self.timestamp, name="actual timestamp")
        object.__setattr__(
            self,
            "recorded_value",
            _finite_number(self.recorded_value, name="recorded value"),
        )
        if self.censoring_assertion is not None and not isinstance(
            self.censoring_assertion,
            CensoringAssertion,
        ):
            raise ObserveError("censoring assertion must be a CensoringAssertion or undeclared")
        if self.availability_bound is not None:
            bound = _finite_number(self.availability_bound, name="availability bound")
            object.__setattr__(self, "availability_bound", float(bound))

    @property
    def key(self) -> ActualKey:
        """Return the observed-history key."""
        return self.series_key, self.timestamp


@dataclass(frozen=True, slots=True, init=False)
class ActualsSubmission:
    """Snapshot one duplicate-free atomic batch of actual records."""

    records: tuple[ActualRecord, ...] = field(default=())

    def __init__(self, records: Iterable[ActualRecord]) -> None:
        values = _snapshot_iterable(records, name="actual records")
        if any(not isinstance(record, ActualRecord) for record in values):
            raise ObserveError("every submitted actual must be an ActualRecord")
        keys = tuple(record.key for record in values)
        if len(set(keys)) != len(keys):
            raise ObserveError("actuals submission contains a duplicate series/timestamp key")
        object.__setattr__(self, "records", values)


def _snapshot_iterable[T](values: Iterable[T], *, name: str) -> tuple[T, ...]:
    if isinstance(values, (str, bytes)):
        raise ObserveError(f"{name} must be an iterable of values")
    try:
        return tuple(values)
    except TypeError as error:
        raise ObserveError(f"{name} must be an iterable of values") from error


def _finite_number(value: object, *, name: str) -> RecordedValue:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ObserveError(f"{name} must be a finite real number")
    if isinstance(value, Integral):
        return int(value)
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ObserveError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise ObserveError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


def _require_timestamp(value: object, *, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise ObserveError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise ObserveError(f"{name} must be timezone-naive")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObserveError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ObserveError(f"{name} must be valid UTF-8") from error
    return value


__all__ = [
    "ActualKey",
    "ActualRecord",
    "ActualsSubmission",
    "ObserveError",
    "RecordedValue",
]
