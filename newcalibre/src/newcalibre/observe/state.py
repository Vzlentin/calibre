"""Define immutable storage-shaped observe-loop state and staged deltas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from newcalibre.conformal import (
    Delivery,
    ForecastKey,
    IssuedBoundFacts,
    ObserveAnnotation,
)
from newcalibre.domain import CensoringAssertion
from newcalibre.observe.submission import (
    ActualKey,
    ActualRecord,
    ObserveError,
    RecordedValue,
    _finite_number,
    _require_text,
    _require_timestamp,
    _snapshot_iterable,
)


@dataclass(frozen=True, slots=True)
class ObservedActual:
    """Store one accepted bottom-series actual under its durable history key."""

    series_key: str
    timestamp: pd.Timestamp
    recorded_value: RecordedValue
    censoring_assertion: CensoringAssertion | None = None
    availability_bound: float | None = None

    def __post_init__(self) -> None:
        record = ActualRecord(
            self.series_key,
            self.timestamp,
            self.recorded_value,
            self.censoring_assertion,
            self.availability_bound,
        )
        object.__setattr__(self, "recorded_value", record.recorded_value)
        object.__setattr__(self, "availability_bound", record.availability_bound)

    @classmethod
    def from_record(cls, record: ActualRecord) -> ObservedActual:
        """Snapshot one validated submission record for durable history."""
        if not isinstance(record, ActualRecord):
            raise ObserveError("observed history requires an ActualRecord")
        return cls(
            series_key=record.series_key,
            timestamp=record.timestamp,
            recorded_value=record.recorded_value,
            censoring_assertion=record.censoring_assertion,
            availability_bound=record.availability_bound,
        )

    @property
    def key(self) -> ActualKey:
        """Return the durable observed-history key."""
        return self.series_key, self.timestamp

    @property
    def recorded_fact(self) -> tuple[RecordedValue, CensoringAssertion | None, float | None]:
        """Return the complete idempotency and conflict-comparison fact."""
        return self.recorded_value, self.censoring_assertion, self.availability_bound


@dataclass(frozen=True, slots=True)
class ObservationResolution:
    """Stage one censoring-aware actual for a keyed pending forecast row."""

    forecast_key: ForecastKey
    target_timestamp: pd.Timestamp
    actual: RecordedValue
    censoring_assertion: CensoringAssertion | None
    availability_bound: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_key, ForecastKey):
            raise ObserveError("resolution forecast key must be a ForecastKey")
        _require_timestamp(self.target_timestamp, name="resolution target timestamp")
        object.__setattr__(self, "actual", _finite_number(self.actual, name="resolved actual"))
        if self.censoring_assertion is not None and not isinstance(
            self.censoring_assertion,
            CensoringAssertion,
        ):
            raise ObserveError(
                "resolution censoring assertion must be a CensoringAssertion or undeclared"
            )
        if self.availability_bound is not None:
            bound = _finite_number(self.availability_bound, name="resolution availability bound")
            object.__setattr__(self, "availability_bound", float(bound))


@dataclass(frozen=True, slots=True)
class PendingObservation:
    """Store one pending forecast row and its optional not-yet-delivered resolution."""

    forecast_key: ForecastKey
    target_timestamp: pd.Timestamp
    point_forecast: float
    issued: IssuedBoundFacts | None = None
    resolution: ObservationResolution | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_key, ForecastKey):
            raise ObserveError("pending observation forecast key must be a ForecastKey")
        _require_timestamp(self.target_timestamp, name="pending target timestamp")
        point = _finite_number(self.point_forecast, name="pending point forecast")
        object.__setattr__(self, "point_forecast", float(point))
        if self.issued is not None:
            try:
                issued = IssuedBoundFacts.snapshot(self.issued)
            except ValueError as error:
                raise ObserveError(str(error)) from error
            object.__setattr__(self, "issued", issued)
        if self.resolution is not None:
            if not isinstance(self.resolution, ObservationResolution):
                raise ObserveError("pending resolution must be an ObservationResolution")
            if self.resolution.forecast_key != self.forecast_key:
                raise ObserveError("pending resolution forecast key must match its row")
            if self.resolution.target_timestamp != self.target_timestamp:
                raise ObserveError("pending resolution target timestamp must match its row")


@dataclass(frozen=True, slots=True, init=False)
class Acceptance:
    """Return newly staged history appends and keys accepted as idempotent no-ops."""

    history_appends: tuple[ObservedActual, ...]
    idempotent_keys: tuple[ActualKey, ...]

    def __init__(
        self,
        history_appends: Iterable[ObservedActual] = (),
        idempotent_keys: Iterable[ActualKey] = (),
    ) -> None:
        appends = _snapshot_iterable(history_appends, name="accepted history appends")
        if any(not isinstance(value, ObservedActual) for value in appends):
            raise ObserveError("history appends must contain ObservedActual values")
        keys = _snapshot_iterable(idempotent_keys, name="idempotent actual keys")
        for key in keys:
            _validate_actual_key(key)
        if len({value.key for value in appends}) != len(appends):
            raise ObserveError("accepted history appends contain duplicate keys")
        if len(set(keys)) != len(keys):
            raise ObserveError("idempotent actual keys contain duplicates")
        object.__setattr__(self, "history_appends", appends)
        object.__setattr__(self, "idempotent_keys", keys)


@dataclass(frozen=True, slots=True, init=False)
class ObserveCycle:
    """Return one immutable staged observe transaction without persisting it."""

    history_appends: tuple[ObservedActual, ...]
    resolutions: tuple[ObservationResolution, ...]
    pending_removals: tuple[ForecastKey, ...]
    pending_retentions: tuple[PendingObservation, ...]
    deliveries: tuple[Delivery, ...]
    annotations: tuple[ObserveAnnotation, ...]
    state_updates: Mapping[str, bytes]

    def __init__(
        self,
        *,
        history_appends: Iterable[ObservedActual] = (),
        resolutions: Iterable[ObservationResolution] = (),
        pending_removals: Iterable[ForecastKey] = (),
        pending_retentions: Iterable[PendingObservation] = (),
        deliveries: Iterable[Delivery] = (),
        annotations: Iterable[ObserveAnnotation] = (),
        state_updates: Mapping[str, bytes] | None = None,
    ) -> None:
        history = _typed_tuple(
            history_appends,
            value_type=ObservedActual,
            name="cycle history appends",
        )
        resolved = _typed_tuple(
            resolutions,
            value_type=ObservationResolution,
            name="cycle resolutions",
        )
        removals = _typed_tuple(
            pending_removals,
            value_type=ForecastKey,
            name="cycle pending removals",
        )
        retained = _typed_tuple(
            pending_retentions,
            value_type=PendingObservation,
            name="cycle pending retentions",
        )
        routed = _typed_tuple(deliveries, value_type=Delivery, name="cycle deliveries")
        annotated = _typed_tuple(
            annotations,
            value_type=ObserveAnnotation,
            name="cycle annotations",
        )
        updates = _state_updates(state_updates)

        _require_unique((value.key for value in history), name="cycle history keys")
        _require_unique(
            (value.forecast_key for value in resolved),
            name="cycle resolution keys",
        )
        _require_unique(removals, name="cycle pending removal keys")
        retained_keys = tuple(value.forecast_key for value in retained)
        _require_unique(retained_keys, name="cycle pending retention keys")
        if set(removals).intersection(retained_keys):
            raise ObserveError("a pending row cannot be both removed and retained")
        _require_unique(
            (value.forecast_key for value in annotated),
            name="cycle annotation keys",
        )

        object.__setattr__(self, "history_appends", history)
        object.__setattr__(self, "resolutions", resolved)
        object.__setattr__(self, "pending_removals", removals)
        object.__setattr__(self, "pending_retentions", retained)
        object.__setattr__(self, "deliveries", routed)
        object.__setattr__(self, "annotations", annotated)
        object.__setattr__(self, "state_updates", MappingProxyType(updates))


def _validate_actual_key(key: object) -> None:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ObserveError("actual key must be a series/timestamp tuple")
    _require_text(key[0], name="actual key series")
    _require_timestamp(key[1], name="actual key timestamp")


def _typed_tuple[T](
    values: Iterable[T],
    *,
    value_type: type[T],
    name: str,
) -> tuple[T, ...]:
    snapshot = _snapshot_iterable(values, name=name)
    if any(not isinstance(value, value_type) for value in snapshot):
        raise ObserveError(f"{name} must contain {value_type.__name__} values")
    return snapshot


def _require_unique(values: Iterable[object], *, name: str) -> None:
    snapshot = tuple(values)
    if len(set(snapshot)) != len(snapshot):
        raise ObserveError(f"{name} contain duplicates")


def _state_updates(values: Mapping[str, bytes] | None) -> dict[str, bytes]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ObserveError("cycle state updates must be a mapping")
    snapshot = dict(values)
    for label, value in snapshot.items():
        _require_text(label, name="state update label")
        if not isinstance(value, bytes):
            raise ObserveError("cycle state updates must contain immutable bytes")
    return snapshot


__all__ = [
    "Acceptance",
    "ObservationResolution",
    "ObserveCycle",
    "ObservedActual",
    "PendingObservation",
]
