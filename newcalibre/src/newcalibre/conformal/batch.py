"""Define canonical immutable batches for conformal routing and state."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import cast

import pandas as pd

from newcalibre.conformal.types import (
    ForecastKey,
    IssuedBoundFacts,
    ObserveAnnotation,
    ResolvedObservation,
    RuntimeContractError,
    _decode_label,
    _snapshot_issuances,
    _snapshot_iterable,
)

type _PairRows[T] = Mapping[str, T] | Iterable[tuple[str, T]]


def _pairs[T](values: _PairRows[T] | None, *, name: str) -> tuple[tuple[str, T], ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        snapshot = tuple(values.items())
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise RuntimeContractError(f"{name} must be a mapping or iterable of pairs")
        raw = tuple(values)
        if any(not isinstance(value, tuple) or len(value) != 2 for value in raw):
            raise RuntimeContractError(f"{name} must contain label/value pairs")
        snapshot = raw
    labels = tuple(label for label, _value in snapshot)
    if any(not isinstance(label, str) for label in labels):
        raise RuntimeContractError(f"{name} labels must be strings")
    if len(set(labels)) != len(labels):
        raise RuntimeContractError(f"{name} contain duplicate labels")
    return cast(tuple[tuple[str, T], ...], snapshot)


def _canonical_labels(labels: Iterable[str], *, partitions_only: bool) -> tuple[str, ...]:
    snapshot = tuple(labels)
    for label in snapshot:
        scope, _payload = _decode_label(label)
        if partitions_only and scope != "partition":
            raise RuntimeContractError("batch rows must use partition labels")
    return tuple(sorted(snapshot, key=str.encode))


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeContractError("calibration scores must be finite nonnegative real numbers")
    normalized = float(value)
    if normalized < 0.0 or not math.isfinite(normalized):
        raise RuntimeContractError("calibration scores must be finite nonnegative real numbers")
    return 0.0 if normalized == 0.0 else normalized


def _observation_order(observation: ResolvedObservation) -> tuple:
    key = observation.forecast_key
    return key.origin, key.horizon_step, key.series_key.encode(), key.model_name.encode()


@dataclass(frozen=True, slots=True, init=False)
class CalibrationSeedBatch:
    """Carry canonical partition-addressed calibration score sequences."""

    _labels: tuple[str, ...]
    _scores: tuple[tuple[float, ...], ...]
    _route_by_label: Mapping[str, int] = field(repr=False)

    def __init__(
        self,
        values: _PairRows[Sequence[float]] | None = None,
    ) -> None:
        supplied = _pairs(values, name="calibration seed rows")
        labels = _canonical_labels((label for label, _value in supplied), partitions_only=True)
        by_label = dict(supplied)
        scores: list[tuple[float, ...]] = []
        for label in labels:
            values_for_label = by_label[label]
            if isinstance(values_for_label, (str, bytes)) or not isinstance(
                values_for_label, Sequence
            ):
                raise RuntimeContractError("partition calibration scores must be a sequence")
            scores.append(tuple(_score(value) for value in values_for_label))
        routes = MappingProxyType({label: route for route, label in enumerate(labels)})
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_scores", tuple(scores))
        object.__setattr__(self, "_route_by_label", routes)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return canonical semantic partition labels."""
        return self._labels

    def scores_for(self, label: str) -> tuple[float, ...]:
        """Return the immutable score row for one semantic label."""
        try:
            return self._scores[self._route_by_label[label]]
        except (KeyError, TypeError) as error:
            raise RuntimeContractError(f"calibration seed batch has no label {label!r}") from error

    def items(self) -> Iterator[tuple[str, tuple[float, ...]]]:
        """Iterate semantic rows in canonical order."""
        return iter(zip(self._labels, self._scores, strict=True))

    def __len__(self) -> int:
        return len(self._labels)


@dataclass(frozen=True, slots=True, init=False)
class DeliveryBatch:
    """Carry canonical partitions with ordered resolved observations."""

    _labels: tuple[str, ...]
    _observations: tuple[tuple[ResolvedObservation, ...], ...]
    _flat_observations: tuple[ResolvedObservation, ...]
    _route_by_label: Mapping[str, int] = field(repr=False)

    def __init__(
        self,
        values: _PairRows[Iterable[ResolvedObservation]] | None = None,
    ) -> None:
        supplied = _pairs(values, name="delivery rows")
        labels = _canonical_labels((label for label, _value in supplied), partitions_only=True)
        by_label = dict(supplied)
        rows: list[tuple[ResolvedObservation, ...]] = []
        seen: set[ForecastKey] = set()
        for label in labels:
            raw = by_label[label]
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
                raise RuntimeContractError("delivery observations must be an iterable")
            observations = tuple(raw)
            if not observations:
                raise RuntimeContractError("delivery partition observations must not be empty")
            if any(not isinstance(value, ResolvedObservation) for value in observations):
                raise RuntimeContractError(
                    "every delivery observation must be a ResolvedObservation"
                )
            if any(value.issued.partition_label != label for value in observations):
                raise RuntimeContractError(
                    "every issued partition must match its delivery partition"
                )
            keys = tuple(value.forecast_key for value in observations)
            duplicates = seen.intersection(keys)
            if len(set(keys)) != len(keys) or duplicates:
                raise RuntimeContractError("delivery batch contains a duplicate forecast key")
            seen.update(keys)
            rows.append(observations)
        routes = MappingProxyType({label: route for route, label in enumerate(labels)})
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_observations", tuple(rows))
        object.__setattr__(
            self,
            "_flat_observations",
            tuple(sorted((value for row in rows for value in row), key=_observation_order)),
        )
        object.__setattr__(self, "_route_by_label", routes)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return canonical semantic partition labels."""
        return self._labels

    @property
    def observations(self) -> tuple[ResolvedObservation, ...]:
        """Flatten observations in the normative total delivery order."""
        return self._flat_observations

    def observations_for(self, label: str) -> tuple[ResolvedObservation, ...]:
        """Return ordered observations for one semantic partition."""
        try:
            return self._observations[self._route_by_label[label]]
        except (KeyError, TypeError) as error:
            raise RuntimeContractError(f"delivery batch has no label {label!r}") from error

    def items(self) -> Iterator[tuple[str, tuple[ResolvedObservation, ...]]]:
        """Iterate semantic partition rows in canonical order."""
        return iter(zip(self._labels, self._observations, strict=True))

    def __len__(self) -> int:
        return len(self._labels)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ConformalStateBatch(Mapping[str, bytes]):
    """Own a canonical complete snapshot of opaque semantic state rows."""

    _labels: tuple[str, ...]
    _states: tuple[bytes, ...]
    _route_by_label: Mapping[str, int] = field(repr=False)

    def __init__(self, values: _PairRows[bytes] | None = None) -> None:
        supplied = _pairs(values, name="conformal state rows")
        labels = _canonical_labels((label for label, _value in supplied), partitions_only=False)
        by_label = dict(supplied)
        states: list[bytes] = []
        for label in labels:
            state = by_label[label]
            if not isinstance(state, bytes):
                raise RuntimeContractError("conformal state rows must contain immutable bytes")
            states.append(state)
        routes = MappingProxyType({label: route for route, label in enumerate(labels)})
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_states", tuple(states))
        object.__setattr__(self, "_route_by_label", routes)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return canonical semantic state labels."""
        return self._labels

    def __getitem__(self, label: str) -> bytes:
        try:
            return self._states[self._route_by_label[label]]
        except (KeyError, TypeError) as error:
            raise KeyError(label) from error

    def __contains__(self, label: object) -> bool:
        return label in self._route_by_label

    def __iter__(self) -> Iterator[str]:
        return iter(self._labels)

    def with_rows(self, rows: _PairRows[bytes]) -> ConformalStateBatch:
        """Return a complete snapshot with exact semantic rows replaced or added."""
        supplied = _pairs(rows, name="conformal state transition rows")
        if not supplied:
            return self
        labels = _canonical_labels(
            (label for label, _state in supplied),
            partitions_only=False,
        )
        by_label = dict(supplied)
        for state in by_label.values():
            if not isinstance(state, bytes):
                raise RuntimeContractError(
                    "conformal state transition rows must contain immutable bytes"
                )
        if any(label not in self for label in labels):
            merged = dict(self.items())
            merged.update(by_label)
            return ConformalStateBatch(merged)

        states = list(self._states)
        changed = False
        for label in labels:
            route = self._route_by_label[label]
            state = by_label[label]
            if states[route] != state:
                states[route] = state
                changed = True
        if not changed:
            return self
        result = object.__new__(type(self))
        object.__setattr__(result, "_labels", self._labels)
        object.__setattr__(result, "_states", tuple(states))
        object.__setattr__(result, "_route_by_label", self._route_by_label)
        return result

    def project(self, labels: Iterable[str]) -> Mapping[str, bytes]:
        """Project an exact validated semantic-label subset as immutable bytes."""
        dirty = _snapshot_dirty_labels(labels, state=self)
        return MappingProxyType({label: self[label] for label in dirty})

    def __len__(self) -> int:
        return len(self._labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(tuple(self.items()))


def state_delta(
    prior: ConformalStateBatch,
    post: ConformalStateBatch,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return removed and changed semantic labels for one state transition."""
    if not isinstance(prior, ConformalStateBatch) or not isinstance(
        post,
        ConformalStateBatch,
    ):
        raise RuntimeContractError("state transition requires conformal state batches")
    removed = tuple(label for label in prior.labels if label not in post)
    changed = tuple(
        label for label in post.labels if label not in prior or prior[label] != post[label]
    )
    return removed, changed


def validate_state_transition(
    prior: ConformalStateBatch,
    post: ConformalStateBatch,
    dirty_labels: Iterable[str],
) -> None:
    """Require a complete non-removing post-state with an exact dirty set."""
    removed, changed = state_delta(prior, post)
    if removed:
        raise RuntimeContractError(f"post-state removed rows: {list(removed)!r}")
    dirty = _snapshot_dirty_labels(dirty_labels, state=post)
    if changed != dirty:
        raise RuntimeContractError("dirty labels must exactly identify changed post-state rows")


def _snapshot_dirty_labels(
    labels: Iterable[str],
    *,
    state: ConformalStateBatch,
) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Iterable):
        raise RuntimeContractError("dirty labels must be an iterable")
    snapshot = tuple(labels)
    if any(not isinstance(label, str) for label in snapshot):
        raise RuntimeContractError("dirty labels must be strings")
    if len(set(snapshot)) != len(snapshot):
        raise RuntimeContractError("dirty labels contain duplicates")
    canonical = _canonical_labels(snapshot, partitions_only=False)
    foreign = tuple(label for label in canonical if label not in state)
    if foreign:
        raise RuntimeContractError(f"dirty labels are absent from post-state: {foreign!r}")
    return canonical


@dataclass(frozen=True, slots=True, init=False)
class ObserveEffect:
    """Return complete post-state, dirty labels, and observe annotations."""

    state: ConformalStateBatch
    dirty_labels: tuple[str, ...]
    annotations: tuple[ObserveAnnotation, ...]

    def __init__(
        self,
        state: ConformalStateBatch,
        dirty_labels: Iterable[str] = (),
        annotations: Iterable[ObserveAnnotation] = (),
    ) -> None:
        if not isinstance(state, ConformalStateBatch):
            raise RuntimeContractError("observe post-state must be a ConformalStateBatch")
        dirty = _snapshot_dirty_labels(dirty_labels, state=state)
        annotated = _snapshot_iterable(annotations, name="observe annotations")
        if any(not isinstance(value, ObserveAnnotation) for value in annotated):
            raise RuntimeContractError("every observe annotation must be an ObserveAnnotation")
        keys = tuple(value.forecast_key for value in annotated)
        if len(set(keys)) != len(keys):
            raise RuntimeContractError("observe annotations contain a duplicate forecast key")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "dirty_labels", dirty)
        object.__setattr__(self, "annotations", annotated)

    @property
    def dirty_state(self) -> Mapping[str, bytes]:
        """Project dirty semantic rows from the complete post-state."""
        return self.state.project(self.dirty_labels)


@dataclass(frozen=True, slots=True, init=False)
class CalibrationResult:
    """Return calibrated forecasts, complete post-state, and dirty labels."""

    _forecasts: pd.DataFrame = field(repr=False)
    state: ConformalStateBatch
    dirty_labels: tuple[str, ...]
    issuances: Mapping[ForecastKey, IssuedBoundFacts]

    def __init__(
        self,
        forecasts: pd.DataFrame,
        state: ConformalStateBatch,
        dirty_labels: Iterable[str] = (),
        issuances: Mapping[ForecastKey, IssuedBoundFacts] | None = None,
    ) -> None:
        if not isinstance(forecasts, pd.DataFrame):
            raise RuntimeContractError("calibrated forecasts must be a pandas DataFrame")
        if forecasts.columns.has_duplicates:
            raise RuntimeContractError("calibrated forecasts cannot have duplicate columns")
        if not isinstance(state, ConformalStateBatch):
            raise RuntimeContractError("apply post-state must be a ConformalStateBatch")
        snapshot = forecasts.copy(deep=True)
        snapshot.attrs = {}
        dirty = _snapshot_dirty_labels(dirty_labels, state=state)
        frozen_issuances = _snapshot_issuances(snapshot, issuances)
        object.__setattr__(self, "_forecasts", snapshot)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "dirty_labels", dirty)
        object.__setattr__(self, "issuances", MappingProxyType(frozen_issuances))

    @property
    def forecasts(self) -> pd.DataFrame:
        """Return an isolated copy of the calibrated forecasts."""
        return self._forecasts.copy(deep=True)

    @property
    def dirty_state(self) -> Mapping[str, bytes]:
        """Project dirty semantic rows from the complete post-state."""
        return self.state.project(self.dirty_labels)


__all__ = [
    "CalibrationResult",
    "CalibrationSeedBatch",
    "ConformalStateBatch",
    "DeliveryBatch",
    "ObserveEffect",
    "state_delta",
]
