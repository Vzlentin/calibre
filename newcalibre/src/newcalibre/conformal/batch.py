"""Define canonical immutable batches for conformal routing and state."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import cast

from newcalibre.conformal.types import (
    ForecastKey,
    ResolvedObservation,
    RuntimeContractError,
    _decode_label,
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
        try:
            scope, _payload = _decode_label(label)
        except RuntimeContractError as error:
            raise RuntimeContractError(str(error)) from error
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
        object.__setattr__(self, "_route_by_label", routes)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return canonical semantic partition labels."""
        return self._labels

    @property
    def observations(self) -> tuple[ResolvedObservation, ...]:
        """Flatten observations in canonical partition and supplied row order."""
        return tuple(value for row in self._observations for value in row)

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
        merged = dict(self.items())
        for label, state in _pairs(rows, name="conformal state transition rows"):
            merged[label] = state
        return ConformalStateBatch(merged)

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


__all__ = ["CalibrationSeedBatch", "ConformalStateBatch", "DeliveryBatch"]
