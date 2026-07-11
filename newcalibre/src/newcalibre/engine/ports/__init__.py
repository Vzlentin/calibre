"""Define the six I/O boundaries used by the engine core."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeVar, runtime_checkable

import pandas as pd

from newcalibre.domain import Calendar, Panel, SessionIdentity
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    ForecastRow,
    OrderRow,
    SettlementRecord,
)

type ActualKey = tuple[str, pd.Timestamp]

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


@dataclass(frozen=True, slots=True)
class ArtifactWrite:
    """Carry one idempotent model-artifact write inside an origin commit."""

    key: str
    value: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or self.key != self.key.strip():
            raise ValueError("artifact key must be a non-empty trimmed string")
        if not isinstance(self.value, bytes):
            raise TypeError("artifact write value must be bytes")


@dataclass(frozen=True, slots=True, init=False)
class ForecastWrite:
    """Carry one defensive forecast-frame snapshot into a ledger commit."""

    _frame: pd.DataFrame = field(repr=False)
    _digest: str = field(repr=False)
    issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]]

    def __init__(
        self,
        frame: pd.DataFrame,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]],
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("forecast write frame must be a pandas DataFrame")
        if not isinstance(issuances, Mapping):
            raise TypeError("forecast write issuances must be a mapping")
        frozen_issuances = {
            key: MappingProxyType(dict(by_bound)) for key, by_bound in issuances.items()
        }
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "issuances", MappingProxyType(frozen_issuances))
        object.__setattr__(self, "_digest", _forecast_write_digest(self))

    @property
    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the staged frame."""
        return self._frame.copy(deep=True)

    @property
    def digest(self) -> str:
        """Return the compact identity of the staged immutable facts."""
        return self._digest


@dataclass(frozen=True, slots=True)
class OriginCommit:
    """Journal one origin's complete durable payload for idempotent repair."""

    session: SessionIdentity
    origin: pd.Timestamp
    resolutions: Mapping[ForecastKey, float] = field(default_factory=dict)
    forecasts: tuple[ForecastWrite, ...] = ()
    orders: tuple[OrderRow, ...] = ()
    settlements: tuple[SettlementRecord, ...] = ()
    artifacts: tuple[ArtifactWrite, ...] = ()
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    _digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("origin commit session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("origin commit origin must be a pandas Timestamp")
        if not isinstance(self.resolutions, Mapping):
            raise TypeError("origin commit resolutions must be a mapping")
        object.__setattr__(self, "resolutions", MappingProxyType(dict(self.resolutions)))
        object.__setattr__(self, "forecasts", tuple(self.forecasts))
        object.__setattr__(self, "orders", tuple(self.orders))
        object.__setattr__(self, "settlements", tuple(self.settlements))
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ArtifactWrite) for artifact in artifacts):
            raise TypeError("origin commit artifacts must contain ArtifactWrite values")
        object.__setattr__(self, "artifacts", artifacts)
        if not isinstance(self.state_updates, Mapping):
            raise TypeError("origin commit state updates must be a mapping")
        updates: dict[str, bytes] = {}
        for partition, value in self.state_updates.items():
            if not isinstance(partition, str) or not partition or partition != partition.strip():
                raise ValueError("state-update partition must be a non-empty trimmed string")
            if not isinstance(value, bytes):
                raise TypeError("origin commit state updates must contain bytes")
            updates[partition] = value
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        object.__setattr__(self, "_digest", _origin_commit_digest(self))

    @property
    def digest(self) -> str:
        """Return the compact identity of the complete origin payload."""
        return self._digest


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """Retain compact idempotency identity plus pending repair payloads."""

    session: SessionIdentity
    origin: pd.Timestamp
    digest: str
    artifacts: tuple[ArtifactWrite, ...]
    state_updates: Mapping[str, bytes]

    @classmethod
    def from_commit(cls, commit: OriginCommit) -> CommitReceipt:
        """Compact a materialized commit after its ledger facts publish."""
        return cls(
            session=commit.session,
            origin=commit.origin,
            digest=commit.digest,
            artifacts=commit.artifacts,
            state_updates=commit.state_updates,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("commit receipt session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("commit receipt origin must be a pandas Timestamp")
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise ValueError("commit receipt digest must be a SHA-256 hex string")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ArtifactWrite) for artifact in artifacts):
            raise TypeError("commit receipt artifacts must contain ArtifactWrite values")
        object.__setattr__(self, "artifacts", artifacts)
        if not isinstance(self.state_updates, Mapping):
            raise TypeError("commit receipt state updates must be a mapping")
        object.__setattr__(self, "state_updates", MappingProxyType(dict(self.state_updates)))


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Expose immutable ledger facts without leaking its mutable owner."""

    session: SessionIdentity
    calendar: Calendar
    forecasts: tuple[ForecastRow, ...]
    orders: tuple[OrderRow, ...]
    settlements: tuple[SettlementRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("ledger snapshot session must be a SessionIdentity")
        if not isinstance(self.calendar, Calendar):
            raise TypeError("ledger snapshot calendar must be a Calendar")
        object.__setattr__(self, "forecasts", tuple(self.forecasts))
        object.__setattr__(self, "orders", tuple(self.orders))
        object.__setattr__(self, "settlements", tuple(self.settlements))


@runtime_checkable
class PanelSource(Protocol):
    """Load the immutable panel that defines a run."""

    def load(self) -> Panel:
        """Return the run's panel."""
        ...


@runtime_checkable
class ActualsSource(Protocol):
    """Reveal actuals admissible strictly before an origin."""

    def for_keys(
        self,
        keys: Sequence[ActualKey],
        *,
        before: pd.Timestamp,
    ) -> Mapping[ActualKey, float]:
        """Return the requested observations admissible before ``before``."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Persist opaque model artifacts under engine-computed keys."""

    def load(self, key: str) -> bytes | None:
        """Return an artifact snapshot, or ``None`` when absent."""
        ...

    def save(self, key: str, value: bytes) -> None:
        """Atomically persist one idempotent opaque artifact."""
        ...


@runtime_checkable
class CalibrationStateStore(Protocol):
    """Persist calibration state by session and partition."""

    def load(self, session: SessionIdentity, partition: str) -> bytes | None:
        """Return a state snapshot, or ``None`` when absent."""
        ...

    def save(self, session: SessionIdentity, partition: str, value: bytes) -> None:
        """Atomically persist one idempotent partition-state value."""
        ...


@runtime_checkable
class LedgerSink(Protocol):
    """Expose due rows and atomically accept one origin's ledger writes."""

    @property
    def session(self) -> SessionIdentity:
        """Return the session owned by the ledger."""
        ...

    @property
    def calendar(self) -> Calendar:
        """Return the calendar owned by the ledger."""
        ...

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        """Return pending rows due strictly before ``origin``."""
        ...

    def snapshot(self) -> LedgerSnapshot:
        """Return immutable forecast, order, and settlement facts."""
        ...

    def receipt(self, origin: pd.Timestamp) -> CommitReceipt | None:
        """Return the immutable commit receipt for an origin, when present."""
        ...

    def commit(self, write: OriginCommit) -> CommitReceipt:
        """Atomically journal and apply a write, returning its repair receipt."""
        ...


@runtime_checkable
class DispatchBackend(Protocol):
    """Place work without changing its values or deterministic order."""

    def map(
        self,
        function: Callable[[_Input], _Output],
        items: Sequence[_Input],
    ) -> tuple[_Output, ...]:
        """Apply ``function`` in input order and return results in that order."""
        ...


def _forecast_write_digest(write: ForecastWrite) -> str:
    digest = hashlib.sha256()
    schema = tuple((str(column), str(write._frame[column].dtype)) for column in write._frame)
    digest.update(repr(schema).encode())
    digest.update(
        pd.util.hash_pandas_object(write._frame, index=False, categorize=True)
        .to_numpy(dtype="uint64")
        .tobytes()
    )
    issuance_facts = tuple(
        sorted(
            (
                repr(forecast_key),
                repr(bound_key),
                repr(issuance),
            )
            for forecast_key, by_bound in write.issuances.items()
            for bound_key, issuance in by_bound.items()
        )
    )
    digest.update(repr(issuance_facts).encode())
    return digest.hexdigest()


def _origin_commit_digest(commit: OriginCommit) -> str:
    digest = hashlib.sha256()
    digest.update(commit.session.value.encode())
    digest.update(f"{commit.origin.isoformat()}:{commit.origin.unit}".encode())
    resolution_facts = tuple(
        sorted((repr(key), float(value).hex()) for key, value in commit.resolutions.items())
    )
    digest.update(repr(resolution_facts).encode())
    digest.update(repr(tuple(write.digest for write in commit.forecasts)).encode())
    digest.update(repr(commit.orders).encode())
    digest.update(repr(commit.settlements).encode())
    for artifact in commit.artifacts:
        digest.update(artifact.key.encode())
        digest.update(artifact.value)
    for partition, state in sorted(commit.state_updates.items()):
        digest.update(partition.encode())
        digest.update(state)
    return digest.hexdigest()


__all__ = [
    "ActualKey",
    "ActualsSource",
    "ArtifactStore",
    "ArtifactWrite",
    "CalibrationStateStore",
    "CommitReceipt",
    "DispatchBackend",
    "ForecastWrite",
    "LedgerSnapshot",
    "LedgerSink",
    "OriginCommit",
    "PanelSource",
]
