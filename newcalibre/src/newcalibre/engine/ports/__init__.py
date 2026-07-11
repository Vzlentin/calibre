"""Define the six I/O boundaries used by the engine core."""

from __future__ import annotations

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

    @property
    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the staged frame."""
        return self._frame.copy(deep=True)


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

    def before(self, origin: pd.Timestamp) -> Mapping[ActualKey, float]:
        """Return observations keyed by series and timestamp."""
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

    def receipt(self, origin: pd.Timestamp) -> OriginCommit | None:
        """Return the immutable commit receipt for an origin, when present."""
        ...

    def commit(self, write: OriginCommit) -> OriginCommit:
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


__all__ = [
    "ActualKey",
    "ActualsSource",
    "ArtifactStore",
    "ArtifactWrite",
    "CalibrationStateStore",
    "DispatchBackend",
    "ForecastWrite",
    "LedgerSnapshot",
    "LedgerSink",
    "OriginCommit",
    "PanelSource",
]
