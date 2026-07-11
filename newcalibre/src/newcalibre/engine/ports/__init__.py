"""Define the six I/O boundaries used by the engine core."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Real
from types import MappingProxyType
from typing import Protocol, TypeVar, runtime_checkable

import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    InventoryPosition,
    Panel,
    SessionIdentity,
)
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    OrderRow,
    SettlementRecord,
)

type ActualKey = tuple[str, pd.Timestamp]

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


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
    state_updates: Mapping[str, bytes]
    settlement_periods: tuple[pd.Timestamp, ...] = ()

    @classmethod
    def from_commit(cls, commit: OriginCommit) -> CommitReceipt:
        """Compact a materialized commit after its ledger facts publish."""
        return cls(
            session=commit.session,
            origin=commit.origin,
            digest=commit.digest,
            state_updates=commit.state_updates,
            settlement_periods=tuple(sorted({record.period for record in commit.settlements})),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("commit receipt session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("commit receipt origin must be a pandas Timestamp")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or self.digest != self.digest.lower()
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("commit receipt digest must be a SHA-256 hex string")
        if not isinstance(self.state_updates, Mapping):
            raise TypeError("commit receipt state updates must be a mapping")
        updates: dict[str, bytes] = {}
        for partition, value in self.state_updates.items():
            if not isinstance(partition, str) or not partition or partition != partition.strip():
                raise ValueError("commit receipt partition must be a non-empty trimmed string")
            if not isinstance(value, bytes):
                raise TypeError("commit receipt state updates must contain bytes")
            updates[partition] = value
        settlement_periods = tuple(self.settlement_periods)
        for period in settlement_periods:
            if not isinstance(period, pd.Timestamp) or pd.isna(period):
                raise TypeError("commit receipt settlement periods must be pandas Timestamps")
            if period.tz is not None:
                raise ValueError("commit receipt settlement periods must be timezone-naive")
        if any(current <= previous for previous, current in pairwise(settlement_periods)):
            raise ValueError(
                "commit receipt settlement periods must be strictly increasing and unique"
            )
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        object.__setattr__(self, "settlement_periods", settlement_periods)


@dataclass(frozen=True, slots=True)
class SettlementSnapshot:
    """Project only the indexed durable facts needed by one settlement window."""

    session: SessionIdentity
    calendar: Calendar
    periods: tuple[pd.Timestamp, ...]
    frontier: pd.Timestamp | None
    latest_positions: Mapping[str, InventoryPosition]
    open_order_quantities: Mapping[str, float]
    due_arrivals: Mapping[ActualKey, float]
    actuals_semantics: ActualsSemantics | None

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("settlement snapshot session must be a SessionIdentity")
        if not isinstance(self.calendar, Calendar):
            raise TypeError("settlement snapshot calendar must be a Calendar")
        periods = tuple(self.periods)
        if not periods:
            raise ValueError("settlement snapshot periods must not be empty")
        for index, period in enumerate(periods):
            self.calendar.require_member(period, name="settlement snapshot period")
            if index and period != self.calendar.advance(periods[index - 1], 1):
                raise ValueError("settlement snapshot periods must be calendar-contiguous")
        if self.frontier is not None:
            self.calendar.require_member(self.frontier, name="settlement snapshot frontier")
        if not isinstance(self.latest_positions, Mapping):
            raise TypeError("settlement snapshot latest positions must be a mapping")
        latest_positions: dict[str, InventoryPosition] = {}
        for series_key, position in self.latest_positions.items():
            if not isinstance(series_key, str) or not series_key:
                raise ValueError("settlement snapshot series keys must be non-empty strings")
            if not isinstance(position, InventoryPosition):
                raise TypeError(
                    "settlement snapshot latest positions must contain InventoryPosition values"
                )
            latest_positions[series_key] = position
        open_quantities = _frozen_nonnegative_quantities(
            self.open_order_quantities,
            name="settlement snapshot open-order quantities",
        )
        due_arrivals = _frozen_nonnegative_quantities(
            self.due_arrivals,
            name="settlement snapshot due arrivals",
        )
        period_set = set(periods)
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not key[0]
            or not isinstance(key[1], pd.Timestamp)
            or key[1] not in period_set
            for key in due_arrivals
        ):
            raise ValueError("settlement snapshot due arrivals must be keyed inside its window")
        if self.actuals_semantics is not None and not isinstance(
            self.actuals_semantics,
            ActualsSemantics,
        ):
            raise TypeError(
                "settlement snapshot actuals semantics must be ActualsSemantics or None"
            )
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "latest_positions", MappingProxyType(latest_positions))
        object.__setattr__(self, "open_order_quantities", MappingProxyType(open_quantities))
        object.__setattr__(self, "due_arrivals", MappingProxyType(due_arrivals))


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

    def save(
        self,
        session: SessionIdentity,
        partition: str,
        value: bytes,
        *,
        origin: pd.Timestamp,
    ) -> None:
        """Persist a partition value without allowing an older origin to replace it."""
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

    def settlement_snapshot(
        self,
        periods: Sequence[pd.Timestamp],
    ) -> SettlementSnapshot:
        """Return the compact indexed facts needed by ``periods``."""
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
    _update_digest(digest, b"schema", repr(schema).encode())
    _update_digest(
        digest,
        b"rows",
        pd.util.hash_pandas_object(write._frame, index=False, categorize=True)
        .to_numpy(dtype="uint64")
        .tobytes(),
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
    _update_digest(digest, b"issuances", repr(issuance_facts).encode())
    return digest.hexdigest()


def _origin_commit_digest(commit: OriginCommit) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, b"schema", b"newcalibre.origin-commit/v1")
    _update_digest(digest, b"session", commit.session.value.encode())
    _update_digest(
        digest,
        b"origin",
        f"{commit.origin.isoformat()}:{commit.origin.unit}".encode(),
    )
    resolution_facts = tuple(
        sorted((repr(key), float(value).hex()) for key, value in commit.resolutions.items())
    )
    _update_digest(digest, b"resolutions", repr(resolution_facts).encode())
    _update_digest(
        digest,
        b"forecasts",
        repr(tuple(write.digest for write in commit.forecasts)).encode(),
    )
    _update_digest(digest, b"orders", repr(commit.orders).encode())
    _update_digest(digest, b"settlements", repr(commit.settlements).encode())
    _update_digest(digest, b"state-count", len(commit.state_updates).to_bytes(8, "big"))
    for partition, state in sorted(commit.state_updates.items()):
        _update_digest(digest, b"state-partition", partition.encode())
        _update_digest(digest, b"state-value", state)
    return digest.hexdigest()


def _update_digest(digest, label: bytes, payload: bytes) -> None:
    """Add one unambiguous domain-tagged byte field to a digest."""
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _frozen_nonnegative_quantities[Key](
    values: Mapping[Key, float],
    *,
    name: str,
) -> dict[Key, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[Key, float] = {}
    for key, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(f"{name} must contain real numbers")
        try:
            quantity = float(raw)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain finite non-negative values") from error
        if not math.isfinite(quantity) or quantity < 0.0:
            raise ValueError(f"{name} must contain finite non-negative values")
        normalized[key] = 0.0 if quantity == 0.0 else quantity
    return normalized


__all__ = [
    "ActualKey",
    "ActualsSource",
    "ArtifactStore",
    "CalibrationStateStore",
    "CommitReceipt",
    "DispatchBackend",
    "ForecastWrite",
    "LedgerSink",
    "OriginCommit",
    "PanelSource",
    "SettlementSnapshot",
]
