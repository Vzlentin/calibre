"""Define the six I/O boundaries used by the engine core."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from itertools import pairwise
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol, TypeVar, runtime_checkable

import pandas as pd

from newcalibre.conformal import IssuedBoundFacts
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
from newcalibre.observe import ActualsSubmission, ObserveCycle, ObservedActual, PendingObservation

type ActualKey = tuple[str, pd.Timestamp]


@dataclass(frozen=True, slots=True, init=False)
class ActualsCommitKey:
    """Identify one atomic actuals transaction by its canonical natural keys."""

    keys: tuple[ActualKey, ...]

    def __init__(self, keys: Sequence[ActualKey]) -> None:
        canonical = _canonical_actual_keys(keys)
        if not canonical:
            raise ValueError("actuals commit key must not be empty")
        object.__setattr__(self, "keys", canonical)


type CommitKey = pd.Timestamp | ActualsCommitKey

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


@dataclass(frozen=True, slots=True, init=False)
class ForecastWrite:
    """Carry one defensive forecast-frame snapshot into a ledger commit."""

    _frame: pd.DataFrame = field(repr=False)
    _digest: str = field(repr=False)
    issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]]
    observation_issuances: Mapping[ForecastKey, IssuedBoundFacts]

    def __init__(
        self,
        frame: pd.DataFrame,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]],
        observation_issuances: Mapping[ForecastKey, IssuedBoundFacts] | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("forecast write frame must be a pandas DataFrame")
        if not isinstance(issuances, Mapping):
            raise TypeError("forecast write issuances must be a mapping")
        if observation_issuances is not None and not isinstance(
            observation_issuances,
            Mapping,
        ):
            raise TypeError("forecast write observation issuances must be a mapping")
        frozen_issuances = {
            key: MappingProxyType(dict(by_bound)) for key, by_bound in issuances.items()
        }
        supplied_observations = {} if observation_issuances is None else observation_issuances
        observed = {
            key: IssuedBoundFacts.snapshot(value) for key, value in supplied_observations.items()
        }
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "issuances", MappingProxyType(frozen_issuances))
        object.__setattr__(self, "observation_issuances", MappingProxyType(observed))
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
    """Journal one complete durable payload for idempotent repair."""

    session: SessionIdentity
    origin: pd.Timestamp
    observe_cycle: ObserveCycle = field(default_factory=ObserveCycle)
    forecasts: tuple[ForecastWrite, ...] = ()
    orders: tuple[OrderRow, ...] = ()
    settlements: tuple[SettlementRecord, ...] = ()
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    actual_keys: tuple[ActualKey, ...] = ()
    input_fingerprint: str | None = None
    expected_forecast_origin_count: int | None = None
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    _digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("origin commit session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("origin commit origin must be a pandas Timestamp")
        if not isinstance(self.observe_cycle, ObserveCycle):
            raise TypeError("origin commit observe cycle must be an ObserveCycle")
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
        actual_keys = _canonical_actual_keys(self.actual_keys)
        fingerprint = _optional_sha256(self.input_fingerprint, name="input fingerprint")
        expected_count = self.expected_forecast_origin_count
        if expected_count is not None and (
            not isinstance(expected_count, Integral)
            or isinstance(expected_count, bool)
            or expected_count < 0
        ):
            raise ValueError("expected forecast origin count must be a non-negative integer")
        if expected_count is not None and not self.forecasts:
            raise ValueError("expected forecast origin count requires forecast rows")
        positions = _frozen_inventory_positions(
            self.inventory_positions,
            name="origin commit inventory positions",
        )
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        object.__setattr__(self, "actual_keys", actual_keys)
        object.__setattr__(self, "input_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "expected_forecast_origin_count",
            None if expected_count is None else int(expected_count),
        )
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        object.__setattr__(self, "_digest", _origin_commit_digest(self))

    @property
    def commit_key(self) -> CommitKey:
        """Return the internally derived journal key for this transaction."""
        if self.actual_keys:
            return ActualsCommitKey(self.actual_keys)
        return self.origin

    @property
    def digest(self) -> str:
        """Return the compact identity of the complete journal payload."""
        return self._digest


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """Retain compact idempotency identity plus pending repair payloads."""

    session: SessionIdentity
    origin: pd.Timestamp
    digest: str
    state_updates: Mapping[str, bytes]
    sequence: int
    has_forecasts: bool = False
    observe_cycle: ObserveCycle = field(default_factory=ObserveCycle)
    settlement_periods: tuple[pd.Timestamp, ...] = ()
    actual_keys: tuple[ActualKey, ...] = ()
    input_fingerprint: str | None = None
    orders: tuple[OrderRow, ...] = ()
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)

    @classmethod
    def from_commit(cls, commit: OriginCommit, *, sequence: int) -> CommitReceipt:
        """Compact a materialized commit after its ledger facts publish."""
        return cls(
            session=commit.session,
            origin=commit.origin,
            digest=commit.digest,
            state_updates=commit.state_updates,
            sequence=sequence,
            has_forecasts=bool(commit.forecasts),
            observe_cycle=commit.observe_cycle,
            settlement_periods=tuple(sorted({record.period for record in commit.settlements})),
            actual_keys=commit.actual_keys,
            input_fingerprint=commit.input_fingerprint,
            orders=commit.orders,
            inventory_positions=commit.inventory_positions,
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
        if (
            not isinstance(self.sequence, Integral)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ValueError("commit receipt sequence must be a non-negative integer")
        if not isinstance(self.has_forecasts, bool):
            raise TypeError("commit receipt has_forecasts must be a bool")
        if not isinstance(self.observe_cycle, ObserveCycle):
            raise TypeError("commit receipt observe cycle must be an ObserveCycle")
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
        actual_keys = _canonical_actual_keys(self.actual_keys)
        fingerprint = _optional_sha256(self.input_fingerprint, name="input fingerprint")
        orders = tuple(self.orders)
        if any(not isinstance(order, OrderRow) for order in orders):
            raise TypeError("commit receipt orders must contain OrderRow values")
        positions = _frozen_inventory_positions(
            self.inventory_positions,
            name="commit receipt inventory positions",
        )
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "settlement_periods", settlement_periods)
        object.__setattr__(self, "actual_keys", actual_keys)
        object.__setattr__(self, "input_fingerprint", fingerprint)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))

    @property
    def commit_key(self) -> CommitKey:
        """Return the internally derived journal key for this transaction."""
        if self.actual_keys:
            return ActualsCommitKey(self.actual_keys)
        return self.origin


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
    origin_order_quantities: Mapping[ActualKey, float] = field(default_factory=dict)
    current_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    window_opening_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)

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
        origin_quantities = _frozen_nonnegative_quantities(
            self.origin_order_quantities,
            name="settlement snapshot origin-order quantities",
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
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not key[0]
            or not isinstance(key[1], pd.Timestamp)
            or key[1] not in period_set
            for key in origin_quantities
        ):
            raise ValueError(
                "settlement snapshot origin-order quantities must be keyed inside its window"
            )
        if self.actuals_semantics is not None and not isinstance(
            self.actuals_semantics,
            ActualsSemantics,
        ):
            raise TypeError(
                "settlement snapshot actuals semantics must be ActualsSemantics or None"
            )
        current_positions = _frozen_inventory_positions(
            self.current_positions,
            name="settlement snapshot current positions",
        )
        opening_positions = _frozen_inventory_positions(
            self.window_opening_positions,
            name="settlement snapshot window opening positions",
        )
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "latest_positions", MappingProxyType(latest_positions))
        object.__setattr__(self, "open_order_quantities", MappingProxyType(open_quantities))
        object.__setattr__(self, "due_arrivals", MappingProxyType(due_arrivals))
        object.__setattr__(
            self,
            "origin_order_quantities",
            MappingProxyType(origin_quantities),
        )
        object.__setattr__(self, "current_positions", MappingProxyType(current_positions))
        object.__setattr__(
            self,
            "window_opening_positions",
            MappingProxyType(opening_positions),
        )


@runtime_checkable
class PanelSource(Protocol):
    """Load the immutable panel that defines a run."""

    def load(self) -> Panel:
        """Return the run's panel."""
        ...


@runtime_checkable
class ActualsSource(Protocol):
    """Reveal actuals admissible strictly before an origin."""

    @property
    def actuals_semantics(self) -> ActualsSemantics:
        """Return the meaning of every observation exposed by this source."""
        ...

    def reveal(self, *, before: pd.Timestamp) -> ActualsSubmission:
        """Return the complete immutable submission admissible before ``before``."""
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

    def load_index(self, key: str) -> bytes | None:
        """Return one mutable artifact-index snapshot, or ``None`` when absent."""
        ...

    def save_index(self, key: str, value: bytes) -> None:
        """Atomically replace one non-authoritative artifact index."""
        ...

    def publish(
        self,
        artifacts: Mapping[str, bytes],
        indexes: Mapping[str, bytes],
    ) -> None:
        """Atomically publish one accepted batch of artifacts and indexes."""
        ...


@runtime_checkable
class CalibrationStateStore(Protocol):
    """Persist calibration state by session and independently addressed label."""

    def snapshot(self, session: SessionIdentity) -> Mapping[str, bytes]:
        """Return one defensive snapshot of every state row for a session."""
        ...

    def save(
        self,
        session: SessionIdentity,
        partition: str,
        value: bytes,
        *,
        sequence: int,
    ) -> None:
        """Persist a partition value without allowing an older journal entry to replace it."""
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

    @property
    def observed_history(self) -> tuple[ObservedActual, ...]:
        """Return a defensive observed-actual history snapshot."""
        ...

    @property
    def pending_observations(self) -> tuple[PendingObservation, ...]:
        """Return a defensive pending-observation snapshot."""
        ...

    @property
    def pending_observation_count(self) -> int:
        """Return the number of pending observations without materializing them."""
        ...

    @property
    def earliest_origin(self) -> pd.Timestamp | None:
        """Return the earliest committed forecast origin, when one exists."""
        ...

    @property
    def latest_origin(self) -> pd.Timestamp | None:
        """Return the latest committed forecast origin, when one exists."""
        ...

    @property
    def forecast_origin_count(self) -> int:
        """Return the number of committed forecast origins."""
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

    def receipt(self, key: CommitKey) -> CommitReceipt | None:
        """Return the immutable commit receipt for a natural journal key, when present."""
        ...

    def settlement_receipt(self, period: pd.Timestamp) -> CommitReceipt | None:
        """Return the receipt containing one durable settlement period, when present."""
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
    _update_digest(digest, b"issuances", _canonical_value_bytes(write.issuances))
    _update_digest(
        digest,
        b"observation-issuances",
        _canonical_value_bytes(write.observation_issuances),
    )
    return digest.hexdigest()


def _origin_commit_digest(commit: OriginCommit) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, b"schema", b"newcalibre.origin-commit/v3")
    _update_digest(digest, b"session", commit.session.value.encode())
    _update_digest(
        digest,
        b"origin",
        f"{commit.origin.isoformat()}:{commit.origin.unit}".encode(),
    )
    cycle = commit.observe_cycle
    _update_digest(digest, b"observe-history", _canonical_value_bytes(cycle.history_appends))
    _update_digest(digest, b"observe-resolutions", _canonical_value_bytes(cycle.resolutions))
    _update_digest(
        digest,
        b"observe-pending-removals",
        _canonical_value_bytes(cycle.pending_removals),
    )
    _update_digest(
        digest,
        b"observe-pending-retentions",
        _canonical_value_bytes(cycle.pending_retentions),
    )
    _update_digest(digest, b"observe-deliveries", _canonical_value_bytes(cycle.deliveries))
    _update_digest(digest, b"observe-annotations", _canonical_value_bytes(cycle.annotations))
    _update_digest(
        digest,
        b"observe-state-updates",
        _canonical_value_bytes(cycle.state_updates),
    )
    _update_digest(
        digest,
        b"forecasts",
        _canonical_value_bytes(tuple(write.digest for write in commit.forecasts)),
    )
    _update_digest(digest, b"orders", _canonical_value_bytes(commit.orders))
    _update_digest(digest, b"settlements", _canonical_value_bytes(commit.settlements))
    _update_digest(digest, b"actual-keys", _canonical_value_bytes(commit.actual_keys))
    _update_digest(
        digest,
        b"input-fingerprint",
        _canonical_value_bytes(commit.input_fingerprint),
    )
    _update_digest(
        digest,
        b"expected-forecast-origin-count",
        _canonical_value_bytes(commit.expected_forecast_origin_count),
    )
    _update_digest(
        digest,
        b"inventory-positions",
        _canonical_value_bytes(commit.inventory_positions),
    )
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


def _canonical_value_bytes(value: object) -> bytes:
    """Encode immutable domain values without repr-dependent aliases."""
    if value is None:
        return _tagged(b"none", b"")
    if isinstance(value, pd.Timestamp):
        payload = f"{value.isoformat()}:{value.unit}".encode()
        return _tagged(b"timestamp", payload)
    if isinstance(value, Enum):
        payload = _tagged(b"type", _type_name(value).encode()) + _canonical_value_bytes(value.value)
        return _tagged(b"enum", payload)
    if isinstance(value, bool):
        return _tagged(b"bool", b"1" if value else b"0")
    if isinstance(value, int):
        return _tagged(b"int", str(value).encode())
    if isinstance(value, float):
        return _tagged(b"float", value.hex().encode())
    if isinstance(value, str):
        return _tagged(b"str", value.encode())
    if isinstance(value, bytes):
        return _tagged(b"bytes", value)
    if is_dataclass(value) and not isinstance(value, type):
        payload = bytearray(_tagged(b"type", _type_name(value).encode()))
        for item in fields(value):
            payload.extend(_tagged(b"field", item.name.encode()))
            payload.extend(_canonical_value_bytes(getattr(value, item.name)))
        return _tagged(b"dataclass", bytes(payload))
    if isinstance(value, Mapping):
        entries = sorted(
            (
                _canonical_value_bytes(key),
                _canonical_value_bytes(item),
            )
            for key, item in value.items()
        )
        payload = b"".join(_tagged(b"key", key) + _tagged(b"value", item) for key, item in entries)
        return _tagged(b"mapping", payload)
    if isinstance(value, (tuple, list)):
        payload = b"".join(_tagged(b"item", _canonical_value_bytes(item)) for item in value)
        kind = b"tuple" if isinstance(value, tuple) else b"list"
        return _tagged(kind, payload)
    raise TypeError(f"unsupported digest value type: {_type_name(value)}")


def _tagged(label: bytes, payload: bytes) -> bytes:
    return len(label).to_bytes(4, "big") + label + len(payload).to_bytes(8, "big") + payload


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_actual_keys(values: Sequence[ActualKey]) -> tuple[ActualKey, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("actual keys must be a sequence")
    keys = tuple(values)
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("actual keys must be series/timestamp tuples")
        series_key, timestamp = key
        if not isinstance(series_key, str) or not series_key:
            raise ValueError("actual key series must be a non-empty string")
        try:
            series_key.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("actual key series must be valid UTF-8") from error
        if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
            raise TypeError("actual key timestamp must be a pandas Timestamp")
        if timestamp.tz is not None:
            raise ValueError("actual key timestamp must be timezone-naive")
    canonical = tuple(sorted(keys, key=lambda key: (key[0].encode(), key[1])))
    if len(set(canonical)) != len(canonical):
        raise ValueError("actual keys must be unique")
    return canonical


def _optional_sha256(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 hex string")
    return value


def _frozen_inventory_positions(
    values: Mapping[str, InventoryPosition],
    *,
    name: str,
) -> dict[str, InventoryPosition]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    positions: dict[str, InventoryPosition] = {}
    for series_key, position in values.items():
        if not isinstance(series_key, str) or not series_key:
            raise ValueError(f"{name} series keys must be non-empty strings")
        if not isinstance(position, InventoryPosition):
            raise TypeError(f"{name} must contain InventoryPosition values")
        positions[series_key] = position
    return positions


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
    "ActualsCommitKey",
    "ActualsSource",
    "ArtifactStore",
    "CalibrationStateStore",
    "CommitKey",
    "CommitReceipt",
    "DispatchBackend",
    "ForecastWrite",
    "LedgerSink",
    "OriginCommit",
    "PanelSource",
    "SettlementSnapshot",
]
