"""Define revision-bound runtime transactions and the indexed store port."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import pandas as pd

from newcalibre.conformal import IssuedBoundFacts
from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    InventoryPosition,
    SessionIdentity,
)
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    OrderRow,
    SettlementRecord,
)
from newcalibre.observe import (
    ActualKey,
    ActualsSubmission,
    ObserveCycle,
    ObservedActual,
    PendingObservation,
)


@dataclass(frozen=True, slots=True, init=False)
class ActualsCommitKey:
    """Identify one actuals transaction by its canonical natural keys."""

    keys: tuple[ActualKey, ...]

    def __init__(self, keys: Sequence[ActualKey]) -> None:
        canonical = _canonical_actual_keys(keys)
        if not canonical:
            raise ValueError("actuals commit key must not be empty")
        object.__setattr__(self, "keys", canonical)


type CommitKey = pd.Timestamp | ActualsCommitKey


@dataclass(frozen=True, slots=True, init=False)
class OriginIntent:
    """Request one read-only origin snapshot at the current store revision."""

    session: SessionIdentity
    origin: pd.Timestamp
    settlement_periods: tuple[pd.Timestamp, ...]

    def __init__(
        self,
        session: SessionIdentity,
        origin: pd.Timestamp,
        settlement_periods: Sequence[pd.Timestamp] = (),
    ) -> None:
        _require_session(session, name="origin intent session")
        _require_timestamp(origin, name="origin intent origin")
        periods = _timestamp_sequence(
            settlement_periods,
            name="origin intent settlement periods",
        )
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "settlement_periods", periods)


@dataclass(frozen=True, slots=True)
class ActualsIntent:
    """Request one read-only snapshot for an atomic actuals submission."""

    session: SessionIdentity
    submission: ActualsSubmission

    def __post_init__(self) -> None:
        _require_session(self.session, name="actuals intent session")
        if not isinstance(self.submission, ActualsSubmission):
            raise TypeError("actuals intent submission must be an ActualsSubmission")
        if not self.submission.records:
            raise ValueError("actuals intent submission must not be empty")

    @property
    def commit_key(self) -> ActualsCommitKey:
        """Return the canonical natural key for the submission."""
        return ActualsCommitKey(tuple(record.key for record in self.submission.records))


@dataclass(frozen=True, slots=True, init=False)
class ForecastWrite:
    """Carry one defensive forecast-frame snapshot into a store commit."""

    _frame: pd.DataFrame = field(repr=False)
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
        observed = {
            key: IssuedBoundFacts.snapshot(value)
            for key, value in (observation_issuances or {}).items()
        }
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "issuances", MappingProxyType(frozen_issuances))
        object.__setattr__(self, "observation_issuances", MappingProxyType(observed))

    @property
    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the staged frame."""
        return self._frame.copy(deep=True)


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
        _require_session(self.session, name="settlement snapshot session")
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
        latest = _frozen_inventory_positions(
            self.latest_positions,
            name="settlement snapshot latest positions",
        )
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
        for values, name in (
            (due_arrivals, "due arrivals"),
            (origin_quantities, "origin-order quantities"),
        ):
            if any(
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or not isinstance(key[1], pd.Timestamp)
                or key[1] not in period_set
                for key in values
            ):
                raise ValueError(f"settlement snapshot {name} must be keyed inside its window")
        if self.actuals_semantics is not None and not isinstance(
            self.actuals_semantics,
            ActualsSemantics,
        ):
            raise TypeError(
                "settlement snapshot actuals semantics must be ActualsSemantics or None"
            )
        current = _frozen_inventory_positions(
            self.current_positions,
            name="settlement snapshot current positions",
        )
        opening = _frozen_inventory_positions(
            self.window_opening_positions,
            name="settlement snapshot window opening positions",
        )
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "latest_positions", MappingProxyType(latest))
        object.__setattr__(self, "open_order_quantities", MappingProxyType(open_quantities))
        object.__setattr__(self, "due_arrivals", MappingProxyType(due_arrivals))
        object.__setattr__(self, "origin_order_quantities", MappingProxyType(origin_quantities))
        object.__setattr__(self, "current_positions", MappingProxyType(current))
        object.__setattr__(self, "window_opening_positions", MappingProxyType(opening))


@dataclass(frozen=True, slots=True)
class OriginCommit:
    """Publish every durable effect produced from one origin snapshot."""

    session: SessionIdentity
    origin: pd.Timestamp
    expected_revision: int
    observe_cycle: ObserveCycle = field(default_factory=ObserveCycle)
    forecasts: tuple[ForecastWrite, ...] = ()
    orders: tuple[OrderRow, ...] = ()
    settlements: tuple[SettlementRecord, ...] = ()
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    checkpoint_updates: Mapping[str, bytes] = field(default_factory=dict)
    checkpoint_indexes: Mapping[str, bytes] = field(default_factory=dict)
    input_fingerprint: str | None = None
    expected_forecast_origin_count: int | None = None
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    resume_marker: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        _validate_commit_common(self)

    @property
    def commit_key(self) -> pd.Timestamp:
        """Return the origin natural key."""
        return self.origin


@dataclass(frozen=True, slots=True)
class ActualsCommit:
    """Publish every durable effect produced from one actuals snapshot."""

    session: SessionIdentity
    origin: pd.Timestamp
    expected_revision: int
    actual_keys: tuple[ActualKey, ...]
    observe_cycle: ObserveCycle = field(default_factory=ObserveCycle)
    settlements: tuple[SettlementRecord, ...] = ()
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    input_fingerprint: str | None = None
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    resume_marker: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        _validate_commit_common(self)
        actual_keys = _canonical_actual_keys(self.actual_keys)
        if not actual_keys:
            raise ValueError("actuals commit keys must not be empty")
        object.__setattr__(self, "actual_keys", actual_keys)

    @property
    def commit_key(self) -> ActualsCommitKey:
        """Return the actuals natural key."""
        return ActualsCommitKey(self.actual_keys)


type RunCommit = OriginCommit | ActualsCommit


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """Bind an accepted transaction to its prior and committed revisions."""

    session: SessionIdentity
    origin: pd.Timestamp
    expected_revision: int
    revision: int
    state_updates: Mapping[str, bytes]
    has_forecasts: bool = False
    observe_cycle: ObserveCycle = field(default_factory=ObserveCycle)
    settlement_periods: tuple[pd.Timestamp, ...] = ()
    actual_keys: tuple[ActualKey, ...] = ()
    input_fingerprint: str | None = None
    orders: tuple[OrderRow, ...] = ()
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    resume_marker: pd.Timestamp | None = None

    @classmethod
    def from_commit(cls, commit: RunCommit, *, revision: int) -> CommitReceipt:
        """Compact one fully materialized transaction into its durable receipt."""
        return cls(
            session=commit.session,
            origin=commit.origin,
            expected_revision=commit.expected_revision,
            revision=revision,
            state_updates=commit.state_updates,
            has_forecasts=isinstance(commit, OriginCommit) and bool(commit.forecasts),
            observe_cycle=commit.observe_cycle,
            settlement_periods=tuple(sorted({record.period for record in commit.settlements})),
            actual_keys=() if isinstance(commit, OriginCommit) else commit.actual_keys,
            input_fingerprint=commit.input_fingerprint,
            orders=() if isinstance(commit, ActualsCommit) else commit.orders,
            inventory_positions=commit.inventory_positions,
            resume_marker=commit.resume_marker,
        )

    def __post_init__(self) -> None:
        _require_session(self.session, name="commit receipt session")
        _require_timestamp(self.origin, name="commit receipt origin")
        _require_revision(self.expected_revision, name="commit receipt expected revision")
        _require_revision(self.revision, name="commit receipt revision")
        if self.revision != self.expected_revision + 1:
            raise ValueError("commit receipt revision must immediately follow its expectation")
        updates = _bytes_mapping(self.state_updates, name="commit receipt state updates")
        if not isinstance(self.has_forecasts, bool):
            raise TypeError("commit receipt has_forecasts must be a bool")
        if not isinstance(self.observe_cycle, ObserveCycle):
            raise TypeError("commit receipt observe cycle must be an ObserveCycle")
        periods = _timestamp_sequence(
            self.settlement_periods,
            name="commit receipt settlement periods",
        )
        if any(current <= previous for previous, current in pairwise(periods)):
            raise ValueError("commit receipt settlement periods must be increasing and unique")
        actual_keys = _canonical_actual_keys(self.actual_keys)
        fingerprint = _optional_sha256(self.input_fingerprint, name="input fingerprint")
        orders = tuple(self.orders)
        if any(not isinstance(order, OrderRow) for order in orders):
            raise TypeError("commit receipt orders must contain OrderRow values")
        positions = _frozen_inventory_positions(
            self.inventory_positions,
            name="commit receipt inventory positions",
        )
        if self.resume_marker is not None:
            _require_timestamp(self.resume_marker, name="commit receipt resume marker")
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        object.__setattr__(self, "settlement_periods", periods)
        object.__setattr__(self, "actual_keys", actual_keys)
        object.__setattr__(self, "input_fingerprint", fingerprint)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))

    @property
    def commit_key(self) -> CommitKey:
        """Return this receipt's natural transaction key."""
        return ActualsCommitKey(self.actual_keys) if self.actual_keys else self.origin


@dataclass(frozen=True, slots=True)
class OriginSnapshot:
    """Expose immutable origin inputs prepared at one store revision."""

    session: SessionIdentity
    origin: pd.Timestamp
    revision: int
    actuals_semantics: ActualsSemantics
    actuals: ActualsSubmission
    observed_history: tuple[ObservedActual, ...]
    pending_observations: tuple[PendingObservation, ...]
    conformal_states: Mapping[str, bytes]
    checkpoints: Mapping[str, bytes]
    checkpoint_indexes: Mapping[str, bytes]
    settlement: SettlementSnapshot | None
    receipt: CommitReceipt | None
    settlement_receipts: Mapping[pd.Timestamp, CommitReceipt]
    earliest_origin: pd.Timestamp | None
    latest_origin: pd.Timestamp | None
    forecast_origin_count: int
    resume_marker: pd.Timestamp | None

    def __post_init__(self) -> None:
        _validate_snapshot(self)


@dataclass(frozen=True, slots=True)
class ActualsSnapshot:
    """Expose immutable actuals inputs prepared at one store revision."""

    session: SessionIdentity
    origin: pd.Timestamp
    revision: int
    actuals_semantics: ActualsSemantics
    actuals: ActualsSubmission
    observed_history: tuple[ObservedActual, ...]
    pending_observations: tuple[PendingObservation, ...]
    conformal_states: Mapping[str, bytes]
    settlement: SettlementSnapshot | None
    receipt: CommitReceipt | None
    earliest_origin: pd.Timestamp | None
    latest_origin: pd.Timestamp | None
    forecast_origin_count: int
    resume_marker: pd.Timestamp | None

    def __post_init__(self) -> None:
        _validate_snapshot(self)


@runtime_checkable
class IndexedRunStore(Protocol):
    """Open immutable revision snapshots and atomically publish their writes."""

    def open(self, intent: OriginIntent | ActualsIntent) -> OriginSnapshot | ActualsSnapshot:
        """Prepare one read-only snapshot without changing visible state."""
        ...

    def commit(self, write: OriginCommit | ActualsCommit) -> CommitReceipt:
        """Atomically publish one revision-bound transaction.

        A natural key admits exactly one transaction. Resubmitting that same
        transaction replays its stored receipt so a lost response is recoverable;
        a write that differs from the stored receipt at a committed key is
        rejected rather than silently discarded, so concurrent writers resolve at
        the store instead of by a read-then-insert race.

        Raises:
            LedgerError: If the write is stale, conflicts with the receipt
                already committed at its natural key, or violates any durable
                invariant the store enforces.
        """
        ...


def _validate_commit_common(commit: OriginCommit | ActualsCommit) -> None:
    _require_session(commit.session, name="run commit session")
    _require_timestamp(commit.origin, name="run commit origin")
    _require_revision(commit.expected_revision, name="run commit expected revision")
    if not isinstance(commit.observe_cycle, ObserveCycle):
        raise TypeError("run commit observe cycle must be an ObserveCycle")
    object.__setattr__(commit, "settlements", tuple(commit.settlements))
    if any(not isinstance(row, SettlementRecord) for row in commit.settlements):
        raise TypeError("run commit settlements must contain SettlementRecord values")
    updates = _bytes_mapping(commit.state_updates, name="run commit state updates")
    for label, value in commit.observe_cycle.state_updates.items():
        if updates.get(label) != value:
            raise ValueError(f"run state updates do not preserve observe update for {label!r}")
    object.__setattr__(commit, "state_updates", MappingProxyType(updates))
    if isinstance(commit, OriginCommit):
        object.__setattr__(commit, "forecasts", tuple(commit.forecasts))
        object.__setattr__(commit, "orders", tuple(commit.orders))
        if any(not isinstance(value, ForecastWrite) for value in commit.forecasts):
            raise TypeError("origin commit forecasts must contain ForecastWrite values")
        if any(not isinstance(value, OrderRow) for value in commit.orders):
            raise TypeError("origin commit orders must contain OrderRow values")
        checkpoints = _bytes_mapping(
            commit.checkpoint_updates,
            name="origin commit checkpoint updates",
        )
        indexes = _bytes_mapping(
            commit.checkpoint_indexes,
            name="origin commit checkpoint indexes",
        )
        object.__setattr__(commit, "checkpoint_updates", MappingProxyType(checkpoints))
        object.__setattr__(commit, "checkpoint_indexes", MappingProxyType(indexes))
        count = commit.expected_forecast_origin_count
        if count is not None and (
            not isinstance(count, Integral) or isinstance(count, bool) or count < 0
        ):
            raise ValueError("expected forecast origin count must be a non-negative integer")
        if count is not None and not commit.forecasts:
            raise ValueError("expected forecast origin count requires forecast rows")
        object.__setattr__(
            commit,
            "expected_forecast_origin_count",
            None if count is None else int(count),
        )
    fingerprint = _optional_sha256(commit.input_fingerprint, name="input fingerprint")
    positions = _frozen_inventory_positions(
        commit.inventory_positions,
        name="run commit inventory positions",
    )
    if commit.resume_marker is not None:
        _require_timestamp(commit.resume_marker, name="run commit resume marker")
    object.__setattr__(commit, "input_fingerprint", fingerprint)
    object.__setattr__(commit, "inventory_positions", MappingProxyType(positions))


def _validate_snapshot(snapshot: OriginSnapshot | ActualsSnapshot) -> None:
    _require_session(snapshot.session, name="run snapshot session")
    _require_timestamp(snapshot.origin, name="run snapshot origin")
    _require_revision(snapshot.revision, name="run snapshot revision")
    if not isinstance(snapshot.actuals_semantics, ActualsSemantics):
        raise TypeError("run snapshot actuals semantics must be ActualsSemantics")
    if not isinstance(snapshot.actuals, ActualsSubmission):
        raise TypeError("run snapshot actuals must be an ActualsSubmission")
    history = tuple(snapshot.observed_history)
    pending = tuple(snapshot.pending_observations)
    if any(not isinstance(value, ObservedActual) for value in history):
        raise TypeError("run snapshot history must contain ObservedActual values")
    if any(not isinstance(value, PendingObservation) for value in pending):
        raise TypeError("run snapshot pending rows must contain PendingObservation values")
    states = _bytes_mapping(snapshot.conformal_states, name="run snapshot conformal states")
    if snapshot.settlement is not None and not isinstance(snapshot.settlement, SettlementSnapshot):
        raise TypeError("run snapshot settlement must be a SettlementSnapshot or None")
    if snapshot.receipt is not None and not isinstance(snapshot.receipt, CommitReceipt):
        raise TypeError("run snapshot receipt must be a CommitReceipt or None")
    for value in (snapshot.earliest_origin, snapshot.latest_origin, snapshot.resume_marker):
        if value is not None:
            _require_timestamp(value, name="run snapshot marker")
    if (
        not isinstance(snapshot.forecast_origin_count, Integral)
        or isinstance(snapshot.forecast_origin_count, bool)
        or snapshot.forecast_origin_count < 0
    ):
        raise ValueError("run snapshot forecast origin count must be non-negative")
    object.__setattr__(snapshot, "observed_history", history)
    object.__setattr__(snapshot, "pending_observations", pending)
    object.__setattr__(snapshot, "conformal_states", MappingProxyType(states))
    if isinstance(snapshot, OriginSnapshot):
        checkpoints = _bytes_mapping(snapshot.checkpoints, name="run snapshot checkpoints")
        indexes = _bytes_mapping(
            snapshot.checkpoint_indexes,
            name="run snapshot checkpoint indexes",
        )
        receipts = dict(snapshot.settlement_receipts)
        if any(
            not isinstance(period, pd.Timestamp) or not isinstance(receipt, CommitReceipt)
            for period, receipt in receipts.items()
        ):
            raise TypeError("run snapshot settlement receipts are invalid")
        object.__setattr__(snapshot, "checkpoints", MappingProxyType(checkpoints))
        object.__setattr__(snapshot, "checkpoint_indexes", MappingProxyType(indexes))
        object.__setattr__(snapshot, "settlement_receipts", MappingProxyType(receipts))


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
        _require_timestamp(timestamp, name="actual key timestamp")
    canonical = tuple(sorted(keys, key=lambda key: (key[0].encode(), key[1])))
    if len(set(canonical)) != len(canonical):
        raise ValueError("actual keys must be unique")
    return canonical


def _optional_sha256(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    _require_digest(value, name=name)
    return value


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 hex string")
    return value


def _bytes_mapping(values: Mapping[str, bytes], *, name: str) -> dict[str, bytes]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    snapshot: dict[str, bytes] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError(f"{name} keys must be non-empty trimmed strings")
        if not isinstance(value, bytes):
            raise TypeError(f"{name} must contain bytes")
        snapshot[key] = value
    return snapshot


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
        quantity = float(raw)
        if not math.isfinite(quantity) or quantity < 0.0:
            raise ValueError(f"{name} must contain finite non-negative values")
        normalized[key] = 0.0 if quantity == 0.0 else quantity
    return normalized


def _timestamp_sequence(values: Sequence[pd.Timestamp], *, name: str) -> tuple[pd.Timestamp, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    snapshot = tuple(values)
    for value in snapshot:
        _require_timestamp(value, name=name)
    return snapshot


def _require_revision(value: object, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_session(value: object, *, name: str) -> SessionIdentity:
    if not isinstance(value, SessionIdentity):
        raise TypeError(f"{name} must be a SessionIdentity")
    return value


def _require_timestamp(value: object, *, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise TypeError(f"{name} must be a pandas Timestamp")
    if value.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")
    return value


__all__ = [
    "ActualKey",
    "ActualsCommit",
    "ActualsCommitKey",
    "ActualsIntent",
    "ActualsSnapshot",
    "CommitKey",
    "CommitReceipt",
    "ForecastWrite",
    "IndexedRunStore",
    "OriginCommit",
    "OriginIntent",
    "OriginSnapshot",
    "SettlementSnapshot",
]
