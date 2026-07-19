"""Drive typed origin and actual events through the closed engine."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    InventoryPosition,
    Scope,
    SessionIdentity,
)
from newcalibre.engine._session import session_decision_inputs
from newcalibre.engine.errors import EngineError
from newcalibre.engine.ports import (
    ActualKey,
    ActualsCommitKey,
    CommitReceipt,
    LedgerSink,
    OriginCommit,
)
from newcalibre.engine.settlement import (
    SettlementRequest,
    SettlementResult,
    validate_snapshot_state,
)
from newcalibre.engine.spine import Engine, OriginRequest, SettlementWindow, Spine
from newcalibre.ledger import OrderRow
from newcalibre.observe import ActualsSubmission, ObservedActual


class EventDriverError(EngineError):
    """Report an event that conflicts with durable session state."""


@dataclass(frozen=True, slots=True, init=False)
class OriginEvent:
    """Declare the caller-owned facts for one forecast origin."""

    session: SessionIdentity
    origin: pd.Timestamp
    scope: Scope
    _future_exogenous: pd.DataFrame | None = field(repr=False)
    _initial_inventory_positions: Mapping[str, InventoryPosition] = field(repr=False)
    _fingerprint: str = field(repr=False)

    def __init__(
        self,
        *,
        session: SessionIdentity,
        origin: pd.Timestamp,
        scope: Scope,
        future_exogenous: pd.DataFrame | None = None,
        initial_inventory_positions: Mapping[str, InventoryPosition] | None = None,
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("origin event session must be a SessionIdentity")
        _require_timestamp(origin, name="origin event origin")
        if not isinstance(scope, Scope):
            raise TypeError("origin event scope must be a Scope")
        if future_exogenous is not None and not isinstance(future_exogenous, pd.DataFrame):
            raise TypeError("origin event future exogenous input must be a pandas DataFrame")
        owned_future = None if future_exogenous is None else future_exogenous.copy(deep=True)
        positions = _inventory_positions(initial_inventory_positions or {})
        fingerprint = _origin_fingerprint(
            session=session,
            origin=origin,
            scope=scope,
            future_exogenous=owned_future,
            inventory_positions=positions,
        )
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "_future_exogenous", owned_future)
        object.__setattr__(
            self,
            "_initial_inventory_positions",
            MappingProxyType(positions),
        )
        object.__setattr__(self, "_fingerprint", fingerprint)

    @property
    def future_exogenous(self) -> pd.DataFrame | None:
        """Return an isolated copy of the future-known facts."""
        if self._future_exogenous is None:
            return None
        return self._future_exogenous.copy(deep=True)

    @property
    def initial_inventory_positions(self) -> Mapping[str, InventoryPosition]:
        """Return the immutable one-time inventory seed."""
        return self._initial_inventory_positions

    @property
    def fingerprint(self) -> str:
        """Return the internally derived canonical input fingerprint."""
        return self._fingerprint


@dataclass(frozen=True, slots=True, init=False)
class ActualsEvent:
    """Carry one non-empty atomic submission for a session."""

    session: SessionIdentity
    submission: ActualsSubmission
    _fingerprint: str = field(repr=False)

    def __init__(self, session: SessionIdentity, submission: ActualsSubmission) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("actuals event session must be a SessionIdentity")
        if not isinstance(submission, ActualsSubmission):
            raise TypeError("actuals event submission must be an ActualsSubmission")
        if not submission.records:
            raise ValueError("actuals event submission must not be empty")
        canonical = ActualsSubmission(
            sorted(
                submission.records,
                key=lambda record: (record.series_key.encode(), record.timestamp),
            )
        )
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "submission", canonical)
        object.__setattr__(
            self,
            "_fingerprint",
            _actuals_fingerprint(session=session, submission=canonical),
        )

    @property
    def fingerprint(self) -> str:
        """Return the internally derived canonical input fingerprint."""
        return self._fingerprint


type DriverEvent = OriginEvent | ActualsEvent


@dataclass(frozen=True, slots=True)
class OriginOutcome:
    """Return the durable origin receipt and its exact materialized orders."""

    receipt: CommitReceipt
    orders: tuple[OrderRow, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, CommitReceipt):
            raise TypeError("origin outcome receipt must be a CommitReceipt")
        orders = tuple(self.orders)
        if orders != self.receipt.orders:
            raise EventDriverError("origin outcome orders must match its durable receipt")
        object.__setattr__(self, "orders", orders)


@dataclass(frozen=True, slots=True)
class ActualsOutcome:
    """Return the durable actuals receipt and resulting settlement periods."""

    receipt: CommitReceipt
    actual_keys: tuple[ActualKey, ...]
    settlement_periods: tuple[pd.Timestamp, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, CommitReceipt):
            raise TypeError("actuals outcome receipt must be a CommitReceipt")
        key = ActualsCommitKey(self.actual_keys)
        if key != self.receipt.commit_key:
            raise EventDriverError("actuals outcome keys must match its durable receipt")
        periods = tuple(self.settlement_periods)
        if periods != self.receipt.settlement_periods:
            raise EventDriverError("actuals outcome periods must match its durable receipt")
        object.__setattr__(self, "actual_keys", key.keys)
        object.__setattr__(self, "settlement_periods", periods)


class EventDriver:
    """Dispatch exactly two event values through one session engine."""

    def __init__(
        self,
        *,
        engine: Engine,
        ledger_sink: LedgerSink,
        actuals_semantics: ActualsSemantics,
    ) -> None:
        if not isinstance(engine, Engine):
            raise TypeError("event driver requires an Engine")
        if not isinstance(ledger_sink, LedgerSink):
            raise TypeError("event driver ledger sink does not satisfy its port")
        if not isinstance(actuals_semantics, ActualsSemantics):
            raise TypeError("event driver actuals semantics must be ActualsSemantics")
        engine._require_event_driver_port(ledger_sink=ledger_sink)
        self._engine = engine
        self._ledger_sink = ledger_sink
        self._actuals_semantics = actuals_semantics
        self._spine = Spine(engine)

    def handle(self, event: DriverEvent) -> OriginOutcome | ActualsOutcome:
        """Validate and handle one typed event."""
        if isinstance(event, OriginEvent):
            return self._handle_origin(event)
        if isinstance(event, ActualsEvent):
            return self._handle_actuals(event)
        raise TypeError("event driver accepts only OriginEvent or ActualsEvent")

    def _handle_origin(self, event: OriginEvent) -> OriginOutcome:
        self._require_session(event.session)
        self._ledger_sink.calendar.require_member(event.origin, name="origin event origin")
        existing = self._ledger_sink.receipt(event.origin)
        if existing is not None:
            self._require_fingerprint(existing, event.fingerprint)
            repaired = self._engine.commit(existing)
            return OriginOutcome(repaired, repaired.orders)

        latest = self._ledger_sink.latest_origin
        if latest is not None and event.origin <= latest:
            raise EventDriverError("origin events must advance strictly monotonically")
        forecast_origin_count = self._ledger_sink.forecast_origin_count
        positions, settlement = self._origin_context(event)
        request = OriginRequest(
            session=event.session,
            origin=event.origin,
            scope=event.scope,
            future_exogenous=event.future_exogenous,
            inventory_positions=positions,
        )
        result = self._spine.run_origin(
            request,
            decision_origin=self._is_decision_origin(
                event.session,
                forecast_origin_count=forecast_origin_count,
            ),
            settlement=settlement,
            submission=ActualsSubmission(()),
            input_fingerprint=event.fingerprint,
            expected_forecast_origin_count=forecast_origin_count,
        )
        return OriginOutcome(result.receipt, result.orders)

    def _handle_actuals(self, event: ActualsEvent) -> ActualsOutcome:
        self._require_session(event.session)
        calendar = self._ledger_sink.calendar
        for record in event.submission.records:
            calendar.require_member(record.timestamp, name="actuals event timestamp")
        key = ActualsCommitKey(tuple(record.key for record in event.submission.records))
        existing = self._ledger_sink.receipt(key)
        if existing is not None:
            self._require_fingerprint(existing, event.fingerprint)
            repaired = self._engine.commit(existing)
            return ActualsOutcome(repaired, key.keys, repaired.settlement_periods)

        latest = self._ledger_sink.latest_origin
        admission_frontier = max(record.timestamp for record in event.submission.records)
        if latest is not None:
            admission_frontier = max(admission_frontier, latest)
        admission_before = calendar.advance(admission_frontier, 1)
        resolution_origin = admission_before if latest is None else calendar.advance(latest, 1)
        observation = self._engine.observe(
            resolution_origin,
            session=event.session,
            submission=event.submission,
        )
        settlement = self._eligible_settlement(observation.cycle.history_appends)
        receipt = self._engine.commit(
            OriginCommit(
                session=event.session,
                origin=admission_before,
                observe_cycle=observation.cycle,
                settlements=() if settlement is None else settlement.records,
                state_updates=observation.cycle.state_updates,
                actual_keys=key.keys,
                input_fingerprint=event.fingerprint,
            )
        )
        return ActualsOutcome(receipt, key.keys, receipt.settlement_periods)

    def _origin_context(
        self,
        event: OriginEvent,
    ) -> tuple[Mapping[str, InventoryPosition], SettlementWindow | None]:
        decision = session_decision_inputs(event.session)
        supplied = event.initial_inventory_positions
        if decision is None:
            if supplied:
                raise EventDriverError(
                    "initial inventory requires a session decision configuration"
                )
            return MappingProxyType({}), None

        snapshot = self._ledger_sink.settlement_snapshot((event.origin,))
        earliest = self._ledger_sink.earliest_origin
        if snapshot.frontier is not None:
            if supplied:
                raise EventDriverError(
                    "initial inventory is rejected after durable settlement begins"
                )
            positions = snapshot.current_positions
        elif earliest is None:
            if not supplied:
                raise EventDriverError("the first origin requires initial inventory")
            validate_snapshot_state(
                snapshot=snapshot,
                positions=supplied,
                series_keys=decision.series_keys,
                actuals_semantics=self._actuals_semantics,
            )
            positions = supplied
        else:
            if supplied:
                raise EventDriverError("initial inventory may seed the session only once")
            if not snapshot.current_positions:
                raise EventDriverError("durable initial inventory is unavailable")
            positions = snapshot.current_positions

        history = {value.key: value for value in self._ledger_sink.observed_history}
        can_settle = all(
            (series_key, event.origin) in history for series_key in decision.series_keys
        )
        if snapshot.frontier is None:
            can_settle = can_settle and earliest is None
        else:
            can_settle = can_settle and event.origin == snapshot.calendar.advance(
                snapshot.frontier,
                1,
            )
        if not can_settle:
            return positions, None
        actuals = {
            (series_key, event.origin): float(history[(series_key, event.origin)].recorded_value)
            for series_key in decision.series_keys
        }
        return positions, SettlementWindow(
            snapshot=snapshot,
            actuals=actuals,
            actuals_semantics=self._actuals_semantics,
        )

    def _eligible_settlement(
        self,
        history_appends: Iterable[ObservedActual],
    ) -> SettlementResult | None:
        decision = session_decision_inputs(self._ledger_sink.session)
        latest = self._ledger_sink.latest_origin
        earliest = self._ledger_sink.earliest_origin
        if decision is None or latest is None or earliest is None:
            return None

        probe = self._ledger_sink.settlement_snapshot((latest,))
        if (
            probe.actuals_semantics is not None
            and probe.actuals_semantics is not self._actuals_semantics
        ):
            raise EventDriverError("event actuals semantics do not match durable settlement state")
        start = earliest if probe.frontier is None else probe.calendar.advance(probe.frontier, 1)
        if start > latest:
            return None

        history = {value.key: value for value in self._ledger_sink.observed_history}
        history.update((value.key, value) for value in history_appends)
        periods: list[pd.Timestamp] = []
        period = start
        while period <= latest:
            if any((series_key, period) not in history for series_key in decision.series_keys):
                break
            periods.append(period)
            period = probe.calendar.advance(period, 1)
        if not periods:
            return None

        snapshot = self._ledger_sink.settlement_snapshot(tuple(periods))
        positions = snapshot.window_opening_positions
        if not positions:
            raise EventDriverError("durable settlement opening inventory is unavailable")
        actuals = {
            (series_key, period): float(history[(series_key, period)].recorded_value)
            for period in periods
            for series_key in decision.series_keys
        }
        return self._engine.settle(
            SettlementRequest(
                session=self._ledger_sink.session,
                snapshot=snapshot,
                actuals=actuals,
                inventory_positions=positions,
                orders=(),
                actuals_semantics=self._actuals_semantics,
            )
        )

    @staticmethod
    def _is_decision_origin(
        session: SessionIdentity,
        *,
        forecast_origin_count: int,
    ) -> bool:
        decision = session_decision_inputs(session)
        if decision is None:
            return True
        return forecast_origin_count % decision.timing.review_period == 0

    def _require_session(self, session: SessionIdentity) -> None:
        if session != self._ledger_sink.session:
            raise EventDriverError("event session does not match the driver session")

    @staticmethod
    def _require_fingerprint(receipt: CommitReceipt, fingerprint: str) -> None:
        if receipt.input_fingerprint != fingerprint:
            raise EventDriverError("natural event identity already has different input facts")


def _origin_fingerprint(
    *,
    session: SessionIdentity,
    origin: pd.Timestamp,
    scope: Scope,
    future_exogenous: pd.DataFrame | None,
    inventory_positions: Mapping[str, InventoryPosition],
) -> str:
    """Derive the canonical origin-event input fingerprint."""
    digest = hashlib.sha256()
    _digest_field(digest, b"schema", b"newcalibre.origin-event/v1")
    _digest_field(digest, b"session", session.value.encode())
    _digest_field(digest, b"origin", _scalar_bytes(origin))
    _digest_field(digest, b"scope", scope.value.encode())
    _digest_field(
        digest,
        b"future-exogenous",
        _scalar_bytes(None) if future_exogenous is None else _frame_bytes(future_exogenous),
    )
    for series_key in sorted(inventory_positions, key=str.encode):
        position = inventory_positions[series_key]
        _digest_field(digest, b"inventory-series", series_key.encode())
        _digest_field(digest, b"inventory-on-hand", struct.pack(">d", position.on_hand))
        _digest_field(digest, b"inventory-on-order", struct.pack(">d", position.on_order))
        _digest_field(digest, b"inventory-backorders", struct.pack(">d", position.backorders))
    return digest.hexdigest()


def _actuals_fingerprint(
    *,
    session: SessionIdentity,
    submission: ActualsSubmission,
) -> str:
    """Derive the canonical actuals-event input fingerprint."""
    digest = hashlib.sha256()
    _digest_field(digest, b"schema", b"newcalibre.actuals-event/v1")
    _digest_field(digest, b"session", session.value.encode())
    for record in submission.records:
        _digest_field(digest, b"series", record.series_key.encode())
        _digest_field(digest, b"timestamp", _scalar_bytes(record.timestamp))
        _digest_field(
            digest,
            b"recorded-value",
            struct.pack(">d", float(record.recorded_value)),
        )
        _digest_field(
            digest,
            b"censoring",
            _scalar_bytes(record.censoring_assertion),
        )
        _digest_field(
            digest,
            b"availability-bound",
            _scalar_bytes(record.availability_bound),
        )
    return digest.hexdigest()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    """Encode frame schema and rows independently of their physical order."""
    if frame.columns.has_duplicates:
        raise EventDriverError("future exogenous input has duplicate column labels")
    columns = tuple(frame.columns)
    if any(not isinstance(column, str) for column in columns):
        raise EventDriverError("future exogenous column labels must be strings")
    canonical_columns = tuple(sorted(columns, key=str.encode))
    encoded = bytearray()
    for column in canonical_columns:
        encoded.extend(_tagged(b"column", column.encode()))
        encoded.extend(_tagged(b"dtype", str(frame[column].dtype).encode()))
    rows = []
    for values in frame.loc[:, canonical_columns].itertuples(index=False, name=None):
        rows.append(b"".join(_tagged(b"value", _scalar_bytes(value)) for value in values))
    for row in sorted(rows):
        encoded.extend(_tagged(b"row", row))
    return _tagged(b"frame", bytes(encoded))


def _scalar_bytes(value: object) -> bytes:
    """Encode one supported scalar with an unambiguous domain tag."""
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return _tagged(b"none", b"")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return _tagged(
            b"timestamp",
            _tagged(b"unit", value.unit.encode()) + _tagged(b"value", value.isoformat().encode()),
        )
    if isinstance(value, Enum):
        return _tagged(b"enum", str(value.value).encode())
    if isinstance(value, bool):
        return _tagged(b"bool", b"1" if value else b"0")
    if isinstance(value, Integral):
        return _tagged(b"int", str(int(value)).encode())
    if isinstance(value, Real):
        return _tagged(b"float64", struct.pack(">d", float(value)))
    if isinstance(value, str):
        return _tagged(b"str", value.encode())
    raise EventDriverError(f"unsupported future exogenous value type: {type(value).__name__}")


def _inventory_positions(
    values: Mapping[str, InventoryPosition],
) -> dict[str, InventoryPosition]:
    if not isinstance(values, Mapping):
        raise TypeError("initial inventory positions must be a mapping")
    positions: dict[str, InventoryPosition] = {}
    for series_key, position in values.items():
        if not isinstance(series_key, str) or not series_key:
            raise EventDriverError("inventory series keys must be non-empty strings")
        try:
            series_key.encode("utf-8")
        except UnicodeError as error:
            raise EventDriverError("inventory series keys must be valid UTF-8") from error
        if not isinstance(position, InventoryPosition):
            raise TypeError("initial inventory positions must contain InventoryPosition values")
        positions[series_key] = position
    return positions


def _require_timestamp(value: object, *, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise TypeError(f"{name} must be a pandas Timestamp")
    if value.tz is not None:
        raise EventDriverError(f"{name} must be timezone-naive")
    return value


def _digest_field(digest, label: bytes, value: bytes) -> None:
    """Append one domain-tagged field to an event fingerprint."""
    digest.update(_tagged(label, value))


def _tagged(label: bytes, value: bytes) -> bytes:
    """Frame one label and value without concatenation ambiguity."""
    return len(label).to_bytes(4, "big") + label + len(value).to_bytes(8, "big") + value


__all__ = [
    "ActualsEvent",
    "ActualsOutcome",
    "DriverEvent",
    "EventDriver",
    "EventDriverError",
    "OriginEvent",
    "OriginOutcome",
]
