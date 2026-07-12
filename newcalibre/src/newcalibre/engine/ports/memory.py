"""Provide in-memory implementations of every engine port."""

from __future__ import annotations

import math
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import TypeVar

import pandas as pd

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    CalendarError,
    Panel,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine._session import (
    session_decision_inputs,
    session_definition,
    session_series_and_frequency,
)
from newcalibre.engine.ports import (
    ActualKey,
    CommitReceipt,
    OriginCommit,
    SettlementSnapshot,
)
from newcalibre.engine.settlement._state import SettlementIndex, SettlementIndexAudit
from newcalibre.ledger import (
    ForecastRow,
    Ledger,
    LedgerError,
    OrderRow,
    SettlementRecord,
)

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


class InMemoryPanelSource:
    """Serve an already validated immutable panel."""

    def __init__(self, panel: Panel) -> None:
        if not isinstance(panel, Panel):
            raise TypeError("in-memory panel source requires a Panel")
        self._panel = panel

    def load(self) -> Panel:
        """Return the immutable panel value."""
        return self._panel


class InMemoryActualsSource:
    """Reveal non-missing panel observations strictly before an origin."""

    def __init__(self, panel: Panel) -> None:
        if not isinstance(panel, Panel):
            raise TypeError("in-memory actuals source requires a Panel")
        self._calendar = panel.calendar
        observed = panel.frame[[SERIES_KEY, TIMESTAMP, OBSERVED_VALUE]].dropna(
            subset=[OBSERVED_VALUE]
        )
        self._actuals = {
            (str(series_key), pd.Timestamp(timestamp)): float(value)
            for series_key, timestamp, value in observed.itertuples(index=False, name=None)
        }

    def for_keys(
        self,
        keys: Sequence[ActualKey],
        *,
        before: pd.Timestamp,
    ) -> Mapping[ActualKey, float]:
        """Look up only requested observations admissible before an origin."""
        self._calendar.require_member(before, name="actuals origin")
        actuals = {
            key: self._actuals[key] for key in keys if key[1] < before and key in self._actuals
        }
        return MappingProxyType(actuals)


class InMemoryArtifactStore:
    """Keep write-once opaque model artifacts in process memory."""

    def __init__(self) -> None:
        self._artifacts: dict[str, bytes] = {}

    def load(self, key: str) -> bytes | None:
        """Return one immutable artifact, or ``None``."""
        _require_key(key, name="artifact key")
        return self._artifacts.get(key)

    def save(self, key: str, value: bytes) -> None:
        """Write an artifact once; an identical retry is idempotent."""
        _require_key(key, name="artifact key")
        if not isinstance(value, bytes):
            raise TypeError("artifact value must be bytes")
        existing = self._artifacts.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"artifact key {key!r} already holds different bytes")
        self._artifacts[key] = value

    @property
    def artifacts(self) -> Mapping[str, bytes]:
        """Return an immutable snapshot for diagnostics."""
        return MappingProxyType(dict(self._artifacts))


class InMemoryCalibrationStateStore:
    """Keep calibration-state bytes by typed session and partition."""

    def __init__(self) -> None:
        self._states: dict[tuple[SessionIdentity, str], tuple[pd.Timestamp, bytes]] = {}

    def load(self, session: SessionIdentity, partition: str) -> bytes | None:
        """Return one immutable state value, or ``None``."""
        _require_session_partition(session, partition)
        stored = self._states.get((session, partition))
        return None if stored is None else stored[1]

    def save(
        self,
        session: SessionIdentity,
        partition: str,
        value: bytes,
        *,
        origin: pd.Timestamp,
    ) -> None:
        """Persist state monotonically by origin with idempotent same-origin retries."""
        _require_session_partition(session, partition)
        if not isinstance(value, bytes):
            raise TypeError("calibration state must be bytes")
        if not isinstance(origin, pd.Timestamp):
            raise TypeError("calibration state origin must be a pandas Timestamp")
        key = (session, partition)
        stored = self._states.get(key)
        if stored is not None:
            stored_origin, stored_value = stored
            if origin < stored_origin:
                return
            if origin == stored_origin and value != stored_value:
                raise ValueError("calibration state origin already holds different bytes")
        self._states[key] = (origin, value)

    @property
    def states(self) -> Mapping[tuple[SessionIdentity, str], bytes]:
        """Return an immutable state snapshot for diagnostics."""
        return MappingProxyType({key: value for key, (_origin, value) in self._states.items()})


class InMemoryLedgerSink:
    """Apply each origin's ledger write atomically against an owned ledger."""

    def __init__(self, *, session: SessionIdentity, calendar: Calendar) -> None:
        self._ledger = Ledger(session=session, calendar=calendar)
        self._forecast_rows: dict[object, ForecastRow] = {}
        self._order_keys: set[object] = set()
        self._settlement_keys: set[object] = set()
        self._commits: dict[pd.Timestamp, CommitReceipt] = {}
        self._decision = session_decision_inputs(session)
        self._series_keys, _frequency = session_series_and_frequency(session_definition(session))
        self._settlement_index = (
            None
            if self._decision is None
            else SettlementIndex(
                session=session,
                calendar=calendar,
                series_keys=self._series_keys,
                cost_structure=self._decision[0],
                timing=self._decision[1],
                stockout_rule=self._decision[2],
            )
        )

    @property
    def session(self) -> SessionIdentity:
        """Return the ledger's typed session."""
        return self._ledger.session

    @property
    def calendar(self) -> Calendar:
        """Return the ledger's bound calendar."""
        return self._ledger.calendar

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        """Return the ledger's defensive due-row snapshot."""
        return self._ledger.due_frame(origin)

    def settlement_snapshot(
        self,
        periods: Sequence[pd.Timestamp],
    ) -> SettlementSnapshot:
        """Return the compact incremental projection for one settlement window."""
        if self._settlement_index is None:
            raise LedgerError("settlement snapshots require a session decision configuration")
        return self._settlement_index.snapshot(periods)

    def rebuild_settlement_index(self) -> SettlementIndexAudit:
        """Rebuild and audit compact settlement state once from durable rows."""
        if self._decision is None:
            raise LedgerError("settlement index rebuild requires a decision configuration")
        self._settlement_index = SettlementIndex.rebuild(
            session=self._ledger.session,
            calendar=self._ledger.calendar,
            series_keys=self._series_keys,
            cost_structure=self._decision[0],
            timing=self._decision[1],
            stockout_rule=self._decision[2],
            orders=self._ledger.orders,
            settlements=self._ledger.settlements,
        )
        return self._settlement_index.audit()

    def settlement_index_audit(self) -> SettlementIndexAudit:
        """Return deterministic work and active-state counts for diagnostics."""
        if self._settlement_index is None:
            raise LedgerError("settlement index audit requires a decision configuration")
        return self._settlement_index.audit()

    def receipt(self, origin: pd.Timestamp) -> CommitReceipt | None:
        """Return the exact immutable receipt for a committed origin."""
        self._ledger.calendar.require_member(origin, name="commit-receipt origin")
        return self._commits.get(origin)

    def commit(self, write: OriginCommit) -> CommitReceipt:
        """Journal and publish a write atomically; return its repair receipt."""
        if not isinstance(write, OriginCommit):
            raise TypeError("ledger sink commit requires an OriginCommit")
        if write.session != self._ledger.session:
            raise LedgerError("ledger commit session does not match the sink session")
        previous = self._commits.get(write.origin)
        if previous is not None:
            if previous.digest == write.digest:
                return previous
            raise LedgerError(f"origin {write.origin} already has a different committed write")

        self._validate_resolutions(write)
        staged = _stage_new_rows(write, calendar=self._ledger.calendar)
        _require_origin_rows(write, staged=staged)
        _reject_collision(self._forecast_rows, (row.key for row in staged.forecasts), "forecast")
        _reject_collision(self._order_keys, (row.key for row in staged.orders), "order")
        _reject_collision(
            self._settlement_keys,
            (row.key for row in staged.settlements),
            "settlement",
        )
        settlement_delta = self._validated_settlement_delta(write)

        # Resolution validates atomically inside Ledger and runs first. Every
        # append below was already validated against a scratch ledger and the
        # owned key indexes, so no later family can partially fail the write.
        if write.resolutions:
            self._ledger.apply_resolutions(write.resolutions, origin=write.origin)
        for forecast in write.forecasts:
            self._ledger.append_forecasts(forecast.frame, issuances=forecast.issuances)
        self._ledger.append_orders(write.orders)
        self._ledger.append_settlements(write.settlements)
        self._forecast_rows.update((row.key, row) for row in staged.forecasts)
        self._order_keys.update(row.key for row in write.orders)
        self._settlement_keys.update(row.key for row in write.settlements)
        if settlement_delta is not None:
            assert self._settlement_index is not None
            self._settlement_index.apply(settlement_delta)
        receipt = CommitReceipt.from_commit(write)
        self._commits[write.origin] = receipt
        return receipt

    def _validate_resolutions(self, write: OriginCommit) -> None:
        try:
            self._ledger.calendar.require_member(write.origin, name="ledger origin")
        except CalendarError as error:
            raise LedgerError(f"ledger origin must lie on the owned calendar: {error}") from error
        for key, value in write.resolutions.items():
            row = self._forecast_rows.get(key)
            if row is None:
                raise LedgerError(f"unknown forecast key: {key!r}")
            if row.actual_value is not None:
                raise LedgerError(f"forecast row is already resolved: {key!r}")
            if row.target_timestamp >= write.origin:
                raise LedgerError(f"forecast row is not yet due: {key!r}")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise LedgerError("resolved actual value must be a real number")
            try:
                normalized = float(value)
            except (OverflowError, TypeError, ValueError) as error:
                raise LedgerError("resolved actual value must be finite") from error
            if not math.isfinite(normalized):
                raise LedgerError("resolved actual value must be finite")

    def _validated_settlement_delta(self, write: OriginCommit):
        """Validate only newly appended settlement facts against the compact index."""
        if not write.orders and not write.settlements:
            return None
        if self._decision is None or self._settlement_index is None:
            noun = "orders" if write.orders else "settlements"
            raise LedgerError(f"durable {noun} require a session decision configuration")
        _cost_structure, timing, stockout_rule = self._decision
        if timing.lead_time < 1:
            noun = "orders" if write.orders else "settlements"
            raise LedgerError(f"durable {noun} require a positive decision lead time")
        if stockout_rule is not StockoutRule.LOST_SALES:
            noun = "order" if write.orders else "settlement"
            raise LedgerError(f"configured {noun} stock-out rule is not supported")
        return self._settlement_index.validate_delta(
            orders=write.orders,
            settlements=write.settlements,
        )

    @property
    def forecasts(self) -> tuple[ForecastRow, ...]:
        """Return forecast rows in stable append order."""
        return self._ledger.forecasts

    @property
    def orders(self) -> tuple[OrderRow, ...]:
        """Return order rows in stable append order."""
        return self._ledger.orders

    @property
    def settlements(self) -> tuple[SettlementRecord, ...]:
        """Return settlement rows in stable append order."""
        return self._ledger.settlements


class InProcessDispatch:
    """Execute work serially and preserve the supplied order."""

    def map(
        self,
        function: Callable[[_Input], _Output],
        items: Sequence[_Input],
    ) -> tuple[_Output, ...]:
        """Apply ``function`` once per item in deterministic order."""
        return tuple(function(item) for item in items)


def _stage_new_rows(write: OriginCommit, *, calendar: Calendar) -> Ledger:
    staged = Ledger(session=write.session, calendar=calendar)
    for forecast in write.forecasts:
        staged.append_forecasts(forecast.frame, issuances=forecast.issuances)
    staged.append_orders(write.orders)
    staged.append_settlements(write.settlements)
    return staged


def _require_origin_rows(write: OriginCommit, *, staged: Ledger) -> None:
    if any(row.origin != write.origin for row in staged.forecasts):
        raise LedgerError("forecast row origin must match its origin commit")
    if any(row.origin != write.origin for row in staged.orders):
        raise LedgerError("order row origin must match its origin commit")


def _reject_collision(existing: Container[object], staged: Iterable[object], family: str) -> None:
    duplicate = next((key for key in staged if key in existing), None)
    if duplicate is not None:
        raise LedgerError(f"duplicate {family} key: {duplicate!r}")


def _require_key(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")


def _require_session_partition(session: object, partition: object) -> None:
    if not isinstance(session, SessionIdentity):
        raise TypeError("calibration state session must be a SessionIdentity")
    _require_key(partition, name="calibration partition")


__all__ = [
    "InMemoryActualsSource",
    "InMemoryArtifactStore",
    "InMemoryCalibrationStateStore",
    "InMemoryLedgerSink",
    "InMemoryPanelSource",
    "InProcessDispatch",
]
