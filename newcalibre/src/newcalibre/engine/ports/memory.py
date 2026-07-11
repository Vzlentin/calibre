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
    Panel,
    SessionIdentity,
)
from newcalibre.engine.ports import ActualKey, LedgerSnapshot, OriginCommit
from newcalibre.ledger import ForecastRow, Ledger, LedgerError, OrderRow, SettlementRecord

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
        self._actuals = panel.frame[[SERIES_KEY, TIMESTAMP, OBSERVED_VALUE]]

    def before(self, origin: pd.Timestamp) -> Mapping[ActualKey, float]:
        """Return only observations admissible before ``origin``."""
        self._calendar.require_member(origin, name="actuals origin")
        eligible = self._actuals[
            self._actuals[TIMESTAMP].lt(origin) & self._actuals[OBSERVED_VALUE].notna()
        ]
        actuals = {
            (str(series_key), pd.Timestamp(timestamp)): float(value)
            for series_key, timestamp, value in eligible.itertuples(index=False, name=None)
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
        self._states: dict[tuple[SessionIdentity, str], bytes] = {}

    def load(self, session: SessionIdentity, partition: str) -> bytes | None:
        """Return one immutable state value, or ``None``."""
        _require_session_partition(session, partition)
        return self._states.get((session, partition))

    def save(self, session: SessionIdentity, partition: str, value: bytes) -> None:
        """Persist one partition's state."""
        _require_session_partition(session, partition)
        if not isinstance(value, bytes):
            raise TypeError("calibration state must be bytes")
        self._states[(session, partition)] = value

    @property
    def states(self) -> Mapping[tuple[SessionIdentity, str], bytes]:
        """Return an immutable state snapshot for diagnostics."""
        return MappingProxyType(dict(self._states))


class InMemoryLedgerSink:
    """Apply each origin's ledger write atomically against an owned ledger."""

    def __init__(self, ledger: Ledger) -> None:
        if not isinstance(ledger, Ledger):
            raise TypeError("in-memory ledger sink requires a Ledger")
        self._ledger = ledger
        self._forecast_rows = {row.key: row for row in ledger.forecasts}
        self._order_keys = {row.key for row in ledger.orders}
        self._settlement_keys = {row.key for row in ledger.settlements}
        self._commits: dict[pd.Timestamp, OriginCommit] = {}

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

    def snapshot(self) -> LedgerSnapshot:
        """Return immutable facts from all three ledger row families."""
        return LedgerSnapshot(
            session=self._ledger.session,
            calendar=self._ledger.calendar,
            forecasts=self._ledger.forecasts,
            orders=self._ledger.orders,
            settlements=self._ledger.settlements,
        )

    def receipt(self, origin: pd.Timestamp) -> OriginCommit | None:
        """Return the exact immutable receipt for a committed origin."""
        self._ledger.calendar.require_member(origin, name="commit-receipt origin")
        return self._commits.get(origin)

    def commit(self, write: OriginCommit) -> OriginCommit:
        """Journal and publish a write atomically; return its repair receipt."""
        if not isinstance(write, OriginCommit):
            raise TypeError("ledger sink commit requires an OriginCommit")
        if write.session != self._ledger.session:
            raise LedgerError("ledger commit session does not match the sink session")
        previous = self._commits.get(write.origin)
        if previous is not None:
            if _same_commit(previous, write):
                return previous
            raise LedgerError(f"origin {write.origin} already has a different committed write")

        self._validate_resolutions(write)
        staged = _stage_new_rows(write, calendar=self._ledger.calendar)
        _reject_collision(self._forecast_rows, (row.key for row in staged.forecasts), "forecast")
        _reject_collision(self._order_keys, (row.key for row in staged.orders), "order")
        _reject_collision(
            self._settlement_keys,
            (row.key for row in staged.settlements),
            "settlement",
        )

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
        self._commits[write.origin] = write
        return write

    def _validate_resolutions(self, write: OriginCommit) -> None:
        self._ledger.due_frame(write.origin)
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


def _reject_collision(existing: Container[object], staged: Iterable[object], family: str) -> None:
    duplicate = next((key for key in staged if key in existing), None)
    if duplicate is not None:
        raise LedgerError(f"duplicate {family} key: {duplicate!r}")


def _same_commit(left: OriginCommit, right: OriginCommit) -> bool:
    if (
        left.session != right.session
        or left.origin != right.origin
        or left.resolutions != right.resolutions
        or left.orders != right.orders
        or left.settlements != right.settlements
        or left.artifacts != right.artifacts
        or left.state_updates != right.state_updates
        or len(left.forecasts) != len(right.forecasts)
    ):
        return False
    return all(
        left_write.issuances == right_write.issuances and left_write.frame.equals(right_write.frame)
        for left_write, right_write in zip(left.forecasts, right.forecasts, strict=True)
    )


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
