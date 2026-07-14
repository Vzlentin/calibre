"""Drive the one engine over an ordered historical clock."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType

import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    CalendarError,
    InventoryPosition,
    Scope,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine._session import (
    decision_from_definition,
    session_definition,
    session_series_and_frequency,
)
from newcalibre.engine.errors import EngineError
from newcalibre.engine.ports import (
    ActualKey,
    ActualsSource,
    CommitReceipt,
    LedgerSink,
    OriginCommit,
)
from newcalibre.engine.settlement import (
    SettlementError,
    SettlementRequest,
    validate_actuals_window,
    validate_snapshot_state,
)
from newcalibre.engine.spine import (
    Engine,
    OriginRequest,
    PhaseEvent,
    SettlementWindow,
    Spine,
)


class TimeLoopError(EngineError):
    """Report a historical run that cannot honor the engine contract."""


@dataclass(frozen=True, slots=True, init=False)
class TimeLoopRequest:
    """Declare the complete immutable input to one historical engine walk."""

    session: SessionIdentity
    origins: tuple[pd.Timestamp, ...]
    settlement_end: pd.Timestamp
    scope: Scope
    calibration_partitions: tuple[str, ...]
    initial_inventory_positions: Mapping[str, InventoryPosition]
    actuals_semantics: ActualsSemantics

    def __init__(
        self,
        *,
        session: SessionIdentity,
        origins: Sequence[pd.Timestamp],
        settlement_end: pd.Timestamp,
        scope: Scope,
        initial_inventory_positions: Mapping[str, InventoryPosition],
        actuals_semantics: ActualsSemantics,
        calibration_partitions: Sequence[str] = (),
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("time-loop session must be a SessionIdentity")
        if isinstance(origins, (str, bytes)):
            raise TypeError("time-loop origins must be a sequence of timestamps")
        frozen_origins = tuple(origins)
        if not frozen_origins:
            raise TimeLoopError("time-loop origins must not be empty")
        for origin in frozen_origins:
            if not isinstance(origin, pd.Timestamp) or pd.isna(origin):
                raise TypeError("time-loop origins must contain pandas Timestamps")
            if origin.tz is not None:
                raise TimeLoopError("time-loop origins must be timezone-naive")
        if any(current <= previous for previous, current in pairwise(frozen_origins)):
            raise TimeLoopError("time-loop origins must be strictly increasing and unique")
        if not isinstance(settlement_end, pd.Timestamp) or pd.isna(settlement_end):
            raise TypeError("time-loop settlement end must be a pandas Timestamp")
        if settlement_end.tz is not None:
            raise TimeLoopError("time-loop settlement end must be timezone-naive")
        if settlement_end < frozen_origins[-1]:
            raise TimeLoopError("time-loop settlement end must be at or after the last origin")
        if not isinstance(scope, Scope):
            raise TypeError("time-loop scope must be a Scope")
        partitions = tuple(calibration_partitions)
        if len(set(partitions)) != len(partitions):
            raise TimeLoopError("time-loop calibration partitions must be unique")
        for partition in partitions:
            _require_text(partition, name="calibration partition", trimmed=True)
        if not isinstance(initial_inventory_positions, Mapping):
            raise TypeError("initial inventory positions must be a mapping")
        positions: dict[str, InventoryPosition] = {}
        for series_key, position in initial_inventory_positions.items():
            _require_text(series_key, name="inventory series key")
            if not isinstance(position, InventoryPosition):
                raise TypeError("initial inventory positions must contain InventoryPosition values")
            positions[series_key] = position
        if not positions:
            raise TimeLoopError("initial inventory positions must not be empty")
        if not isinstance(actuals_semantics, ActualsSemantics):
            raise TypeError("time-loop actuals semantics must be ActualsSemantics")

        object.__setattr__(self, "session", session)
        object.__setattr__(self, "origins", frozen_origins)
        object.__setattr__(self, "settlement_end", settlement_end)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "calibration_partitions", partitions)
        object.__setattr__(
            self,
            "initial_inventory_positions",
            MappingProxyType(positions),
        )
        object.__setattr__(self, "actuals_semantics", actuals_semantics)


@dataclass(frozen=True, slots=True)
class TimeLoopResult:
    """Return the durable schedule and final inventory facts of a run."""

    settlement_periods: tuple[pd.Timestamp, ...]
    decision_origins: tuple[pd.Timestamp, ...]
    receipts: tuple[CommitReceipt, ...]
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        periods = tuple(self.settlement_periods)
        decisions = tuple(self.decision_origins)
        receipts = tuple(self.receipts)
        if len(periods) != len(receipts):
            raise TimeLoopError("time-loop result requires one receipt per settlement period")
        if any(not isinstance(receipt, CommitReceipt) for receipt in receipts):
            raise TypeError("time-loop result receipts must contain CommitReceipt values")
        positions = dict(self.inventory_positions)
        if any(not isinstance(value, InventoryPosition) for value in positions.values()):
            raise TypeError("time-loop result inventory must contain InventoryPosition values")
        object.__setattr__(self, "settlement_periods", periods)
        object.__setattr__(self, "decision_origins", decisions)
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))


class TimeLoop:
    """Replay ordered origins while the engine remains the only computation path."""

    def __init__(
        self,
        *,
        engine: Engine,
        actuals_source: ActualsSource,
        ledger_sink: LedgerSink,
        request: TimeLoopRequest,
        reporter: Callable[[PhaseEvent], None] | None = None,
    ) -> None:
        if not isinstance(engine, Engine):
            raise TypeError("time loop requires an Engine")
        if not isinstance(actuals_source, ActualsSource):
            raise TypeError("time loop actuals source does not satisfy its port")
        if not isinstance(ledger_sink, LedgerSink):
            raise TypeError("time loop ledger sink does not satisfy its port")
        if not isinstance(request, TimeLoopRequest):
            raise TypeError("time loop requires a TimeLoopRequest")
        if reporter is not None and not callable(reporter):
            raise TypeError("time loop reporter must be callable")
        if actuals_source.actuals_semantics is not request.actuals_semantics:
            raise TimeLoopError("time-loop actuals semantics do not match the actuals source")
        engine._require_time_loop_ports(
            actuals_source=actuals_source,
            ledger_sink=ledger_sink,
        )
        if ledger_sink.session != request.session:
            raise TimeLoopError("time-loop session does not match the ledger sink")

        calendar = ledger_sink.calendar
        for origin in request.origins:
            try:
                calendar.require_member(origin, name="time-loop origin")
            except CalendarError as error:
                raise TimeLoopError(str(error)) from error
        try:
            calendar.require_member(request.settlement_end, name="time-loop settlement end")
        except CalendarError as error:
            raise TimeLoopError(str(error)) from error
        definition = session_definition(request.session)
        _session_series, frequency = session_series_and_frequency(definition)
        if frequency != calendar.frequency:
            raise TimeLoopError("time-loop calendar does not match its session")
        decision = decision_from_definition(definition)
        if decision is None:
            raise TimeLoopError("time loop requires a complete decision configuration")
        series_keys = decision.series_keys
        if set(request.initial_inventory_positions) != set(series_keys):
            raise TimeLoopError(
                "initial inventory positions must exactly match the decision series set"
            )
        timing = decision.timing
        stockout_rule = decision.stockout_rule
        if timing.lead_time < 1:
            raise TimeLoopError("time loop requires a positive decision lead time")
        if stockout_rule is not StockoutRule.LOST_SALES:
            raise TimeLoopError("time loop supports only the lost-sales stock-out rule")

        decision_origins = tuple(
            origin
            for index, origin in enumerate(request.origins)
            if index % timing.review_period == 0
        )
        final_decision = decision_origins[-1]
        drain_target = calendar.advance(final_decision, timing.lead_time)
        if request.settlement_end < drain_target:
            raise TimeLoopError(
                "time-loop settlement end must be at or after the final decision plus lead time"
            )
        final_period = request.settlement_end
        initial_snapshot = ledger_sink.settlement_snapshot((request.origins[0],))
        if (
            initial_snapshot.frontier is not None
            and initial_snapshot.actuals_semantics is not request.actuals_semantics
        ):
            raise TimeLoopError(
                "time-loop actuals semantics do not match the durable settlement state"
            )
        if initial_snapshot.frontier is None:
            try:
                validate_snapshot_state(
                    snapshot=initial_snapshot,
                    positions=request.initial_inventory_positions,
                    series_keys=series_keys,
                    actuals_semantics=request.actuals_semantics,
                )
            except SettlementError as error:
                raise TimeLoopError(f"invalid initial inventory state: {error}") from error
        if initial_snapshot.frontier is not None and initial_snapshot.frontier > final_period:
            raise TimeLoopError(
                f"time-loop end {final_period} precedes durable settlement frontier "
                f"{initial_snapshot.frontier}"
            )
        first_period = request.origins[0]
        if initial_snapshot.frontier is not None and initial_snapshot.frontier < request.origins[0]:
            first_period = calendar.advance(initial_snapshot.frontier, 1)
        frontier_receipt = (
            None
            if initial_snapshot.frontier is None
            else ledger_sink.receipt(initial_snapshot.frontier)
        )
        if initial_snapshot.frontier is not None and frontier_receipt is None:
            raise TimeLoopError(
                f"settlement frontier {initial_snapshot.frontier} has no commit receipt"
            )
        if (
            initial_snapshot.frontier is not None
            and frontier_receipt is not None
            and frontier_receipt.settlement_periods != (initial_snapshot.frontier,)
        ):
            raise TimeLoopError(
                f"settlement frontier receipt at {initial_snapshot.frontier} does not contain "
                "exactly that settlement period"
            )

        self._engine = engine
        self._actuals_source = actuals_source
        self._ledger_sink = ledger_sink
        self._request = request
        self._calendar = calendar
        self._series_keys = series_keys
        self._decision_origins = decision_origins
        self._decision_origin_set = frozenset(decision_origins)
        self._forecast_origins = frozenset(request.origins)
        self._settlement_periods = _calendar_window(
            calendar,
            first_period,
            final_period,
        )
        self._resume_receipt = frontier_receipt
        self._spine = Spine(engine, reporter=reporter)
        uncommitted_periods = tuple(
            period
            for period in self._settlement_periods
            if initial_snapshot.frontier is None or period > initial_snapshot.frontier
        )
        window_due_arrivals = (
            {}
            if not uncommitted_periods
            else ledger_sink.settlement_snapshot(uncommitted_periods).due_arrivals
        )
        _require_open_orders_inside_window(
            open_quantities=initial_snapshot.open_order_quantities,
            due_arrivals=window_due_arrivals,
            series_keys=series_keys,
        )
        self._actuals_by_key = MappingProxyType(
            self._preflight_settlement_actuals(uncommitted_periods)
        )

    def run(self) -> TimeLoopResult:
        """Run or resume the complete schedule from its first uncommitted period."""
        receipts = self._receipt_prefix()
        if self._resume_receipt is not None:
            self._engine.commit(self._resume_receipt)
        for period in self._settlement_periods:
            receipt = receipts.get(period)
            if receipt is not None:
                self._engine.commit(receipt)
                continue
            snapshot = self._ledger_sink.settlement_snapshot((period,))
            positions = (
                self._request.initial_inventory_positions
                if snapshot.frontier is None
                else snapshot.latest_positions
            )
            actuals = self._settlement_actuals(period)
            if period in self._forecast_origins:
                self._spine.run_origin(
                    OriginRequest(
                        session=self._request.session,
                        origin=period,
                        scope=self._request.scope,
                        calibration_partitions=self._request.calibration_partitions,
                        inventory_positions=positions,
                    ),
                    decision_origin=period in self._decision_origin_set,
                    settlement=SettlementWindow(
                        snapshot=snapshot,
                        actuals=actuals,
                        actuals_semantics=self._request.actuals_semantics,
                    ),
                )
                receipt = self._ledger_sink.receipt(period)
                if receipt is None:
                    raise TimeLoopError(f"time loop did not commit settlement period {period}")
            else:
                settled = self._engine.settle(
                    SettlementRequest(
                        session=self._request.session,
                        snapshot=snapshot,
                        actuals=actuals,
                        inventory_positions=positions,
                        orders=(),
                        actuals_semantics=self._request.actuals_semantics,
                    )
                )
                receipt = self._engine.commit(
                    OriginCommit(
                        session=self._request.session,
                        origin=period,
                        settlements=settled.records,
                    )
                )
            if receipt.settlement_periods != (period,):
                raise TimeLoopError(
                    f"commit receipt at {period} does not contain exactly that settlement period"
                )
            receipts[period] = receipt

        next_period = self._calendar.advance(self._settlement_periods[-1], 1)
        final_snapshot = self._ledger_sink.settlement_snapshot((next_period,))
        if any(quantity != 0.0 for quantity in final_snapshot.open_order_quantities.values()):
            raise TimeLoopError("time-loop drain ended with open orders")
        return TimeLoopResult(
            settlement_periods=self._settlement_periods,
            decision_origins=self._decision_origins,
            receipts=tuple(receipts[period] for period in self._settlement_periods),
            inventory_positions=final_snapshot.latest_positions,
        )

    def _preflight_settlement_actuals(
        self,
        periods: tuple[pd.Timestamp, ...],
    ) -> dict[ActualKey, float]:
        if not periods:
            return {}
        keys = tuple((series_key, period) for period in periods for series_key in self._series_keys)
        before = self._calendar.advance(periods[-1], 1)
        supplied = self._actuals_source.for_keys(keys, before=before)
        try:
            return validate_actuals_window(
                supplied,
                periods=periods,
                series_keys=self._series_keys,
            )
        except (SettlementError, TypeError) as error:
            raise TimeLoopError(
                f"settlement actuals do not cover a valid window: {error}"
            ) from error

    def _settlement_actuals(self, period: pd.Timestamp) -> Mapping[ActualKey, float]:
        keys = tuple((series_key, period) for series_key in self._series_keys)
        return MappingProxyType({key: self._actuals_by_key[key] for key in keys})

    def _receipt_prefix(self) -> dict[pd.Timestamp, CommitReceipt]:
        receipts: dict[pd.Timestamp, CommitReceipt] = {}
        first_missing: pd.Timestamp | None = None
        for period in self._settlement_periods:
            receipt = self._ledger_sink.receipt(period)
            if receipt is None:
                if first_missing is None:
                    first_missing = period
                continue
            if receipt.settlement_periods != (period,):
                raise TimeLoopError(
                    f"commit receipt at {period} does not contain exactly that settlement period"
                )
            if first_missing is not None:
                raise TimeLoopError(
                    f"ledger receipt at {period} follows uncommitted period {first_missing}"
                )
            receipts[period] = receipt
        return receipts


def _calendar_window(
    calendar: Calendar,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, ...]:
    periods: list[pd.Timestamp] = []
    current = start
    while current <= end:
        periods.append(current)
        current = calendar.advance(current, 1)
    if not periods or periods[-1] != end:
        raise TimeLoopError("time-loop end period does not lie on its calendar")
    return tuple(periods)


def _require_open_orders_inside_window(
    *,
    open_quantities: Mapping[str, float],
    due_arrivals: Mapping[ActualKey, float],
    series_keys: Sequence[str],
) -> None:
    covered_quantities = {series_key: [] for series_key in series_keys}
    for (series_key, _period), quantity in due_arrivals.items():
        if series_key in covered_quantities:
            covered_quantities[series_key].append(quantity)
    for series_key in series_keys:
        covered_quantity = math.fsum(covered_quantities[series_key])
        if covered_quantity != open_quantities[series_key]:
            raise TimeLoopError("time-loop settlement window must contain every open-order arrival")


def _require_text(value: object, *, name: str, trimmed: bool = False) -> None:
    if not isinstance(value, str) or not value or (trimmed and value != value.strip()):
        qualifier = "non-empty trimmed" if trimmed else "non-empty"
        raise TimeLoopError(f"{name} must be a {qualifier} string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise TimeLoopError(f"{name} must be valid UTF-8") from error


__all__ = [
    "TimeLoop",
    "TimeLoopError",
    "TimeLoopRequest",
    "TimeLoopResult",
]
