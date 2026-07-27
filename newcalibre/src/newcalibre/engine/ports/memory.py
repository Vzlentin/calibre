"""Provide in-memory implementations of every engine port."""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Iterator, Mapping, Sequence
from threading import RLock
from types import MappingProxyType
from typing import TypeVar, cast

import pandas as pd

from newcalibre.domain import (
    AVAILABILITY_BOUND,
    CENSOR_STATUS,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    UNDECLARED_CENSORING,
    ActualsSemantics,
    Calendar,
    CensoringAssertion,
    EmissionScope,
    InventoryPosition,
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
    ActualsCommitKey,
    CommitKey,
    CommitReceipt,
    OriginCommit,
    SettlementSnapshot,
)
from newcalibre.engine.reporting import (
    LedgerBatch,
    LedgerBoundIssuance,
    LedgerBoundScore,
    LedgerColumn,
    LedgerForecastKey,
    LedgerObservationAnnotation,
    LedgerResolution,
    LedgerSelection,
    LedgerSessionMetadata,
)
from newcalibre.engine.settlement._state import SettlementIndex, SettlementIndexAudit
from newcalibre.ledger import (
    CoverageReport,
    CoverageTarget,
    ForecastKey,
    ForecastRow,
    Ledger,
    LedgerError,
    OrderRow,
    PredicateRegistry,
    SettlementRecord,
    _resolved_window_sum,
    _score_bound,
)
from newcalibre.observe import ActualRecord, ActualsSubmission, ObservedActual, PendingObservation

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")
type _ForecastScanSegment = tuple[pd.Timestamp, tuple[ForecastKey, ...]]


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

    def __init__(
        self,
        panel: Panel,
        *,
        actuals_semantics: ActualsSemantics,
    ) -> None:
        if not isinstance(panel, Panel):
            raise TypeError("in-memory actuals source requires a Panel")
        if not isinstance(actuals_semantics, ActualsSemantics):
            raise TypeError("in-memory actuals semantics must be ActualsSemantics")
        if panel.has_censoring_facts and actuals_semantics is ActualsSemantics.DEMAND:
            raise ValueError("a panel with censoring facts cannot supply demand-honest actuals")
        self._calendar = panel.calendar
        self._actuals_semantics = actuals_semantics
        frame = panel.frame
        records: list[ActualRecord] = []
        for values in frame.to_dict("records"):
            recorded = values[OBSERVED_VALUE]
            if pd.isna(recorded):
                continue
            status = values.get(CENSOR_STATUS, UNDECLARED_CENSORING)
            assertion = None if status == UNDECLARED_CENSORING else CensoringAssertion(status)
            raw_bound = values.get(AVAILABILITY_BOUND)
            bound = None if raw_bound is None or pd.isna(raw_bound) else float(raw_bound)
            records.append(
                ActualRecord(
                    series_key=str(values[SERIES_KEY]),
                    timestamp=pd.Timestamp(values[TIMESTAMP]),
                    recorded_value=recorded,
                    censoring_assertion=assertion,
                    availability_bound=bound,
                )
            )
        self._records = tuple(records)

    @property
    def actuals_semantics(self) -> ActualsSemantics:
        """Return the explicitly bound meaning of the observed values."""
        return self._actuals_semantics

    def reveal(self, *, before: pd.Timestamp) -> ActualsSubmission:
        """Reveal every recorded observation admissible strictly before an origin."""
        self._calendar.require_member(before, name="actuals origin")
        return ActualsSubmission(record for record in self._records if record.timestamp < before)


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
    """Keep calibration-state bytes by typed session and state label."""

    def __init__(self) -> None:
        self._states: dict[tuple[SessionIdentity, str], tuple[int, bytes]] = {}

    def snapshot(self, session: SessionIdentity) -> Mapping[str, bytes]:
        """Return a defensive immutable snapshot of one session's state rows."""
        if not isinstance(session, SessionIdentity):
            raise TypeError("calibration state session must be a SessionIdentity")
        return MappingProxyType(
            {
                label: value
                for (stored_session, label), (_sequence, value) in self._states.items()
                if stored_session == session
            }
        )

    def save(
        self,
        session: SessionIdentity,
        partition: str,
        value: bytes,
        *,
        sequence: int,
    ) -> None:
        """Persist state monotonically by journal sequence with idempotent retries."""
        _require_session_partition(session, partition)
        if not isinstance(value, bytes):
            raise TypeError("calibration state must be bytes")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise TypeError("calibration state sequence must be a non-negative integer")
        key = (session, partition)
        stored = self._states.get(key)
        if stored is not None:
            stored_sequence, stored_value = stored
            if sequence < stored_sequence:
                return
            if sequence == stored_sequence and value != stored_value:
                raise ValueError("calibration state sequence already holds different bytes")
        self._states[key] = (sequence, value)

    @property
    def states(self) -> Mapping[tuple[SessionIdentity, str], bytes]:
        """Return an immutable state snapshot for diagnostics."""
        return MappingProxyType({key: value for key, (_sequence, value) in self._states.items()})


class InMemoryLedgerSink:
    """Apply each origin's ledger write atomically against an owned ledger."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        initial_arrivals: Mapping[ActualKey, float] | None = None,
    ) -> None:
        self._lock = RLock()
        self._ledger = Ledger(session=session, calendar=calendar)
        self._forecast_rows: dict[ForecastKey, ForecastRow] = {}
        self._forecast_scan_segments: list[_ForecastScanSegment] = []
        self._order_keys: set[object] = set()
        self._settlement_keys: set[object] = set()
        self._commits: dict[CommitKey, CommitReceipt] = {}
        self._settlement_receipts: dict[pd.Timestamp, CommitReceipt | None] = {}
        self._next_sequence = 1
        self._earliest_origin: pd.Timestamp | None = None
        self._latest_origin: pd.Timestamp | None = None
        self._forecast_origin_count = 0
        self._decision = session_decision_inputs(session)
        if self._decision is None and initial_arrivals is not None:
            if not isinstance(initial_arrivals, Mapping):
                raise TypeError("initial arrivals must be a mapping")
            if initial_arrivals:
                raise LedgerError("initial arrivals require a session decision configuration")
        session_series, _frequency = session_series_and_frequency(session_definition(session))
        self._series_keys = session_series if self._decision is None else self._decision.series_keys
        self._settlement_index = (
            None
            if self._decision is None
            else SettlementIndex(
                session=session,
                calendar=calendar,
                series_keys=self._series_keys,
                costs_by_series=self._decision.costs_by_series,
                timing=self._decision.timing,
                stockout_rule=self._decision.stockout_rule,
                initial_arrivals=initial_arrivals,
            )
        )
        self._initial_arrivals: Mapping[ActualKey, float] = (
            MappingProxyType({})
            if self._settlement_index is None
            else self._settlement_index.initial_arrivals
        )
        self._initial_positions: Mapping[str, InventoryPosition] = MappingProxyType({})

    @property
    def session(self) -> SessionIdentity:
        """Return the ledger's typed session."""
        return self._ledger.session

    @property
    def calendar(self) -> Calendar:
        """Return the ledger's bound calendar."""
        return self._ledger.calendar

    @property
    def observed_history(self) -> tuple[ObservedActual, ...]:
        """Return the ledger's defensive observed-history snapshot."""
        return self._ledger.observed_history

    @property
    def pending_observations(self) -> tuple[PendingObservation, ...]:
        """Return the ledger's defensive pending-observation snapshot."""
        return self._ledger.pending_observations

    @property
    def pending_observation_count(self) -> int:
        """Return the number of pending observations without materializing them."""
        return self._ledger.pending_observation_count

    @property
    def earliest_origin(self) -> pd.Timestamp | None:
        """Return the earliest committed forecast origin, when one exists."""
        return self._earliest_origin

    @property
    def latest_origin(self) -> pd.Timestamp | None:
        """Return the latest committed forecast origin, when one exists."""
        return self._latest_origin

    @property
    def forecast_origin_count(self) -> int:
        """Return the number of committed forecast origins."""
        return self._forecast_origin_count

    @property
    def observation_resolutions(self):
        """Return durable censoring-aware row resolutions."""
        return self._ledger.observation_resolutions

    @property
    def observe_annotations(self):
        """Return durable observe annotations."""
        return self._ledger.observe_annotations

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
            costs_by_series=self._decision.costs_by_series,
            timing=self._decision.timing,
            stockout_rule=self._decision.stockout_rule,
            initial_arrivals=self._initial_arrivals,
            orders=self._ledger.orders,
            settlements=self._ledger.settlements,
        )
        if not self._ledger.settlements and self._initial_positions:
            self._settlement_index.apply_initial_positions(self._initial_positions)
        return self._settlement_index.audit()

    def settlement_index_audit(self) -> SettlementIndexAudit:
        """Return deterministic work and active-state counts for diagnostics."""
        if self._settlement_index is None:
            raise LedgerError("settlement index audit requires a decision configuration")
        return self._settlement_index.audit()

    def receipt(self, key: CommitKey) -> CommitReceipt | None:
        """Return the exact immutable receipt for a committed natural key."""
        if isinstance(key, pd.Timestamp):
            self._ledger.calendar.require_member(key, name="commit-receipt origin")
        elif isinstance(key, ActualsCommitKey):
            for _series_key, timestamp in key.keys:
                self._ledger.calendar.require_member(
                    timestamp,
                    name="commit-receipt actual timestamp",
                )
        else:
            raise TypeError("commit receipt key must be an origin or actuals key")
        with self._lock:
            return self._commits.get(key)

    def settlement_receipt(self, period: pd.Timestamp) -> CommitReceipt | None:
        """Return the receipt containing one durable settlement period, when present."""
        self._ledger.calendar.require_member(period, name="settlement-receipt period")
        with self._lock:
            receipt = self._settlement_receipts.get(period)
            if period in self._settlement_receipts and receipt is None:
                raise LedgerError(f"settlement period {period} has multiple journal receipts")
            return receipt

    def commit(self, write: OriginCommit) -> CommitReceipt:
        """Journal and publish a write atomically; return its repair receipt."""
        with self._lock:
            return self._commit(write)

    def _commit(self, write: OriginCommit) -> CommitReceipt:
        if not isinstance(write, OriginCommit):
            raise TypeError("ledger sink commit requires an OriginCommit")
        if write.session != self._ledger.session:
            raise LedgerError("ledger commit session does not match the sink session")
        key = write.commit_key
        previous = self._commits.get(key)
        if previous is not None:
            if previous.digest == write.digest:
                return previous
            raise LedgerError(f"journal key {key!r} already has a different committed write")
        if (
            write.forecasts
            and self._latest_origin is not None
            and write.origin <= self._latest_origin
        ):
            raise LedgerError("forecast origins must advance strictly monotonically")

        staged = _stage_new_rows(write, calendar=self._ledger.calendar)
        _require_origin_rows(write, staged=staged)
        staged_forecasts = tuple((row.key, row) for row in staged.forecasts)
        canonical_forecasts = tuple(
            sorted(staged_forecasts, key=lambda item: _forecast_scan_key(item[0]))
        )
        canonical_forecast_keys = tuple(key for key, _row in canonical_forecasts)
        for label, value in write.observe_cycle.state_updates.items():
            if write.state_updates.get(label) != value:
                raise LedgerError(
                    f"origin state updates do not preserve observe update for {label!r}"
                )
        _reject_collision(self._forecast_rows, canonical_forecast_keys, "forecast")
        _reject_collision(self._order_keys, (row.key for row in staged.orders), "order")
        _reject_collision(
            self._settlement_keys,
            (row.key for row in staged.settlements),
            "settlement",
        )
        initial_positions = None
        if self._settlement_index is not None and write.inventory_positions:
            if self._settlement_index.has_initial_positions:
                supplied_positions = dict(write.inventory_positions)
                durable_positions = dict(
                    self._settlement_index.snapshot((write.origin,)).current_positions
                )
                if supplied_positions not in (dict(self._initial_positions), durable_positions):
                    raise LedgerError(
                        "origin inventory positions do not match durable inventory state"
                    )
            else:
                initial_positions = self._settlement_index.validate_initial_positions(
                    write.inventory_positions
                )
        settlement_delta = self._validated_settlement_delta(
            write,
            initial_positions=initial_positions,
        )
        if (
            write.expected_forecast_origin_count is not None
            and write.expected_forecast_origin_count != self._forecast_origin_count
        ):
            raise LedgerError("forecast origin count changed during admission")

        # The observe cycle validates and publishes first. Every later family
        # was prevalidated against scratch/indexed state, so only infallible
        # owned-container updates remain before the receipt becomes observable.
        self._ledger.apply_observe_cycle(write.observe_cycle, origin=write.origin)
        for forecast in write.forecasts:
            self._ledger.append_forecasts(
                forecast.frame,
                issuances=forecast.issuances,
                observation_issuances=forecast.observation_issuances,
            )
        self._ledger.append_orders(write.orders)
        self._ledger.append_settlements(write.settlements)
        self._forecast_rows.update(canonical_forecasts)
        if canonical_forecast_keys:
            self._forecast_scan_segments.append((write.origin, canonical_forecast_keys))
        self._order_keys.update(row.key for row in write.orders)
        self._settlement_keys.update(row.key for row in write.settlements)
        if initial_positions is not None:
            assert self._settlement_index is not None
            self._settlement_index.apply_initial_positions(initial_positions)
            self._initial_positions = MappingProxyType(dict(initial_positions))
        if settlement_delta is not None:
            assert self._settlement_index is not None
            self._settlement_index.apply(settlement_delta)
        receipt = CommitReceipt.from_commit(write, sequence=self._next_sequence)
        self._next_sequence += 1
        self._commits[key] = receipt
        for period in receipt.settlement_periods:
            previous = self._settlement_receipts.get(period)
            if period in self._settlement_receipts and previous != receipt:
                self._settlement_receipts[period] = None
            else:
                self._settlement_receipts[period] = receipt
        if write.forecasts:
            self._forecast_origin_count += 1
            if self._earliest_origin is None or write.origin < self._earliest_origin:
                self._earliest_origin = write.origin
            if self._latest_origin is None or write.origin > self._latest_origin:
                self._latest_origin = write.origin
        return receipt

    def _validated_settlement_delta(
        self,
        write: OriginCommit,
        *,
        initial_positions: Mapping[str, InventoryPosition] | None,
    ):
        """Validate only newly appended settlement facts against the compact index."""
        if not write.orders and not write.settlements:
            return None
        if self._decision is None or self._settlement_index is None:
            noun = "orders" if write.orders else "settlements"
            raise LedgerError(f"durable {noun} require a session decision configuration")
        timing = self._decision.timing
        stockout_rule = self._decision.stockout_rule
        if timing.lead_time < 1:
            noun = "orders" if write.orders else "settlements"
            raise LedgerError(f"durable {noun} require a positive decision lead time")
        if stockout_rule is not StockoutRule.LOST_SALES:
            noun = "order" if write.orders else "settlement"
            raise LedgerError(f"configured {noun} stock-out rule is not supported")
        return self._settlement_index.validate_delta(
            orders=write.orders,
            settlements=write.settlements,
            initial_positions=initial_positions,
        )

    def coverage_report(self) -> CoverageReport:
        """Return the owned ledger's complete registered coverage projection."""
        return self._ledger.coverage_report(PredicateRegistry.gate_a())

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


class InMemoryLedgerReader:
    """Stream bounded logical reporting batches from one closed in-memory sink."""

    def __init__(self, sink: InMemoryLedgerSink) -> None:
        if not isinstance(sink, InMemoryLedgerSink):
            raise TypeError("in-memory ledger reader requires an InMemoryLedgerSink")
        self._sink = sink
        self._metadata = LedgerSessionMetadata(sink.session, sink.session.series_keys)
        self._registry = PredicateRegistry.gate_a()

    @property
    def metadata(self) -> LedgerSessionMetadata:
        """Return immutable identity for the closed sink session."""
        return self._metadata

    def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
        """Return canonical batches after validating the complete selection."""
        if not isinstance(selection, LedgerSelection):
            raise TypeError("ledger scan requires a LedgerSelection")
        if selection.session != self._sink.session:
            raise ValueError("ledger selection session does not match the reader session")
        with self._sink._lock:
            segment_stop = len(self._sink._forecast_scan_segments)
        return self._scan(selection, segment_stop=segment_stop)

    def _scan(
        self,
        selection: LedgerSelection,
        *,
        segment_stop: int,
    ) -> Iterator[LedgerBatch]:
        segment_index = 0
        row_index = 0
        while True:
            with self._sink._lock:
                entries, segment_index, row_index = self._next_entries(
                    selection,
                    segment_index=segment_index,
                    row_index=row_index,
                    segment_stop=segment_stop,
                )
                if not entries:
                    return
                keys: list[LedgerForecastKey] = []
                columns: dict[str, list[object]] = {name: [] for name in selection.columns}
                for forecast_key, row in entries:
                    keys.append(
                        LedgerForecastKey(
                            series_key=row.series_key,
                            origin=row.origin,
                            horizon_step=row.horizon_step,
                            model_name=row.model_name,
                        )
                    )
                    for name, values in columns.items():
                        values.append(
                            self._project_column(
                                name,
                                forecast_key=forecast_key,
                                row=row,
                            )
                        )
                batch = LedgerBatch(
                    session=selection.session,
                    keys=keys,
                    columns=columns,
                    batch_size=selection.batch_size,
                )
            yield batch

    def _next_entries(
        self,
        selection: LedgerSelection,
        *,
        segment_index: int,
        row_index: int,
        segment_stop: int,
    ) -> tuple[list[tuple[ForecastKey, ForecastRow]], int, int]:
        """Collect the next canonical batch from the captured segment prefix."""
        entries: list[tuple[ForecastKey, ForecastRow]] = []
        segments = self._sink._forecast_scan_segments
        while segment_index < segment_stop and len(entries) < selection.batch_size:
            origin, forecast_keys = segments[segment_index]
            if selection.origin_start is not None and origin < selection.origin_start:
                segment_index += 1
                row_index = 0
                continue
            if selection.origin_end is not None and origin > selection.origin_end:
                return entries, segment_stop, 0
            while row_index < len(forecast_keys) and len(entries) < selection.batch_size:
                forecast_key = forecast_keys[row_index]
                entries.append((forecast_key, self._sink._ledger._forecasts[forecast_key]))
                row_index += 1
            if row_index == len(forecast_keys):
                segment_index += 1
                row_index = 0
        return entries, segment_index, row_index

    def _project_column(
        self,
        name: str,
        *,
        forecast_key: ForecastKey,
        row: ForecastRow,
    ) -> object:
        if name == LedgerColumn.SERIES_KEY.value:
            return row.series_key
        if name == LedgerColumn.ORIGIN.value:
            return row.origin
        if name == LedgerColumn.HORIZON_STEP.value:
            return row.horizon_step
        if name == LedgerColumn.MODEL_NAME.value:
            return row.model_name
        if name == LedgerColumn.TARGET_TIMESTAMP.value:
            return row.target_timestamp
        if name == LedgerColumn.POINT_FORECAST.value:
            return row.point_forecast
        if name == LedgerColumn.ISSUANCES.value:
            return self._issuances(row)
        if name == LedgerColumn.RESOLUTION.value:
            return self._resolution(forecast_key)
        if name == LedgerColumn.SCORES.value:
            return self._scores(forecast_key, row=row)
        raise AssertionError(f"unhandled ledger column {name!r}")

    def _issuances(self, row: ForecastRow) -> tuple[LedgerBoundIssuance, ...]:
        values = row.values
        return tuple(
            LedgerBoundIssuance(
                bound_key=bound_key,
                bound_values=cast(
                    tuple[float | None, ...],
                    tuple(values[column] for column in bound_key),
                ),
                descriptor=issuance.descriptor,
                guaranteed_side=(
                    None if issuance.guaranteed_side is None else issuance.guaranteed_side.value
                ),
                calibration_ready=issuance.calibration_ready,
                bounds_finite=issuance.bounds_finite,
                bounds_null_reason=issuance.bounds_null_reason,
            )
            for bound_key, issuance in row.issuances.items()
        )

    def _resolution(self, forecast_key: ForecastKey) -> LedgerResolution | None:
        resolution = self._sink._ledger._resolutions.get(forecast_key)
        if resolution is None:
            return None
        annotation = self._sink._ledger._annotations.get(forecast_key)
        projected_annotation = (
            None
            if annotation is None
            else LedgerObservationAnnotation(
                score=annotation.score,
                exclusion_cause=annotation.exclusion_cause,
                advanced_delivered_score=annotation.advanced_delivered_score,
            )
        )
        return LedgerResolution(
            target_timestamp=resolution.target_timestamp,
            actual_value=resolution.actual,
            censoring_assertion=resolution.censoring_assertion,
            availability_bound=resolution.availability_bound,
            annotation=projected_annotation,
        )

    def _scores(
        self,
        forecast_key: ForecastKey,
        *,
        row: ForecastRow,
    ) -> tuple[LedgerBoundScore, ...]:
        outcomes: list[LedgerBoundScore] = []
        window_sum_actual = (
            _resolved_window_sum(
                self._sink._ledger._forecasts,
                forecast_key=forecast_key,
            )
            if any(
                issuance.bounds_finite and issuance.descriptor.window is EmissionScope.WINDOW_SUM
                for issuance in row.issuances.values()
            )
            else None
        )
        for bound_key, issuance in row.issuances.items():
            target = CoverageTarget(
                descriptor=issuance.descriptor,
                guaranteed_side=issuance.guaranteed_side,
                bound_key=bound_key,
            )
            actual_value = row.actual_value
            if issuance.bounds_finite and target.descriptor.window is EmissionScope.WINDOW_SUM:
                actual_value = window_sum_actual
            outcome = _score_bound(
                forecast_key=forecast_key,
                row=row,
                actual_value=actual_value,
                target=target,
                issuance=issuance,
                annotation=self._sink._ledger._annotations.get(forecast_key),
                registry=self._registry,
            )
            outcomes.append(
                LedgerBoundScore(
                    bound_key=outcome.bound_key,
                    descriptor=outcome.target.descriptor,
                    guaranteed_side=(
                        None
                        if outcome.target.guaranteed_side is None
                        else outcome.target.guaranteed_side.value
                    ),
                    resolved=outcome.resolved,
                    scored=outcome.scored,
                    value=outcome.value,
                    covered=outcome.covered,
                    unscored_reason=outcome.unscored_reason,
                )
            )
        return tuple(outcomes)


class InProcessDispatch:
    """Execute work serially and preserve the supplied order."""

    def map(
        self,
        function: Callable[[_Input], _Output],
        items: Sequence[_Input],
    ) -> tuple[_Output, ...]:
        """Apply ``function`` once per item in deterministic order."""
        return tuple(function(item) for item in items)


def _forecast_scan_key(key: ForecastKey) -> tuple[pd.Timestamp, str, str, int]:
    series_key, origin, horizon_step, model_name = key
    return origin, series_key, model_name, horizon_step


def _stage_new_rows(write: OriginCommit, *, calendar: Calendar) -> Ledger:
    staged = Ledger(session=write.session, calendar=calendar)
    for forecast in write.forecasts:
        staged.append_forecasts(
            forecast.frame,
            issuances=forecast.issuances,
            observation_issuances=forecast.observation_issuances,
        )
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
    "InMemoryLedgerReader",
    "InMemoryLedgerSink",
    "InMemoryPanelSource",
    "InProcessDispatch",
]
