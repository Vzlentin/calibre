"""Provide in-memory panel, dispatch, run-store, and reporting adapters."""

from __future__ import annotations

import json
from bisect import bisect_left, insort
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import cast

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
    HierarchyIndex,
    HistoryCursor,
    InventoryPosition,
    Panel,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.engine._session import (
    session_decision_inputs,
    session_definition,
    session_series_and_frequency,
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
from newcalibre.engine.run_store import (
    ActualKey,
    ActualsCommit,
    ActualsCommitKey,
    ActualsIntent,
    ActualsSnapshot,
    CommitKey,
    CommitReceipt,
    OriginCommit,
    OriginIntent,
    OriginSnapshot,
    SettlementSnapshot,
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
    _ledger_forecast_key,
    _resolved_window_sum,
    _score_bound,
)
from newcalibre.observe import (
    ActualRecord,
    ActualsSubmission,
    ObserveCycle,
    ObservedActual,
    PendingObservation,
)

type _ForecastScanSegment = tuple[pd.Timestamp, tuple[ForecastKey, ...]]
type _PendingGroup = tuple[str, pd.Timestamp, str]

_CHECKPOINT_INDEX_SCHEMA = "newcalibre.forecast-checkpoint-index/v1"


@dataclass(frozen=True, slots=True)
class RunStoreAudit:
    """Report cumulative indexed work without relying on wall-clock timing."""

    origin_opens: int
    actuals_opens: int
    source_rows_examined: int
    target_buckets_examined: int
    pending_rows_examined: int
    history_rows_examined: int
    commits: int
    history_rows_appended: int
    forecast_rows_appended: int
    resolution_rows_applied: int
    staged_rows_validated: int
    due_targets_indexed: int
    checkpoint_indexes_decoded: int


class InMemoryPanelSource:
    """Serve an already validated immutable panel."""

    def __init__(self, panel: Panel) -> None:
        if not isinstance(panel, Panel):
            raise TypeError("in-memory panel source requires a Panel")
        self._panel = panel

    def load(self) -> Panel:
        """Return the immutable panel value."""
        return self._panel


class _IndexedLedgerDataPlane:
    """Maintain indexed ledger and settlement facts behind one store."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        hierarchy: HierarchyIndex,
        initial_arrivals: Mapping[ActualKey, float] | None = None,
    ) -> None:
        self._lock = RLock()
        self._ledger = Ledger(session=session, calendar=calendar)
        self._forecast_rows: dict[ForecastKey, ForecastRow] = {}
        self._forecast_scan_segments: list[_ForecastScanSegment] = []
        self._pending_by_target: dict[
            pd.Timestamp,
            dict[ForecastKey, PendingObservation],
        ] = {}
        self._pending_by_group: dict[_PendingGroup, dict[ForecastKey, PendingObservation]] = {}
        self._pending_by_actual: dict[
            ActualKey,
            dict[ForecastKey, PendingObservation],
        ] = {}
        self._pending_rows: dict[ForecastKey, PendingObservation] = {}
        self._due_targets: list[pd.Timestamp] = []
        self._known_due_targets: set[pd.Timestamp] = set()
        self._due_target_cursor = 0
        self._history_by_timestamp: dict[
            pd.Timestamp,
            dict[ActualKey, ObservedActual],
        ] = {}
        self._order_keys: set[object] = set()
        self._settlement_keys: set[object] = set()
        self._commits: dict[CommitKey, CommitReceipt] = {}
        self._settlement_receipts: dict[pd.Timestamp, CommitReceipt | None] = {}
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
        if not isinstance(hierarchy, HierarchyIndex):
            raise TypeError("in-memory run store hierarchy must be a HierarchyIndex")
        if not set(session_series) <= set(hierarchy.node_labels):
            raise LedgerError("run-store session series are outside its hierarchy")
        self._hierarchy = hierarchy
        self._members_by_node = {node.label: node.members for node in hierarchy.nodes}
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

    def _apply(self, write: OriginCommit | ActualsCommit) -> tuple[int, int]:
        """Validate and apply one unpublished ledger delta."""
        if not isinstance(write, (OriginCommit, ActualsCommit)):
            raise TypeError("run-store data plane requires a transaction value")
        if write.session != self._ledger.session:
            raise LedgerError("run-store commit session does not match the data plane")
        forecasts = () if isinstance(write, ActualsCommit) else write.forecasts
        orders = () if isinstance(write, ActualsCommit) else write.orders
        expected_count = (
            None if isinstance(write, ActualsCommit) else write.expected_forecast_origin_count
        )
        if forecasts and self._latest_origin is not None and write.origin <= self._latest_origin:
            raise LedgerError("forecast origins must advance strictly monotonically")

        staged = _stage_new_rows(write, calendar=self._ledger.calendar)
        _require_origin_rows(write, staged=staged)
        staged_forecasts = tuple((row.key, row) for row in staged.forecasts)
        canonical_forecasts = tuple(
            sorted(staged_forecasts, key=lambda item: _forecast_segment_key(item[0]))
        )
        canonical_forecast_keys = tuple(key for key, _row in canonical_forecasts)
        pending_memberships = self._validated_pending_memberships(
            (*write.observe_cycle.pending_retentions, *staged.pending_observations)
        )
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
        if expected_count is not None and expected_count != self._forecast_origin_count:
            raise LedgerError("forecast origin count changed during admission")

        due_target_count = len(self._known_due_targets)
        # The observe cycle validates and publishes first. Every later family
        # was prevalidated against scratch/indexed state, so only infallible
        # owned-container updates remain before the receipt becomes observable.
        self._ledger.apply_observe_cycle(write.observe_cycle, origin=write.origin)
        self._ledger._publish_staged_rows(staged)
        self._apply_observe_indexes(
            write.observe_cycle,
            pending_memberships=pending_memberships,
        )
        self._forecast_rows.update(canonical_forecasts)
        if canonical_forecast_keys:
            self._forecast_scan_segments.append((write.origin, canonical_forecast_keys))
            for key in canonical_forecast_keys:
                pending = self._ledger._pending_forecasts[key]
                self._index_pending(pending, members=pending_memberships[key])
        self._order_keys.update(row.key for row in orders)
        self._settlement_keys.update(row.key for row in write.settlements)
        if initial_positions is not None:
            assert self._settlement_index is not None
            self._settlement_index.apply_initial_positions(initial_positions)
            self._initial_positions = MappingProxyType(dict(initial_positions))
        if settlement_delta is not None:
            assert self._settlement_index is not None
            self._settlement_index.apply(settlement_delta)
        if forecasts:
            self._forecast_origin_count += 1
            if self._earliest_origin is None or write.origin < self._earliest_origin:
                self._earliest_origin = write.origin
            if self._latest_origin is None or write.origin > self._latest_origin:
                self._latest_origin = write.origin
        return (
            len(staged.forecasts) + len(staged.orders) + len(staged.settlements),
            len(self._known_due_targets) - due_target_count,
        )

    def _apply_observe_indexes(
        self,
        cycle: ObserveCycle,
        *,
        pending_memberships: Mapping[ForecastKey, tuple[str, ...]],
    ) -> None:
        """Apply one already validated observe delta to the lookup indexes."""
        for value in cycle.history_appends:
            self._history_by_timestamp.setdefault(value.timestamp, {})[value.key] = value
        for forecast_key in cycle.pending_removals:
            self._remove_pending(_ledger_forecast_key(forecast_key))
        for retained in cycle.pending_retentions:
            key = _ledger_forecast_key(retained.forecast_key)
            self._index_pending(retained, members=pending_memberships[key])

    def _validated_pending_memberships(
        self,
        pending_rows: Iterable[PendingObservation],
    ) -> dict[ForecastKey, tuple[str, ...]]:
        """Bind every pending row to known bottom members before publication."""
        memberships: dict[ForecastKey, tuple[str, ...]] = {}
        for pending in pending_rows:
            key = _ledger_forecast_key(pending.forecast_key)
            try:
                memberships[key] = self._members_by_node[pending.forecast_key.series_key]
            except KeyError as error:
                raise LedgerError(
                    "pending forecast names unknown hierarchy node "
                    f"{pending.forecast_key.series_key!r}"
                ) from error
        return memberships

    def _index_pending(
        self,
        pending: PendingObservation,
        *,
        members: tuple[str, ...],
    ) -> None:
        """Insert or replace one pending row in its target and lineage indexes."""
        key = _ledger_forecast_key(pending.forecast_key)
        target = pending.target_timestamp
        self._pending_rows[key] = pending
        self._pending_by_target.setdefault(target, {})[key] = pending
        self._pending_by_group.setdefault(_pending_group(pending), {})[key] = pending
        for member in members:
            self._pending_by_actual.setdefault((member, target), {})[key] = pending
        if target not in self._known_due_targets:
            insort(self._due_targets, target, lo=self._due_target_cursor)
            self._known_due_targets.add(target)

    def _remove_pending(self, key: ForecastKey) -> None:
        """Remove one addressed pending row without scanning unrelated rows."""
        pending = self._pending_rows.pop(key)
        target_rows = self._pending_by_target[pending.target_timestamp]
        target_rows.pop(key)
        if not target_rows:
            self._pending_by_target.pop(pending.target_timestamp)
        group = _pending_group(pending)
        group_rows = self._pending_by_group[group]
        group_rows.pop(key)
        if not group_rows:
            self._pending_by_group.pop(group)
        for member in self._members_by_node[pending.forecast_key.series_key]:
            actual_rows = self._pending_by_actual[(member, pending.target_timestamp)]
            actual_rows.pop(key)
            if not actual_rows:
                self._pending_by_actual.pop((member, pending.target_timestamp))

    def _validated_settlement_delta(
        self,
        write: OriginCommit | ActualsCommit,
        *,
        initial_positions: Mapping[str, InventoryPosition] | None,
    ):
        """Validate only newly appended settlement facts against the compact index."""
        orders = () if isinstance(write, ActualsCommit) else write.orders
        if not orders and not write.settlements:
            return None
        if self._decision is None or self._settlement_index is None:
            noun = "orders" if orders else "settlements"
            raise LedgerError(f"durable {noun} require a session decision configuration")
        timing = self._decision.timing
        stockout_rule = self._decision.stockout_rule
        if timing.lead_time < 1:
            noun = "orders" if orders else "settlements"
            raise LedgerError(f"durable {noun} require a positive decision lead time")
        if stockout_rule is not StockoutRule.LOST_SALES:
            noun = "order" if orders else "settlement"
            raise LedgerError(f"configured {noun} stock-out rule is not supported")
        return self._settlement_index.validate_delta(
            orders=orders,
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
    def logical_forecasts(self) -> tuple[ForecastRow, ...]:
        """Join resolution facts into terminal protocol-facing forecast values."""
        return tuple(
            row
            if (resolution := self._ledger._resolutions.get(row.key)) is None
            else row._with_actual_value(resolution.actual)
            for row in self._ledger._forecasts.values()
        )

    @property
    def orders(self) -> tuple[OrderRow, ...]:
        """Return order rows in stable append order."""
        return self._ledger.orders

    @property
    def settlements(self) -> tuple[SettlementRecord, ...]:
        """Return settlement rows in stable append order."""
        return self._ledger.settlements


class InMemoryIndexedRunStore(_IndexedLedgerDataPlane):
    """Own one session's revisioned actuals, checkpoints, state, and ledger facts."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        actuals: Panel | None = None,
        actuals_semantics: ActualsSemantics,
        hierarchy: HierarchyIndex | None = None,
        initial_arrivals: Mapping[ActualKey, float] | None = None,
    ) -> None:
        if not isinstance(actuals_semantics, ActualsSemantics):
            raise TypeError("in-memory run-store semantics must be ActualsSemantics")
        session_series, _frequency = session_series_and_frequency(session_definition(session))
        super().__init__(
            session=session,
            calendar=calendar,
            hierarchy=hierarchy or HierarchyIndex.flat(session_series),
            initial_arrivals=initial_arrivals,
        )
        source_records = (
            ()
            if actuals is None
            else _panel_actual_records(actuals, actuals_semantics=actuals_semantics)
        )
        self._source_buckets = _actual_record_buckets(source_records)
        self._source_cursor = 0
        self._actuals_semantics = actuals_semantics
        self._revision = 1
        self._states: dict[str, bytes] = {}
        self._checkpoints: dict[str, bytes] = {}
        self._checkpoint_indexes: dict[str, bytes] = {}
        self._checkpoint_keys_by_index: dict[str, str | None] = {}
        self._resume_marker: pd.Timestamp | None = None
        self._audit_counts = {
            "origin_opens": 0,
            "actuals_opens": 0,
            "source_rows_examined": 0,
            "target_buckets_examined": 0,
            "pending_rows_examined": 0,
            "history_rows_examined": 0,
            "commits": 0,
            "history_rows_appended": 0,
            "forecast_rows_appended": 0,
            "resolution_rows_applied": 0,
            "staged_rows_validated": 0,
            "due_targets_indexed": 0,
            "checkpoint_indexes_decoded": 0,
        }

    def open(self, intent: OriginIntent | ActualsIntent) -> OriginSnapshot | ActualsSnapshot:
        """Prepare one immutable transaction snapshot without publishing state."""
        if not isinstance(intent, (OriginIntent, ActualsIntent)):
            raise TypeError("run-store open requires an OriginIntent or ActualsIntent")
        if intent.session != self.session:
            raise LedgerError("run-store intent session does not match the store session")
        with self._lock:
            if isinstance(intent, OriginIntent):
                self.calendar.require_member(intent.origin, name="run-store origin")
                actuals, source_rows_examined = self._source_delta(intent.origin)
                due_targets = tuple(
                    self._due_targets[
                        self._due_target_cursor : bisect_left(
                            self._due_targets,
                            intent.origin,
                            lo=self._due_target_cursor,
                        )
                    ]
                )
                pending, target_buckets_examined, pending_rows_examined = self._pending_snapshot(
                    due_targets=due_targets,
                    actual_keys=(record.key for record in actuals.records),
                )
                settlement = (
                    None
                    if not intent.settlement_periods
                    else self.settlement_snapshot(intent.settlement_periods)
                )
                history_periods = {
                    *(value.target_timestamp for value in pending),
                    *intent.settlement_periods,
                }
                history, history_rows_examined = self._history_snapshot(history_periods)
                checkpoints = self._active_checkpoints()
                self._record_audit(
                    origin_opens=1,
                    source_rows_examined=source_rows_examined,
                    target_buckets_examined=target_buckets_examined,
                    pending_rows_examined=pending_rows_examined,
                    history_rows_examined=history_rows_examined,
                )
                receipt_periods = set(intent.settlement_periods)
                if settlement is not None and settlement.frontier is not None:
                    receipt_periods.add(settlement.frontier)
                settlement_receipts = {
                    period: receipt
                    for period in receipt_periods
                    if (receipt := self.settlement_receipt(period)) is not None
                }
                return OriginSnapshot(
                    session=self.session,
                    origin=intent.origin,
                    revision=self._revision,
                    actuals_semantics=self._actuals_semantics,
                    actuals=actuals,
                    observed_history=history,
                    pending_observations=pending,
                    conformal_states=self._states,
                    checkpoints=checkpoints,
                    checkpoint_indexes=self._checkpoint_indexes,
                    settlement=settlement,
                    receipt=self._commits.get(intent.origin),
                    settlement_receipts=settlement_receipts,
                    earliest_origin=self.earliest_origin,
                    latest_origin=self.latest_origin,
                    forecast_origin_count=self.forecast_origin_count,
                    resume_marker=self._resume_marker,
                )

            key = intent.commit_key
            for _series_key, timestamp in key.keys:
                self.calendar.require_member(timestamp, name="run-store actual timestamp")
            admission_frontier = max(timestamp for _series_key, timestamp in key.keys)
            if self.latest_origin is not None:
                admission_frontier = max(admission_frontier, self.latest_origin)
            origin = self.calendar.advance(admission_frontier, 1)
            settlement = self._actuals_settlement_snapshot(intent)
            pending, target_buckets_examined, pending_rows_examined = self._pending_snapshot(
                due_targets=(),
                actual_keys=(record.key for record in intent.submission.records),
            )
            history_periods = {
                *(value.target_timestamp for value in pending),
                *(record.timestamp for record in intent.submission.records),
                *(() if settlement is None else settlement.periods),
            }
            history, history_rows_examined = self._history_snapshot(history_periods)
            self._record_audit(
                actuals_opens=1,
                target_buckets_examined=target_buckets_examined,
                pending_rows_examined=pending_rows_examined,
                history_rows_examined=history_rows_examined,
            )
            return ActualsSnapshot(
                session=self.session,
                origin=origin,
                revision=self._revision,
                actuals_semantics=self._actuals_semantics,
                actuals=intent.submission,
                observed_history=history,
                pending_observations=pending,
                conformal_states=self._states,
                settlement=settlement,
                receipt=self._commits.get(key),
                earliest_origin=self.earliest_origin,
                latest_origin=self.latest_origin,
                forecast_origin_count=self.forecast_origin_count,
                resume_marker=self._resume_marker,
            )

    def _actuals_settlement_snapshot(
        self,
        intent: ActualsIntent,
    ) -> SettlementSnapshot | None:
        """Prepare the contiguous newly complete settlement window."""
        if self._decision is None or self.earliest_origin is None or self.latest_origin is None:
            return None
        probe = self.settlement_snapshot((self.latest_origin,))
        start = (
            self.earliest_origin
            if probe.frontier is None
            else self.calendar.advance(probe.frontier, 1)
        )
        if start > self.latest_origin:
            return None
        submitted = {record.key for record in intent.submission.records}
        periods: list[pd.Timestamp] = []
        period = start
        while period <= self.latest_origin:
            if any(
                (series_key, period) not in submitted
                and (series_key, period) not in self._ledger._observed_history
                for series_key in self._series_keys
            ):
                break
            periods.append(period)
            period = self.calendar.advance(period, 1)
        if not periods:
            return None
        return self.settlement_snapshot(periods)

    def commit(
        self,
        write: OriginCommit | ActualsCommit,
    ) -> CommitReceipt:
        """Validate and publish one complete transaction at the expected revision."""
        if not isinstance(write, (OriginCommit, ActualsCommit)):
            raise TypeError("run-store commit requires an OriginCommit or ActualsCommit")
        if write.session != self.session:
            raise LedgerError("run-store commit session does not match the store session")
        with self._lock:
            # The natural key is the transaction identity, so a retry replays its
            # stored receipt without re-examining the payload. Checking the
            # revision first would instead reject honest retries, which always
            # carry the pre-commit revision.
            previous = self._commits.get(write.commit_key)
            if previous is not None:
                return previous
            if write.expected_revision != self._revision:
                raise LedgerError(
                    "run-store commit revision is stale: "
                    f"expected {write.expected_revision}, current {self._revision}"
                )
            checkpoint_updates = (
                {} if isinstance(write, ActualsCommit) else dict(write.checkpoint_updates)
            )
            checkpoint_indexes = (
                {} if isinstance(write, ActualsCommit) else dict(write.checkpoint_indexes)
            )
            for key, value in checkpoint_updates.items():
                existing = self._checkpoints.get(key)
                if existing is not None and existing != value:
                    raise LedgerError(f"checkpoint key {key!r} already holds different bytes")

            decoded_indexes = {
                key: _checkpoint_key_from_index(value) for key, value in checkpoint_indexes.items()
            }
            available_checkpoints = set(self._checkpoints) | set(checkpoint_updates)
            for index_key, checkpoint_key in decoded_indexes.items():
                if checkpoint_key not in available_checkpoints:
                    raise LedgerError(
                        f"checkpoint index {index_key!r} names missing checkpoint "
                        f"{checkpoint_key!r}"
                    )
            revision = self._revision + 1
            receipt = CommitReceipt.from_commit(write, revision=revision)

            staged_rows_validated, due_targets_indexed = super()._apply(write)
            self._states.update(write.state_updates)
            self._checkpoints.update(checkpoint_updates)
            self._checkpoint_indexes.update(checkpoint_indexes)
            self._checkpoint_keys_by_index.update(decoded_indexes)
            self._resume_marker = write.resume_marker
            self._revision = revision
            self._commits[write.commit_key] = receipt
            for period in receipt.settlement_periods:
                previous = self._settlement_receipts.get(period)
                if period in self._settlement_receipts and previous != receipt:
                    self._settlement_receipts[period] = None
                else:
                    self._settlement_receipts[period] = receipt
            source_rows_examined = 0
            if isinstance(write, OriginCommit):
                self._source_cursor, source_rows_examined = self._advanced_source_cursor(
                    write.origin
                )
                self._due_target_cursor = bisect_left(
                    self._due_targets,
                    write.origin,
                    lo=self._due_target_cursor,
                )
                if self._due_target_cursor:
                    crossed = self._due_targets[: self._due_target_cursor]
                    del self._due_targets[: self._due_target_cursor]
                    self._known_due_targets.difference_update(crossed)
                    self._due_target_cursor = 0
            self._record_audit(
                commits=1,
                source_rows_examined=source_rows_examined,
                history_rows_appended=len(write.observe_cycle.history_appends),
                forecast_rows_appended=(
                    0
                    if isinstance(write, ActualsCommit)
                    else sum(len(value._frame) for value in write.forecasts)
                ),
                resolution_rows_applied=len(write.observe_cycle.resolutions),
                staged_rows_validated=staged_rows_validated,
                due_targets_indexed=due_targets_indexed,
                checkpoint_indexes_decoded=len(checkpoint_indexes),
            )
            return receipt

    def audit(self) -> RunStoreAudit:
        """Return an immutable snapshot of cumulative indexed work."""
        with self._lock:
            return RunStoreAudit(**self._audit_counts)

    def _source_delta(self, origin: pd.Timestamp) -> tuple[ActualsSubmission, int]:
        """Read only newly eligible source buckets without advancing their cursor."""
        records: list[ActualRecord] = []
        examined = 0
        index = self._source_cursor
        while index < len(self._source_buckets):
            timestamp, bucket = self._source_buckets[index]
            if timestamp >= origin:
                break
            examined += len(bucket)
            records.extend(
                value for value in bucket if value.key not in self._ledger._observed_history
            )
            index += 1
        return ActualsSubmission(records), examined

    def _advanced_source_cursor(self, origin: pd.Timestamp) -> tuple[int, int]:
        """Advance through eligible buckets whose records are now all durable."""
        index = self._source_cursor
        examined = 0
        while index < len(self._source_buckets):
            timestamp, bucket = self._source_buckets[index]
            if timestamp >= origin:
                break
            complete = True
            for value in bucket:
                examined += 1
                if value.key not in self._ledger._observed_history:
                    complete = False
                    break
            if not complete:
                break
            index += 1
        return index, examined

    def _pending_snapshot(
        self,
        *,
        due_targets: Iterable[pd.Timestamp],
        actual_keys: Iterable[ActualKey],
    ) -> tuple[tuple[PendingObservation, ...], int, int]:
        """Read crossed targets, affected memberships, and bounded lineages."""
        canonical_targets = tuple(sorted(set(due_targets)))
        canonical_actuals = tuple(
            sorted(set(actual_keys), key=lambda key: (key[1], key[0].encode()))
        )
        rows: dict[ForecastKey, PendingObservation] = {}
        for target in canonical_targets:
            rows.update(self._pending_by_target.get(target, {}))
        for actual_key in canonical_actuals:
            rows.update(self._pending_by_actual.get(actual_key, {}))
        for group in {_pending_group(value) for value in rows.values()}:
            rows.update(self._pending_by_group[group])
        pending = tuple(
            value
            for _key, value in sorted(
                rows.items(),
                key=lambda item: (
                    item[1].target_timestamp,
                    item[0][0].encode(),
                    item[0][3].encode(),
                    item[0][1],
                    item[0][2],
                ),
            )
        )
        addressed_targets = set(canonical_targets) | {key[1] for key in canonical_actuals}
        return pending, len(addressed_targets), len(pending)

    def _history_snapshot(
        self,
        periods: Iterable[pd.Timestamp],
    ) -> tuple[tuple[ObservedActual, ...], int]:
        """Read observed facts only for the addressed target and settlement periods."""
        history: dict[ActualKey, ObservedActual] = {}
        for period in sorted(set(periods)):
            history.update(self._history_by_timestamp.get(period, {}))
        values = tuple(
            value
            for _key, value in sorted(
                history.items(),
                key=lambda item: (item[0][1], item[0][0].encode()),
            )
        )
        return values, len(values)

    def _active_checkpoints(self) -> Mapping[str, bytes]:
        """Project only checkpoint blobs named by the compact lineage indexes."""
        keys = {key for key in self._checkpoint_keys_by_index.values() if key is not None}
        return {key: self._checkpoints[key] for key in keys if key in self._checkpoints}

    def _record_audit(self, **deltas: int) -> None:
        """Accumulate diagnostic work after a completed read or publication."""
        for name, value in deltas.items():
            self._audit_counts[name] += value

    @property
    def revision(self) -> int:
        """Return the current monotonic revision for diagnostics."""
        return self._revision

    @property
    def resume_marker(self) -> pd.Timestamp | None:
        """Return the latest durable resume marker for diagnostics."""
        return self._resume_marker

    @property
    def receipts(self) -> Mapping[CommitKey, CommitReceipt]:
        """Return the immutable natural-key journal for diagnostics."""
        return MappingProxyType(dict(self._commits))

    @property
    def settlement_receipts(self) -> Mapping[pd.Timestamp, CommitReceipt | None]:
        """Return immutable settlement-to-receipt journal links for diagnostics."""
        return MappingProxyType(dict(self._settlement_receipts))

    @property
    def states(self) -> Mapping[str, bytes]:
        """Return immutable conformal rows for diagnostics."""
        return MappingProxyType(dict(self._states))

    @property
    def state_footprint(self) -> tuple[int, int]:
        """Return committed conformal ``(row_count, total_bytes)`` without copying.

        Sized for repeated per-origin measurement where :attr:`states` would
        materialize a whole second mapping and perturb what it measures.
        """
        return len(self._states), sum(map(len, self._states.values()))

    @property
    def checkpoints(self) -> Mapping[str, bytes]:
        """Return immutable model checkpoints for diagnostics."""
        return MappingProxyType(dict(self._checkpoints))

    @property
    def checkpoint_indexes(self) -> Mapping[str, bytes]:
        """Return immutable checkpoint indexes for diagnostics."""
        return MappingProxyType(dict(self._checkpoint_indexes))


class InMemoryLedgerReader:
    """Join immutable store segments and resolution facts into reporting batches."""

    def __init__(self, store: InMemoryIndexedRunStore) -> None:
        if not isinstance(store, InMemoryIndexedRunStore):
            raise TypeError("in-memory ledger reader requires an InMemoryIndexedRunStore")
        self._store = store
        self._metadata = LedgerSessionMetadata(store.session, store.session.series_keys)
        self._registry = PredicateRegistry.gate_a()

    @property
    def metadata(self) -> LedgerSessionMetadata:
        """Return immutable identity for the closed store session."""
        return self._metadata

    def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
        """Return canonical batches after validating the complete selection."""
        if not isinstance(selection, LedgerSelection):
            raise TypeError("ledger scan requires a LedgerSelection")
        if selection.session != self._store.session:
            raise ValueError("ledger selection session does not match the reader session")
        with self._store._lock:
            segment_stop = len(self._store._forecast_scan_segments)
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
            with self._store._lock:
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
        segments = self._store._forecast_scan_segments
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
                entries.append((forecast_key, self._store._ledger._forecasts[forecast_key]))
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
        resolution = self._store._ledger._resolutions.get(forecast_key)
        if resolution is None:
            return None
        annotation = self._store._ledger._annotations.get(forecast_key)
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
                self._store._ledger._forecasts,
                forecast_key=forecast_key,
                resolutions=self._store._ledger._resolutions,
            )
            if any(
                issuance.bounds_finite and issuance.descriptor.window is EmissionScope.WINDOW_SUM
                for issuance in row.issuances.values()
            )
            else None
        )
        resolution = self._store._ledger._resolutions.get(forecast_key)
        row_actual = None if resolution is None else resolution.actual
        annotation = self._store._ledger._annotations.get(forecast_key)
        for bound_key, issuance in row.issuances.items():
            target = CoverageTarget(
                descriptor=issuance.descriptor,
                guaranteed_side=issuance.guaranteed_side,
                bound_key=bound_key,
            )
            actual_value = row_actual
            if issuance.bounds_finite and target.descriptor.window is EmissionScope.WINDOW_SUM:
                actual_value = window_sum_actual
            outcome = _score_bound(
                forecast_key=forecast_key,
                row=row,
                actual_value=actual_value,
                target=target,
                issuance=issuance,
                annotation=annotation,
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


def _forecast_segment_key(key: ForecastKey) -> tuple[int, bytes, bytes]:
    series_key, _origin, horizon_step, model_name = key
    return horizon_step, series_key.encode(), model_name.encode()


def _panel_actual_records(
    panel: Panel,
    *,
    actuals_semantics: ActualsSemantics,
) -> tuple[ActualRecord, ...]:
    """Index every non-missing panel observation as a canonical actual record."""
    if not isinstance(panel, Panel):
        raise TypeError("in-memory run-store actuals must be a Panel")
    if panel.has_censoring_facts and actuals_semantics is ActualsSemantics.DEMAND:
        raise ValueError("a panel with censoring facts cannot supply demand-honest actuals")
    records: list[ActualRecord] = []
    for values in panel.frame.to_dict("records"):
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
    return tuple(records)


def _actual_record_buckets(
    records: Iterable[ActualRecord],
) -> tuple[tuple[pd.Timestamp, tuple[ActualRecord, ...]], ...]:
    """Group source records into timestamp-ordered, key-canonical buckets."""
    by_timestamp: dict[pd.Timestamp, list[ActualRecord]] = {}
    for record in records:
        by_timestamp.setdefault(record.timestamp, []).append(record)
    return tuple(
        (
            timestamp,
            tuple(sorted(by_timestamp[timestamp], key=lambda value: value.series_key.encode())),
        )
        for timestamp in sorted(by_timestamp)
    )


def _pending_group(value: PendingObservation) -> _PendingGroup:
    """Return the bounded forecast lineage needed for window-sum readiness."""
    key = value.forecast_key
    return key.series_key, key.origin, key.model_name


def _checkpoint_key_from_index(encoded: bytes) -> str:
    """Validate one canonical checkpoint index and return its checkpoint key."""
    try:
        payload = json.loads(encoded)
        if not isinstance(payload, dict) or set(payload) != {
            "checkpoint_key",
            "cursor",
            "schema",
        }:
            raise ValueError("exact object fields")
        if canonical_json_bytes(payload, path="forecast checkpoint index") != encoded:
            raise ValueError("canonical encoding")
        if payload["schema"] != _CHECKPOINT_INDEX_SCHEMA:
            raise ValueError("supported schema")
        key = payload["checkpoint_key"]
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint key")
        cursor = payload["cursor"]
        if not isinstance(cursor, dict) or set(cursor) != {
            "panel_identity",
            "series_start",
            "series_stop",
            "time_bound",
        }:
            raise ValueError("cursor")
        HistoryCursor(
            cursor["panel_identity"],
            cursor["series_start"],
            cursor["series_stop"],
            cursor["time_bound"],
        )
    except (
        CanonicalJsonError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
    ) as error:
        raise LedgerError(f"checkpoint index is malformed: {error}") from error
    return key


def _stage_new_rows(
    write: OriginCommit | ActualsCommit,
    *,
    calendar: Calendar,
) -> Ledger:
    staged = Ledger(session=write.session, calendar=calendar)
    forecasts = () if isinstance(write, ActualsCommit) else write.forecasts
    orders = () if isinstance(write, ActualsCommit) else write.orders
    for forecast in forecasts:
        staged.append_forecasts(
            forecast.frame,
            issuances=forecast.issuances,
            observation_issuances=forecast.observation_issuances,
        )
    staged.append_orders(orders)
    staged.append_settlements(write.settlements)
    return staged


def _require_origin_rows(
    write: OriginCommit | ActualsCommit,
    *,
    staged: Ledger,
) -> None:
    if any(row.origin != write.origin for row in staged.forecasts):
        raise LedgerError("forecast row origin must match its origin commit")
    if any(row.origin != write.origin for row in staged.orders):
        raise LedgerError("order row origin must match its origin commit")


def _reject_collision(existing: Container[object], staged: Iterable[object], family: str) -> None:
    duplicate = next((key for key in staged if key in existing), None)
    if duplicate is not None:
        raise LedgerError(f"duplicate {family} key: {duplicate!r}")


__all__ = [
    "InMemoryIndexedRunStore",
    "InMemoryLedgerReader",
    "InMemoryPanelSource",
    "RunStoreAudit",
]
