"""Run the fixed chapter-03 engine spine over six abstract ports."""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import islice
from types import MappingProxyType
from typing import overload

import pandas as pd

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    CalibrationContext,
    ConformalRuntime,
    ConformalStateBatch,
    EmissionForm,
    IssuedBoundFacts,
    resolve_method,
)
from newcalibre.conformal import CalibrationResult as RuntimeCalibrationResult
from newcalibre.conformal import (
    ForecastKey as ConformalForecastKey,
)
from newcalibre.domain import (
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    ActualsSemantics,
    Calendar,
    CycleToken,
    DecisionTiming,
    ForecastTask,
    GuaranteeClaim,
    HierarchyIndex,
    InventoryPosition,
    Scope,
    SessionIdentity,
    StockoutRule,
    forecast_bound_groups,
    interval_columns,
    validate_forecast_frame,
)
from newcalibre.engine._ordering_runtime import (
    ConfiguredPolicyOrderer,
    DecisionBatch,
    DecisionEvidence,
    DecisionKey,
    OrderProposal,
)
from newcalibre.engine._ordering_runtime import (
    materialize_decisions as _materialize_decisions,
)
from newcalibre.engine._session import SessionCosts
from newcalibre.engine._session import (
    require_panel_session_binding as _require_panel_session_binding,
)
from newcalibre.engine._session import (
    require_task_session_binding as _require_task_session_binding,
)
from newcalibre.engine._session import session_conformal_config as _session_conformal_config
from newcalibre.engine._session import session_decision_inputs as _session_decision_inputs
from newcalibre.engine._session import session_model_config as _session_model_config
from newcalibre.engine._session import (
    session_ordering_configuration as _session_ordering_configuration,
)
from newcalibre.engine._session import (
    session_origin_inputs as _session_origin_inputs,
)
from newcalibre.engine.errors import EngineError as _EngineError
from newcalibre.engine.forecast_lifecycle import ForecastLifecycle, ForecastLifecycleResult
from newcalibre.engine.indexed_panel import IndexedPanel
from newcalibre.engine.ports import DispatchBackend, PanelSource
from newcalibre.engine.run_store import (
    ActualKey,
    ActualsCommit,
    ActualsSnapshot,
    CommitReceipt,
    ForecastWrite,
    IndexedRunStore,
    OriginCommit,
    OriginSnapshot,
    SettlementSnapshot,
)
from newcalibre.engine.settlement import SettlementRequest as _SettlementRequest
from newcalibre.engine.settlement import SettlementResult as _SettlementResult
from newcalibre.engine.settlement import settle as _settle
from newcalibre.forecasting import ForecastAdapter, resolve_adapter
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    GuaranteedSide,
    OrderRow,
)
from newcalibre.observe import ActualsSubmission, ObserveCycle, ObserveLoop
from newcalibre.ordering import OrderingInputError
from newcalibre.reconcile import ReconciliationContext, resolve_strategy

ENGINE_VERBS = (
    "fit",
    "predict",
    "reconcile",
    "calibrate",
    "order",
    "observe",
    "settle",
    "commit",
)


class Phase(StrEnum):
    """Name the six phases in their only legal per-origin order."""

    RESOLVE = "Resolve"
    PREDICT = "Predict"
    RECONCILE = "Reconcile"
    CALIBRATE = "Calibrate"
    ORDER = "Order"
    COMMIT = "Commit"


class PhaseError(RuntimeError):
    """Wrap a phase failure with the phase and origin that failed."""

    def __init__(self, phase: Phase, origin: pd.Timestamp, cause: Exception) -> None:
        super().__init__(f"{phase.value} failed at origin {origin}: {cause}")
        self.phase = phase
        self.origin = origin
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class PhaseEvent:
    """Record one success or failure timing from the public spine."""

    phase: Phase
    origin: pd.Timestamp
    duration_seconds: float
    error: str | None


@dataclass(frozen=True, slots=True, init=False)
class ForecastBatch:
    """Carry a defensive forecast frame and its per-row issuance facts."""

    _frame: pd.DataFrame = field(repr=False)
    _calendar: Calendar = field(repr=False)
    _token: CycleToken | None = field(repr=False)
    issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]]
    observation_issuances: Mapping[ForecastKey, IssuedBoundFacts]

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        calendar: Calendar,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]] | None = None,
        observation_issuances: Mapping[ForecastKey, IssuedBoundFacts] | None = None,
    ) -> None:
        owned_frame = pd.DataFrame(frame, copy=True)
        validated = validate_forecast_frame(owned_frame, calendar=calendar)
        keys = _forecast_keys(validated)
        bound_groups = forecast_bound_groups(validated.columns)
        interval_groups = tuple(group for group in bound_groups if len(group) == 2)
        if issuances is None:
            if interval_groups:
                raise _EngineError("forecast interval columns require explicit issuance metadata")
            supplied: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]] = {
                key: {} for key in keys
            }
        else:
            supplied = issuances
        if set(supplied) != set(keys):
            raise _EngineError("forecast issuance keys must exactly match the frame keys")
        observed = {} if observation_issuances is None else dict(observation_issuances)
        unknown_observed = set(observed) - set(keys)
        if unknown_observed:
            raise _EngineError("observation issuance names an unknown forecast row key")
        frozen_observed = {key: IssuedBoundFacts.snapshot(value) for key, value in observed.items()}
        frozen: dict[ForecastKey, Mapping[BoundKey, ForecastIssuance]] = {}
        for key in keys:
            row_issuances = dict(supplied[key])
            intervals_issued = all(
                group in row_issuances or any((column,) in row_issuances for column in group)
                for group in interval_groups
            )
            if not intervals_issued:
                raise _EngineError("forecast interval columns require explicit issuance metadata")
            frozen[key] = MappingProxyType(row_issuances)
        for quantile_group in (group for group in bound_groups if len(group) == 1):
            issued = [quantile_group in row_issuances for row_issuances in frozen.values()]
            if any(issued) and not all(issued):
                raise _EngineError(
                    "forecast quantile issuance metadata must cover every row or no rows"
                )
        object.__setattr__(self, "_frame", validated)
        object.__setattr__(self, "_calendar", calendar)
        object.__setattr__(self, "_token", None)
        object.__setattr__(self, "issuances", MappingProxyType(frozen))
        object.__setattr__(
            self,
            "observation_issuances",
            MappingProxyType(frozen_observed),
        )

    @property
    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the validated forecast frame."""
        return self._frame.copy(deep=True)

    @property
    def calendar(self) -> Calendar:
        """Return the bound calendar used to validate this batch."""
        return self._calendar

    @property
    def session(self) -> SessionIdentity | None:
        """Return immutable engine provenance, if this batch has been bound."""
        return None if self._token is None else self._token.session

    @property
    def token(self) -> CycleToken | None:
        """Return immutable cycle provenance, if this batch has been bound."""
        return self._token


@dataclass(frozen=True, slots=True)
class FittedTask:
    """Address fitted state without carrying model objects, bytes, or locations."""

    token: CycleToken
    task: ForecastTask

    def __post_init__(self) -> None:
        if not isinstance(self.token, CycleToken):
            raise TypeError("fitted task token must be a CycleToken")
        if not isinstance(self.task, ForecastTask):
            raise TypeError("fitted task task must be a ForecastTask")
        if self.task.origin != self.token.origin:
            raise _EngineError("fitted task origin must match its cycle token")
        _require_task_session_binding(self.task, session=self.token.session)

    @property
    def session(self) -> SessionIdentity:
        """Return the cycle-bound session."""
        return self.token.session


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Authenticate one immutable staged observe cycle and its prior state."""

    cycle: ObserveCycle
    prior_states: Mapping[str, bytes] = field(default_factory=dict)
    token: CycleToken | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, ObserveCycle):
            raise TypeError("observation result cycle must be an ObserveCycle")
        prior_states = _validated_state_snapshot(self.prior_states)
        object.__setattr__(self, "prior_states", MappingProxyType(prior_states))
        if self.token is not None and not isinstance(self.token, CycleToken):
            raise TypeError("observation result token must be a CycleToken")

    @property
    def session(self) -> SessionIdentity | None:
        """Return the cycle-bound session, if present."""
        return None if self.token is None else self.token.session

    @property
    def origin(self) -> pd.Timestamp | None:
        """Return the cycle-bound origin, if present."""
        return None if self.token is None else self.token.origin


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Return a calibrated frame plus state mutations staged for Commit."""

    forecasts: ForecastBatch
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    token: CycleToken | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.forecasts, ForecastBatch):
            raise TypeError("calibration result forecasts must be a ForecastBatch")
        updates = _validated_state_updates(self.state_updates)
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        if self.token is not None and not isinstance(self.token, CycleToken):
            raise TypeError("calibration result token must be a CycleToken")
        if self.session is not None:
            if self.forecasts.session != self.session:
                raise _EngineError("calibration result forecasts must match its session")
            assert self.origin is not None
            if any(key[1] != self.origin for key in self.forecasts.issuances):
                raise _EngineError("calibration result forecasts must match its origin")

    @property
    def session(self) -> SessionIdentity | None:
        """Return the cycle-bound session, if present."""
        return None if self.token is None else self.token.session

    @property
    def origin(self) -> pd.Timestamp | None:
        """Return the cycle-bound origin, if present."""
        return None if self.token is None else self.token.origin


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Give an ordering policy every declared decision-time fact."""

    session: SessionIdentity
    origin: pd.Timestamp
    forecasts: ForecastBatch
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    costs_by_series: SessionCosts | None = field(init=False)
    timing: DecisionTiming | None = field(init=False)
    stockout_rule: StockoutRule | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("order request session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("order request origin must be a pandas Timestamp")
        if not isinstance(self.forecasts, ForecastBatch):
            raise TypeError("order request forecasts must be a ForecastBatch")
        if any(key[1] != self.origin for key in self.forecasts.issuances):
            raise _EngineError("order request forecasts must all match its origin")
        if self.forecasts.session != self.session:
            raise _EngineError("order request forecasts must match its session")
        positions = _validated_inventory_positions(self.inventory_positions)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        decision = _session_decision_inputs(self.session)
        if decision is None:
            costs_by_series = timing = stockout_rule = None
        else:
            costs_by_series = decision.costs_by_series
            timing = decision.timing
            stockout_rule = decision.stockout_rule
        object.__setattr__(self, "costs_by_series", costs_by_series)
        object.__setattr__(self, "timing", timing)
        object.__setattr__(self, "stockout_rule", stockout_rule)


@dataclass(frozen=True, slots=True, init=False)
class OriginRequest:
    """Declare every input needed to compute one origin."""

    session: SessionIdentity
    origin: pd.Timestamp
    horizon: int
    scope: Scope
    _model_config: bytes = field(repr=False)
    _future_exogenous: pd.DataFrame | None = field(repr=False)
    _inventory_positions: Mapping[str, InventoryPosition] = field(repr=False)

    def __init__(
        self,
        *,
        session: SessionIdentity,
        origin: pd.Timestamp,
        scope: Scope,
        future_exogenous: pd.DataFrame | None = None,
        inventory_positions: Mapping[str, InventoryPosition] | None = None,
    ) -> None:
        if not isinstance(session, SessionIdentity):
            raise TypeError("origin request session must be a SessionIdentity")
        if not isinstance(origin, pd.Timestamp):
            raise TypeError("origin request origin must be a pandas Timestamp")
        if not isinstance(scope, Scope):
            raise TypeError("origin request scope must be a Scope")
        horizon, encoded_config = _session_origin_inputs(session)
        if future_exogenous is not None and not isinstance(future_exogenous, pd.DataFrame):
            raise TypeError("future exogenous input must be a pandas DataFrame")
        positions = _validated_inventory_positions(inventory_positions or {})
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "_model_config", encoded_config)
        object.__setattr__(
            self,
            "_future_exogenous",
            None if future_exogenous is None else future_exogenous.copy(deep=True),
        )
        object.__setattr__(self, "_inventory_positions", MappingProxyType(positions))

    @property
    def model_config(self) -> Mapping[str, object]:
        """Return a fresh materialization of the canonical nested configuration."""
        return json.loads(self._model_config)

    @property
    def future_exogenous(self) -> pd.DataFrame | None:
        """Return an isolated copy of the future-known regressors."""
        if self._future_exogenous is None:
            return None
        return self._future_exogenous.copy(deep=True)

    @property
    def inventory_positions(self) -> Mapping[str, InventoryPosition]:
        """Return the immutable per-series inventory facts for ordering."""
        return self._inventory_positions


@dataclass(frozen=True, slots=True)
class OriginResult:
    """Return the committed forecast, order, and journal facts for one origin."""

    forecasts: ForecastBatch
    orders: tuple[OrderRow, ...]
    receipt: CommitReceipt


@dataclass(frozen=True, slots=True)
class SettlementWindow:
    """Carry authoritative actuals and the compact ledger projection for one cycle."""

    snapshot: SettlementSnapshot
    actuals: Mapping[ActualKey, float]
    actuals_semantics: ActualsSemantics
    token: CycleToken | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SettlementSnapshot):
            raise TypeError("settlement window snapshot must be a SettlementSnapshot")
        if not isinstance(self.actuals, Mapping):
            raise TypeError("settlement window actuals must be a mapping")
        if not isinstance(self.actuals_semantics, ActualsSemantics):
            raise TypeError("settlement window semantics must be ActualsSemantics")
        if self.token is not None and not isinstance(self.token, CycleToken):
            raise TypeError("settlement window token must be a CycleToken or None")
        if self.token is not None and self.token.session != self.snapshot.session:
            raise _EngineError("settlement window token must match its session")
        object.__setattr__(self, "actuals", MappingProxyType(dict(self.actuals)))


@dataclass(frozen=True, slots=True)
class CommitRequest:
    """Carry typed phase results into the one materializing Commit boundary."""

    session: SessionIdentity
    origin: pd.Timestamp
    token: CycleToken
    observation: ObservationResult
    calibration: CalibrationResult
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    decisions: DecisionBatch | None = None
    settlement: SettlementWindow | None = None
    input_fingerprint: str | None = None
    expected_forecast_origin_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("commit request session must be a SessionIdentity")
        _require_timestamp(self.origin, name="commit request origin")
        if not isinstance(self.token, CycleToken):
            raise TypeError("commit request token must be a CycleToken")
        if self.token.session != self.session or self.token.origin != self.origin:
            raise _EngineError("commit request token must match its session and origin")
        if not isinstance(self.observation, ObservationResult):
            raise TypeError("commit request observation must be an ObservationResult")
        if not isinstance(self.calibration, CalibrationResult):
            raise TypeError("commit request calibration must be a CalibrationResult")
        if self.observation.session != self.session or self.observation.origin != self.origin:
            raise _EngineError("commit request observation must match its session and origin")
        if self.observation.token != self.token:
            raise _EngineError("commit request observation must match its cycle token")
        if self.calibration.session != self.session or self.calibration.origin != self.origin:
            raise _EngineError("commit request calibration must match its session and origin")
        if self.calibration.token != self.token:
            raise _EngineError("commit request calibration must match its cycle token")
        if any(key[1] != self.origin for key in self.calibration.forecasts.issuances):
            raise _EngineError("commit request forecasts must all match its origin")
        if self.decisions is not None:
            if not isinstance(self.decisions, DecisionBatch):
                raise TypeError("commit request decisions must be a DecisionBatch or None")
            if self.decisions.session != self.session or self.decisions.origin != self.origin:
                raise _EngineError("commit request decisions must match its session and origin")
            if self.decisions.token != self.token:
                raise _EngineError("commit request decisions must match its cycle token")
        if self.settlement is not None:
            if not isinstance(self.settlement, SettlementWindow):
                raise TypeError("commit request settlement must be a SettlementWindow or None")
            if self.settlement.snapshot.session != self.session:
                raise _EngineError("commit request settlement must match its session")
            if self.settlement.snapshot.periods != (self.origin,):
                raise _EngineError("commit request settlement must contain exactly its origin")
            if self.settlement.token != self.token:
                raise _EngineError("commit request settlement must match its cycle token")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str)
            or len(self.input_fingerprint) != 64
            or self.input_fingerprint != self.input_fingerprint.lower()
            or any(character not in "0123456789abcdef" for character in self.input_fingerprint)
        ):
            raise ValueError("commit request input fingerprint must be a SHA-256 hex string")
        expected_count = self.expected_forecast_origin_count
        if expected_count is not None and (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count < 0
        ):
            raise ValueError("expected forecast origin count must be a non-negative integer")
        positions = _validated_inventory_positions(self.inventory_positions)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Return the durable receipt and the exact rows materialized by Commit."""

    token: CycleToken
    receipt: CommitReceipt
    orders: tuple[OrderRow, ...]
    settlement: _SettlementResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token, CycleToken):
            raise TypeError("commit result token must be a CycleToken")
        if not isinstance(self.receipt, CommitReceipt):
            raise TypeError("commit result receipt must be a CommitReceipt")
        if self.receipt.session != self.token.session or self.receipt.origin != self.token.origin:
            raise _EngineError("commit result receipt must match its cycle token")
        orders = tuple(self.orders)
        if any(not isinstance(order, OrderRow) for order in orders):
            raise TypeError("commit result orders must contain OrderRow values")
        if self.settlement is not None and not isinstance(self.settlement, _SettlementResult):
            raise TypeError("commit result settlement must be a SettlementResult or None")
        if self.settlement is not None and self.settlement.token != self.token:
            raise _EngineError("commit result settlement must match its cycle token")
        object.__setattr__(self, "orders", orders)


type AdapterResolver = Callable[[Mapping[str, object]], ForecastAdapter]
type CustomOrderer = Callable[[OrderRequest], Iterable[OrderProposal]]
type Orderer = CustomOrderer | ConfiguredPolicyOrderer
type PhaseReporter = Callable[[PhaseEvent], None]


class Engine:
    """Own the exact eight-verb engine surface over three ports."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        panel_source: PanelSource,
        run_store: IndexedRunStore,
        dispatch_backend: DispatchBackend,
        hierarchy: HierarchyIndex | None,
        adapter_resolver: AdapterResolver = resolve_adapter,
        reconciliation_strategy: str = "none",
        orderer: Orderer | None = None,
    ) -> None:
        ports = (
            (panel_source, PanelSource, "panel source"),
            (run_store, IndexedRunStore, "run store"),
            (dispatch_backend, DispatchBackend, "dispatch backend"),
        )
        for adapter, port, name in ports:
            if not isinstance(adapter, port):
                raise TypeError(f"engine {name} does not satisfy its port")
        if hierarchy is not None and not isinstance(hierarchy, HierarchyIndex):
            raise TypeError("engine hierarchy must be a HierarchyIndex or None")
        if not callable(adapter_resolver):
            raise TypeError("engine adapter resolver must be callable")
        if orderer is not None and not (
            callable(orderer) or isinstance(orderer, ConfiguredPolicyOrderer)
        ):
            raise TypeError("engine orderer must be callable or a ConfiguredPolicyOrderer")
        if not isinstance(session, SessionIdentity):
            raise TypeError("engine session must be a SessionIdentity")
        # Resolve configuration before the panel port can perform any data load.
        adapter_resolver(_session_model_config(session))
        reconciler = resolve_strategy(reconciliation_strategy)
        ordering_configuration = _session_ordering_configuration(session)
        conformal_configuration = _session_conformal_config(session)
        runtime = (
            None
            if conformal_configuration is None
            else resolve_method(conformal_configuration, states={})
        )
        self._ordering_configuration = ordering_configuration
        self._runtime: ConformalRuntime | None = runtime
        self._reconciliation_hierarchy = hierarchy
        loaded_panel = panel_source.load()
        _require_panel_session_binding(
            loaded_panel,
            session=session,
            ledger_calendar=loaded_panel.calendar,
        )
        self._session = session
        self._calendar = loaded_panel.calendar
        self._panel = IndexedPanel.from_panel(loaded_panel)
        self._hierarchy = hierarchy or HierarchyIndex.flat(self._panel.series_keys)
        self._run_store = run_store
        self._dispatch_backend = dispatch_backend
        self._forecast_lifecycle = ForecastLifecycle(
            adapter_resolver=adapter_resolver,
        )
        self._reconciler = reconciler
        self._orderer = orderer
        self._active_cycles: dict[tuple[SessionIdentity, pd.Timestamp], CycleToken] = {}
        self._snapshots: dict[CycleToken, OriginSnapshot | ActualsSnapshot] = {}
        self._forecast_results: dict[tuple[CycleToken, str], ForecastLifecycleResult] = {}
        self._checkpoint_effects: dict[
            CycleToken,
            tuple[Mapping[str, bytes], Mapping[str, bytes]],
        ] = {}
        self._issued_decisions: dict[CycleToken, DecisionBatch] = {}
        hierarchy_node_series = tuple(sorted(self._hierarchy.node_labels, key=str.encode))
        if self._panel.series_keys not in (
            self._hierarchy.bottom_series,
            hierarchy_node_series,
        ):
            raise _EngineError(
                "engine panel series must exactly match the hierarchy bottoms or all nodes"
            )

    def fit(self, request: OriginRequest) -> tuple[FittedTask, ...]:
        """Build deterministic tasks and fit or restore their adapters."""
        if not isinstance(request, OriginRequest):
            raise TypeError("fit requires an OriginRequest")
        self._require_session(request.session)
        token = self._cycle_for(session=request.session, origin=request.origin)
        snapshot = self._origin_snapshot(token)
        try:
            provisional = self._panel.tasks(
                origin=request.origin,
                horizon=request.horizon,
                scope=request.scope,
                model_config=request.model_config,
                future_exogenous=request.future_exogenous,
            )
            previous_cursors = self._forecast_lifecycle.previous_cursors(
                session=request.session,
                tasks=provisional,
                checkpoint_indexes=snapshot.checkpoint_indexes,
            )
            tasks = (
                provisional
                if not previous_cursors
                else self._panel.tasks(
                    origin=request.origin,
                    horizon=request.horizon,
                    scope=request.scope,
                    model_config=request.model_config,
                    future_exogenous=request.future_exogenous,
                    previous_cursors=previous_cursors,
                )
            )
            items = tuple(
                (
                    request.session,
                    task,
                    token,
                    snapshot.checkpoints,
                    snapshot.checkpoint_indexes,
                )
                for task in tasks
            )
            results = self._dispatch_backend.map(self._forecast_lifecycle.run_item, items)
            staged = {
                (token, task.identity): result for task, result in zip(tasks, results, strict=True)
            }
        except Exception:
            self._retire_cycle(token)
            raise
        self._forecast_results.update(staged)
        return tuple(FittedTask(token=token, task=task) for task in tasks)

    def predict(self, fitted_tasks: Sequence[FittedTask]) -> ForecastBatch:
        """Predict fitted tasks in deterministic dispatch order."""
        tasks = tuple(fitted_tasks)
        if not tasks:
            raise _EngineError("predict requires at least one fitted task")
        for fitted in tasks:
            if not isinstance(fitted, FittedTask):
                raise TypeError("predict requires FittedTask values")
            self._require_session(fitted.session)
            self._require_cycle(fitted.token)
        token = tasks[0].token
        if any(fitted.token != token for fitted in tasks):
            raise _EngineError("predict fitted tasks must share one cycle token")
        result_keys = tuple((token, fitted.task.identity) for fitted in tasks)
        if len(set(result_keys)) != len(result_keys) or any(
            key not in self._forecast_results for key in result_keys
        ):
            raise _EngineError("predict requires engine-issued fitted tasks exactly once")
        results = tuple(self._forecast_results[key] for key in result_keys)
        try:
            combined = pd.concat((result.frame for result in results), ignore_index=True)
            batch = _bind_forecast_batch(
                ForecastBatch(
                    combined,
                    calendar=self._calendar,
                ),
                token=token,
            )
            self._checkpoint_effects[token] = self._forecast_lifecycle.staged_updates(results)
        except Exception:
            self._retire_cycle(token)
            raise
        for key in result_keys:
            del self._forecast_results[key]
        return batch

    def reconcile(self, forecasts: ForecastBatch) -> ForecastBatch:
        """Apply the configured registered point-reconciliation strategy."""
        if not isinstance(forecasts, ForecastBatch):
            raise TypeError("reconcile requires a ForecastBatch")
        self._require_forecast_batch(forecasts)
        reconciliation_input = forecasts.frame
        preserved_distributional: pd.DataFrame | None = None
        if self._reconciliation_hierarchy is None:
            bound_columns = tuple(
                column
                for group in forecast_bound_groups(reconciliation_input.columns)
                for column in group
            )
            if bound_columns:
                # The reconciliation seam owns points only; no-hierarchy runs
                # retain native distributional outputs outside that seam.
                preserved_distributional = reconciliation_input
                reconciliation_input = reconciliation_input.drop(columns=list(bound_columns))
        result = self._reconciler(
            reconciliation_input,
            self._reconciliation_hierarchy,
            ReconciliationContext(target_support=self._panel.target_support),
        )
        if not isinstance(result, pd.DataFrame):
            raise _EngineError("reconciliation strategy must return a pandas DataFrame")
        if preserved_distributional is not None:
            point_result = validate_forecast_frame(
                pd.DataFrame(result, copy=True),
                calendar=forecasts.calendar,
            )
            if _forecast_keys(point_result) != _forecast_keys(reconciliation_input):
                raise _EngineError(
                    "no-hierarchy reconciliation changed forecast rows before bound restoration"
                )
            restored = preserved_distributional.copy(deep=True)
            restored[POINT_FORECAST] = point_result[POINT_FORECAST].to_numpy(copy=True)
            result = restored
        normalized = validate_forecast_frame(
            pd.DataFrame(result, copy=True),
            calendar=forecasts.calendar,
        )
        result_keys = _forecast_keys(normalized)
        original_keys = set(forecasts.issuances)
        removed = original_keys - set(result_keys)
        if removed:
            raise _EngineError("reconciliation removed or changed forecast row keys")
        origin = _forecast_batch_origin(forecasts)
        if any(key[1] != origin for key in result_keys):
            raise _EngineError("reconciliation changed the forecast origin")
        issuances = {
            key: dict(forecasts.issuances[key]) if key in original_keys else {}
            for key in result_keys
        }
        observation_issuances = {
            key: value
            for key, value in forecasts.observation_issuances.items()
            if key in original_keys
        }
        reconciled = ForecastBatch(
            normalized,
            calendar=forecasts.calendar,
            issuances=issuances,
            observation_issuances=observation_issuances,
        )
        assert forecasts.session is not None
        assert forecasts.token is not None
        return _bind_forecast_batch(
            reconciled,
            token=forecasts.token,
        )

    def calibrate(
        self,
        forecasts: ForecastBatch,
        *,
        session: SessionIdentity,
        observation: ObservationResult,
    ) -> CalibrationResult:
        """Apply once from the observe-updated origin state snapshot."""
        if not isinstance(forecasts, ForecastBatch):
            raise TypeError("calibrate requires a ForecastBatch")
        if not isinstance(observation, ObservationResult):
            raise TypeError("calibrate requires an ObservationResult")
        self._require_session(session)
        self._require_forecast_batch(forecasts)
        if forecasts.session != session:
            raise _EngineError("calibration forecasts do not match its session")
        origin = _forecast_batch_origin(forecasts)
        if observation.session != session or observation.origin != origin:
            raise _EngineError("calibration observation does not match its session and origin")
        if observation.token != forecasts.token:
            raise _EngineError("calibration inputs must share one cycle token")
        assert forecasts.token is not None
        self._require_cycle(forecasts.token)

        observed_updates = dict(observation.cycle.state_updates)
        prior_state = ConformalStateBatch(observation.prior_states)
        observed_state = prior_state.with_rows(observed_updates)
        if self._runtime is None:
            return _bind_calibration_result(
                forecasts,
                observed_updates,
                token=forecasts.token,
            )

        context = _calibration_context(
            self._runtime,
            forecasts._frame,
            hierarchy=self._hierarchy,
        )
        try:
            runtime_result = self._runtime.apply(
                forecasts.frame,
                observed_state,
                context=context,
            )
        except ValueError as error:
            raise _EngineError(f"conformal apply failed: {error}") from error
        if not isinstance(runtime_result, RuntimeCalibrationResult):
            raise _EngineError("conformal apply must return a CalibrationResult")
        observation_issuances = _ledger_observation_issuances(runtime_result.issuances)
        calibrated_issuances = _ledger_bound_issuances(observation_issuances)
        ledger_issuances = {
            key: dict(forecasts.issuances[key]) | calibrated_issuances.get(key, {})
            for key in forecasts.issuances
        }
        calibrated_value = ForecastBatch(
            runtime_result.forecasts,
            calendar=forecasts.calendar,
            issuances=ledger_issuances,
            observation_issuances=observation_issuances,
        )
        if set(calibrated_value.issuances) != set(forecasts.issuances):
            raise _EngineError("conformal apply changed forecast row keys")
        calibrated = _bind_forecast_batch(
            calibrated_value,
            token=forecasts.token,
        )
        if _forecast_batch_origin(calibrated) != origin:
            raise _EngineError("conformal apply changed the forecast origin")

        foreign_dirty = set(runtime_result.dirty_labels).difference({METHOD_SCOPE_LABEL})
        if foreign_dirty:
            raise _EngineError(
                f"conformal apply dirtied foreign partition state: {sorted(foreign_dirty)!r}"
            )
        final_state = runtime_result.state
        dirty_candidates = {
            *observation.cycle.state_updates,
            *runtime_result.dirty_labels,
        }
        final_dirty = tuple(
            label
            for label in dirty_candidates
            if label not in prior_state or prior_state[label] != final_state[label]
        )
        merged_updates = dict(final_state.project(final_dirty))
        return _bind_calibration_result(
            calibrated,
            merged_updates,
            token=forecasts.token,
        )

    def order(self, request: OrderRequest) -> DecisionBatch | None:
        """Apply the configured orderer, or report that decisions are disabled."""
        if not isinstance(request, OrderRequest):
            raise TypeError("order requires an OrderRequest")
        self._require_session(request.session)
        if self._orderer is None:
            return None
        self._require_forecast_batch(request.forecasts)
        configuration = self._ordering_configuration
        if configuration is None:
            raise _EngineError("order requires session ordering configuration")
        decision_request = _bottom_node_order_request(
            request,
            series_keys=configuration.series_keys,
            token=request.forecasts.token,
        )
        task_horizon, _model_config = _session_origin_inputs(request.session)
        _require_decision_series_coverage(
            decision_request.forecasts,
            series_keys=configuration.series_keys,
            task_horizon=task_horizon,
        )
        requested = _decision_keys(decision_request.forecasts)
        _require_inventory_coverage(decision_request.inventory_positions, requested=requested)
        if isinstance(self._orderer, ConfiguredPolicyOrderer):
            produced = self._orderer.propose(
                frame=decision_request.forecasts._frame,
                issuances=decision_request.forecasts.issuances,
                inventory_positions=decision_request.inventory_positions,
                configuration=configuration,
            )
        else:
            produced = self._orderer(decision_request)
        proposals = tuple(islice(produced, len(requested) + 1))
        decisions = DecisionBatch(
            token=self._required_batch_token(request.forecasts),
            requested=requested,
            proposals=proposals,
        )
        self._issued_decisions[decisions.token] = decisions
        return decisions

    def observe(
        self,
        origin: pd.Timestamp,
        *,
        session: SessionIdentity,
        snapshot: OriginSnapshot | ActualsSnapshot | None = None,
        submission: ActualsSubmission | None = None,
    ) -> ObservationResult:
        """Stage one observe cycle from the active revision snapshot."""
        self._require_session(session)
        if snapshot is not None:
            if snapshot.session != session or snapshot.origin != origin:
                raise _EngineError("observe snapshot must match its session and origin")
            self._begin_cycle(snapshot)
        token = self._cycle_for(session=session, origin=origin)
        snapshot = self._snapshots[token]
        if submission is not None and not isinstance(submission, ActualsSubmission):
            raise TypeError("observe submission must be an ActualsSubmission or None")
        prior_states = dict(snapshot.conformal_states)
        loop = ObserveLoop(
            hierarchy=self._hierarchy,
            observed_history=snapshot.observed_history,
            pending_observations=snapshot.pending_observations,
            conformal_states=prior_states,
            runtime=self._runtime,
        )
        accepted = snapshot.actuals if submission is None else submission
        loop.accept(accepted)
        cycle = loop.cycle(origin)
        return _bind_observation_result(
            cycle,
            prior_states,
            token=token,
        )

    def settle(self, request: _SettlementRequest) -> _SettlementResult:
        """Apply the engine's single pure settlement implementation."""
        if not isinstance(request, _SettlementRequest):
            raise TypeError("settle requires a SettlementRequest")
        self._require_session(request.session)
        if request.token is None:
            raise _EngineError("engine settlement requires a cycle token")
        self._require_cycle(request.token)
        if request.snapshot.calendar != self._calendar:
            raise _EngineError("settlement snapshot calendar does not match the engine store")
        active = self._snapshots[request.token]
        if active.settlement != request.snapshot:
            raise _EngineError("settlement snapshot does not match the active store snapshot")
        return _settle(request)

    @overload
    def commit(self, request: CommitRequest) -> CommitResult: ...

    @overload
    def commit(self, request: OriginCommit | ActualsCommit) -> CommitReceipt: ...

    def commit(
        self,
        request: CommitRequest | OriginCommit | ActualsCommit,
    ) -> CommitResult | CommitReceipt:
        """Persist an origin's ledger and monotone calibration-state mutations."""
        if isinstance(request, CommitRequest):
            try:
                return self._commit_request(request)
            except Exception:
                self._retire_cycle(request.token)
                raise
        if not isinstance(request, (OriginCommit, ActualsCommit)):
            raise TypeError("commit requires a CommitRequest, OriginCommit, or ActualsCommit")
        self._require_session(request.session)
        receipt = _snapshot_commit_receipt(self._run_store.commit(request))
        expected = CommitReceipt.from_commit(
            request,
            revision=request.expected_revision + 1,
        )
        if receipt != expected:
            raise _EngineError("run store returned a mismatched commit receipt")
        return receipt

    def _commit_request(self, request: CommitRequest) -> CommitResult:
        self._require_session(request.session)
        self._require_cycle(request.token)
        self._require_forecast_batch(request.calibration.forecasts)
        configuration = self._ordering_configuration
        if request.decisions is not None:
            if configuration is None:
                raise _EngineError("commit decisions require ordering configuration")
            decision_request = _bottom_node_order_request(
                OrderRequest(
                    session=request.session,
                    origin=request.origin,
                    forecasts=request.calibration.forecasts,
                    inventory_positions=request.inventory_positions,
                ),
                series_keys=configuration.series_keys,
                token=request.token,
            )
            task_horizon, _model_config = _session_origin_inputs(request.session)
            _require_decision_series_coverage(
                decision_request.forecasts,
                series_keys=configuration.series_keys,
                task_horizon=task_horizon,
            )
            expected = _decision_keys(decision_request.forecasts)
            _require_inventory_coverage(decision_request.inventory_positions, requested=expected)
            if request.decisions.requested != expected:
                raise _EngineError("commit decisions do not match calibrated decision groups")
            if self._issued_decisions.get(request.token) is not request.decisions:
                raise _EngineError("commit decisions were not issued by this engine")
        orders = _materialize_decisions(
            request.decisions,
            configuration=configuration,
            calendar=self._calendar,
        )
        settlement_result: _SettlementResult | None = None
        if request.settlement is not None:
            settlement_request = _SettlementRequest(
                session=request.session,
                snapshot=request.settlement.snapshot,
                actuals=request.settlement.actuals,
                inventory_positions=request.inventory_positions,
                orders=orders,
                actuals_semantics=request.settlement.actuals_semantics,
                token=request.token,
            )
            settlement_result = self.settle(settlement_request)
        checkpoint_updates, checkpoint_indexes = self._checkpoint_effects.get(
            request.token,
            ({}, {}),
        )
        receipt = self.commit(
            OriginCommit(
                session=request.session,
                origin=request.origin,
                expected_revision=request.token.revision,
                observe_cycle=request.observation.cycle,
                forecasts=(
                    ForecastWrite(
                        request.calibration.forecasts.frame,
                        request.calibration.forecasts.issuances,
                        request.calibration.forecasts.observation_issuances,
                    ),
                ),
                orders=orders,
                settlements=() if settlement_result is None else settlement_result.records,
                state_updates=request.calibration.state_updates,
                checkpoint_updates=checkpoint_updates,
                checkpoint_indexes=checkpoint_indexes,
                input_fingerprint=request.input_fingerprint,
                expected_forecast_origin_count=request.expected_forecast_origin_count,
                inventory_positions=request.inventory_positions,
                resume_marker=request.origin,
            )
        )
        result = CommitResult(
            token=request.token,
            receipt=receipt,
            orders=orders,
            settlement=settlement_result,
        )
        self._retire_cycle(request.token)
        return result

    def _require_session(self, session: SessionIdentity) -> None:
        if session != self._session:
            raise _EngineError("engine session does not match its run store")

    def _require_forecast_batch(self, forecasts: ForecastBatch) -> None:
        if forecasts.token is None or forecasts.session is None:
            raise _EngineError("forecast batch was not produced by this engine")
        self._require_cycle(forecasts.token)
        self._require_session(forecasts.session)
        if forecasts.calendar != self._calendar:
            raise _EngineError("forecast batch calendar does not match the engine store")

    def _required_batch_token(self, forecasts: ForecastBatch) -> CycleToken:
        self._require_forecast_batch(forecasts)
        assert forecasts.token is not None
        return forecasts.token

    def _cycle_for(self, *, session: SessionIdentity, origin: pd.Timestamp) -> CycleToken:
        key = (session, origin)
        active = self._active_cycles.get(key)
        if active is not None:
            return active
        raise _EngineError("state-bearing work requires an opened run-store snapshot")

    def _begin_cycle(
        self,
        snapshot: OriginSnapshot | ActualsSnapshot,
    ) -> CycleToken:
        self._require_session(snapshot.session)
        session = snapshot.session
        origin = snapshot.origin
        active = self._active_cycles.get((session, origin))
        if active is not None:
            self._retire_cycle(active)
        token = CycleToken(session, origin, snapshot.revision)
        self._active_cycles[(session, origin)] = token
        self._snapshots[token] = snapshot
        return token

    def _require_cycle(self, token: CycleToken) -> None:
        if not isinstance(token, CycleToken):
            raise TypeError("engine cycle token must be a CycleToken")
        if token != self._active_cycles.get((token.session, token.origin)):
            raise _EngineError("value belongs to a stale or foreign engine cycle")

    def _retire_cycle(self, token: CycleToken) -> None:
        """Invalidate every controller-owned value staged for one cycle."""
        key = (token.session, token.origin)
        if self._active_cycles.get(key) == token:
            del self._active_cycles[key]
        self._snapshots.pop(token, None)
        self._checkpoint_effects.pop(token, None)
        self._issued_decisions.pop(token, None)
        stale_results = [key for key in self._forecast_results if key[0] == token]
        for result_key in stale_results:
            del self._forecast_results[result_key]

    def _abort_origin(self, origin: pd.Timestamp) -> None:
        """Invalidate the active cycle at a failed spine phase boundary."""
        token = self._active_cycles.get((self._session, origin))
        if token is not None:
            self._retire_cycle(token)

    def _origin_snapshot(self, token: CycleToken) -> OriginSnapshot:
        """Return the active origin snapshot for a forecast lifecycle."""
        snapshot = self._snapshots[token]
        if not isinstance(snapshot, OriginSnapshot):
            raise _EngineError("forecast lifecycle requires an origin snapshot")
        return snapshot

    def _require_driver_store(self, run_store: IndexedRunStore) -> None:
        """Reject a driver wired to a different store than this engine."""
        if run_store is not self._run_store:
            raise _EngineError("driver run store does not belong to the engine")


class Spine:
    """Compose the exact six phases without adding another engine verb."""

    def __init__(self, engine: Engine, *, reporter: PhaseReporter | None = None) -> None:
        if not isinstance(engine, Engine):
            raise TypeError("spine requires an Engine")
        self._engine = engine
        self._reporter = reporter

    def run_origin(
        self,
        request: OriginRequest,
        *,
        snapshot: OriginSnapshot,
        decision_origin: bool = True,
        settlement: SettlementWindow | None = None,
        input_fingerprint: str | None = None,
        expected_forecast_origin_count: int | None = None,
    ) -> OriginResult:
        """Run Resolve through Commit once for a declared origin."""
        if not isinstance(request, OriginRequest):
            raise TypeError("spine requires an OriginRequest")
        if not isinstance(snapshot, OriginSnapshot):
            raise TypeError("spine requires an OriginSnapshot")
        if snapshot.session != request.session or snapshot.origin != request.origin:
            raise _EngineError("spine snapshot must match its request session and origin")
        if not isinstance(decision_origin, bool):
            raise TypeError("decision_origin must be boolean")
        if settlement is not None and not isinstance(settlement, SettlementWindow):
            raise TypeError("settlement must be a SettlementWindow or None")
        if settlement is not None and settlement.snapshot.periods != (request.origin,):
            raise ValueError("spine settlement window must contain exactly its origin")
        if settlement is not None and settlement.token is not None:
            raise _EngineError("spine settlement token is engine-owned")
        if expected_forecast_origin_count is not None and (
            not isinstance(expected_forecast_origin_count, int)
            or isinstance(expected_forecast_origin_count, bool)
            or expected_forecast_origin_count < 0
        ):
            raise ValueError("expected forecast origin count must be a non-negative integer")

        def resolve_phase() -> ObservationResult:
            return self._engine.observe(
                request.origin,
                session=request.session,
                snapshot=snapshot,
            )

        observation = self._phase(
            Phase.RESOLVE,
            request.origin,
            resolve_phase,
        )
        assert observation.token is not None
        token = observation.token
        if settlement is not None:
            settlement = SettlementWindow(
                snapshot=settlement.snapshot,
                actuals=settlement.actuals,
                actuals_semantics=settlement.actuals_semantics,
                token=token,
            )

        def predict_phase() -> ForecastBatch:
            fitted = self._engine.fit(request)
            return self._engine.predict(fitted)

        predicted = self._phase(Phase.PREDICT, request.origin, predict_phase)
        reconciled = self._phase(
            Phase.RECONCILE,
            request.origin,
            lambda: self._engine.reconcile(predicted),
        )
        calibrated = self._phase(
            Phase.CALIBRATE,
            request.origin,
            lambda: self._engine.calibrate(
                reconciled,
                session=request.session,
                observation=observation,
            ),
        )
        order_request = OrderRequest(
            session=request.session,
            origin=request.origin,
            forecasts=calibrated.forecasts,
            inventory_positions=request.inventory_positions,
        )
        decisions = self._phase(
            Phase.ORDER,
            request.origin,
            lambda: self._engine.order(order_request) if decision_origin else None,
            unwrapped=(OrderingInputError,),
        )

        def commit_phase() -> CommitResult:
            return self._engine.commit(
                CommitRequest(
                    session=request.session,
                    origin=request.origin,
                    token=token,
                    observation=observation,
                    calibration=calibrated,
                    inventory_positions=request.inventory_positions,
                    decisions=decisions,
                    settlement=settlement,
                    input_fingerprint=input_fingerprint,
                    expected_forecast_origin_count=expected_forecast_origin_count,
                )
            )

        committed = self._phase(
            Phase.COMMIT,
            request.origin,
            commit_phase,
        )
        return OriginResult(
            forecasts=calibrated.forecasts,
            orders=committed.orders,
            receipt=committed.receipt,
        )

    def _phase(
        self,
        phase: Phase,
        origin: pd.Timestamp,
        action,
        *,
        unwrapped: tuple[type[Exception], ...] = (),
    ):
        started = time.perf_counter()
        try:
            result = action()
        except Exception as error:
            self._engine._abort_origin(origin)
            self._report(phase, origin, started, error)
            if isinstance(error, unwrapped):
                raise
            raise PhaseError(phase, origin, error) from error
        self._report(phase, origin, started, None)
        return result

    def _report(
        self,
        phase: Phase,
        origin: pd.Timestamp,
        started: float,
        error: Exception | None,
    ) -> None:
        if self._reporter is None:
            return
        try:
            self._reporter(
                PhaseEvent(
                    phase=phase,
                    origin=origin,
                    duration_seconds=time.perf_counter() - started,
                    error=None if error is None else str(error),
                )
            )
        except Exception as reporter_error:
            try:
                warnings.warn(
                    f"phase reporter failed for {phase.value} at {origin}: {reporter_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                return


def _bind_forecast_batch(
    forecasts: ForecastBatch,
    *,
    token: CycleToken,
) -> ForecastBatch:
    if forecasts.token is not None and forecasts.token != token:
        raise _EngineError("forecast batch belongs to another cycle")
    if forecasts.token == token:
        return forecasts
    bound = ForecastBatch(
        forecasts.frame,
        calendar=forecasts.calendar,
        issuances=forecasts.issuances,
        observation_issuances=forecasts.observation_issuances,
    )
    object.__setattr__(bound, "_token", token)
    return bound


def _snapshot_commit_receipt(receipt: object) -> CommitReceipt:
    """Collapse a store callback result into one stable, exact receipt."""
    if not isinstance(receipt, CommitReceipt):
        raise _EngineError("run store must return a CommitReceipt")
    return CommitReceipt(
        session=receipt.session,
        origin=receipt.origin,
        digest=receipt.digest,
        expected_revision=receipt.expected_revision,
        revision=receipt.revision,
        state_updates=dict(receipt.state_updates),
        has_forecasts=receipt.has_forecasts,
        observe_cycle=receipt.observe_cycle,
        settlement_periods=tuple(receipt.settlement_periods),
        actual_keys=tuple(receipt.actual_keys),
        input_fingerprint=receipt.input_fingerprint,
        orders=tuple(receipt.orders),
        inventory_positions=dict(receipt.inventory_positions),
        resume_marker=receipt.resume_marker,
    )


def _bind_observation_result(
    cycle: ObserveCycle,
    prior_states: Mapping[str, bytes],
    *,
    token: CycleToken,
) -> ObservationResult:
    return ObservationResult(cycle, prior_states, token=token)


def _bind_calibration_result(
    forecasts: ForecastBatch,
    state_updates: Mapping[str, bytes],
    *,
    token: CycleToken,
) -> CalibrationResult:
    return CalibrationResult(forecasts, state_updates, token=token)


def _forecast_batch_origin(forecasts: ForecastBatch) -> pd.Timestamp:
    origins = {key[1] for key in forecasts.issuances}
    if len(origins) != 1:
        raise _EngineError("forecast batch must contain exactly one origin")
    return next(iter(origins))


def _forecast_keys(frame: pd.DataFrame) -> tuple[ForecastKey, ...]:
    return tuple(
        (str(series), pd.Timestamp(origin), int(step), str(model))
        for series, origin, step, model in zip(
            frame[SERIES_KEY],
            frame[ORIGIN],
            frame[HORIZON_STEP],
            frame[MODEL_NAME],
            strict=True,
        )
    )


def _ledger_observation_issuances(
    values: Mapping[ConformalForecastKey, IssuedBoundFacts],
) -> dict[ForecastKey, IssuedBoundFacts]:
    issuances: dict[ForecastKey, IssuedBoundFacts] = {}
    for key, facts in values.items():
        ledger_key = (key.series_key, key.origin, key.horizon_step, key.model_name)
        if ledger_key in issuances:
            raise _EngineError(f"duplicate conformal issuance key: {ledger_key!r}")
        issuances[ledger_key] = IssuedBoundFacts.snapshot(facts)
    return issuances


def _ledger_bound_issuances(
    values: Mapping[ForecastKey, IssuedBoundFacts],
) -> dict[ForecastKey, dict[BoundKey, ForecastIssuance]]:
    issuances: dict[ForecastKey, dict[BoundKey, ForecastIssuance]] = {}
    for key, facts in values.items():
        lower, upper = interval_columns(facts.effective_descriptor.level)
        if facts.emission_form is EmissionForm.TWO_SIDED:
            bound_key: BoundKey = (lower, upper)
            selected = (facts.lower_bound, facts.upper_bound)
            side = None
        elif facts.emission_form is EmissionForm.ONE_SIDED_LOWER:
            bound_key = (lower,)
            selected = (facts.lower_bound,)
            side = GuaranteedSide.LOWER
        elif facts.emission_form is EmissionForm.ONE_SIDED_UPPER:
            bound_key = (upper,)
            selected = (facts.upper_bound,)
            side = GuaranteedSide.UPPER
        else:
            raise _EngineError(f"unsupported conformal emission form: {facts.emission_form!r}")
        guaranteed_side = (
            side
            if facts.effective_descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
            else None
        )
        finite = all(math.isfinite(value) for value in selected)
        issuances[key] = {
            bound_key: ForecastIssuance(
                descriptor=facts.effective_descriptor,
                guaranteed_side=guaranteed_side,
                calibration_ready=facts.calibration_ready,
                bounds_finite=finite,
                bounds_null_reason=None if finite else facts.bounds_null_reason,
            )
        }
    return issuances


def _calibration_context(
    runtime: ConformalRuntime,
    forecasts: pd.DataFrame,
    *,
    hierarchy: HierarchyIndex,
) -> CalibrationContext | None:
    if not runtime.manifest.consumes_calibration_context:
        return None
    nodes = {node.label: node for node in hierarchy.nodes}
    try:
        aligned = tuple(nodes[str(series_key)] for series_key in forecasts[SERIES_KEY])
    except KeyError as error:
        raise _EngineError(
            f"forecast row names an unknown hierarchy node: {error.args[0]!r}"
        ) from error
    return CalibrationContext(
        series_keys=tuple(node.label for node in aligned),
        lattice_levels=tuple(node.kind.value for node in aligned),
        aggregate_memberships=tuple(node.members for node in aligned),
    )


def _bottom_node_order_request(
    request: OrderRequest,
    *,
    series_keys: tuple[str, ...],
    token: CycleToken | None,
) -> OrderRequest:
    if token is None:
        raise _EngineError("order request forecasts must carry a cycle token")
    allowed = frozenset(series_keys)
    foreign_positions = set(request.inventory_positions) - allowed
    if foreign_positions:
        raise _EngineError(
            f"inventory positions contain non-decision series: {sorted(foreign_positions)!r}"
        )
    if all(key[0] in allowed for key in request.forecasts.issuances):
        return request

    frame = request.forecasts.frame
    filtered = frame.loc[frame[SERIES_KEY].isin(allowed)].reset_index(drop=True)
    issuances = {
        key: value for key, value in request.forecasts.issuances.items() if key[0] in allowed
    }
    observation_issuances = {
        key: value
        for key, value in request.forecasts.observation_issuances.items()
        if key[0] in allowed
    }
    filtered_forecasts = _bind_forecast_batch(
        ForecastBatch(
            filtered,
            calendar=request.forecasts.calendar,
            issuances=issuances,
            observation_issuances=observation_issuances,
        ),
        token=token,
    )
    return OrderRequest(
        session=request.session,
        origin=request.origin,
        forecasts=filtered_forecasts,
        inventory_positions=request.inventory_positions,
    )


def _decision_keys(forecasts: ForecastBatch) -> tuple[DecisionKey, ...]:
    return tuple(
        sorted(
            {
                (series_key, model_name)
                for series_key, _origin, _step, model_name in forecasts.issuances
            },
            key=lambda key: (key[0].encode(), key[1].encode()),
        )
    )


def _require_decision_series_coverage(
    forecasts: ForecastBatch,
    *,
    series_keys: tuple[str, ...],
    task_horizon: int,
) -> None:
    expected = set(series_keys)
    supplied = {series_key for series_key, _origin, _step, _model in forecasts.issuances}
    if supplied == expected:
        expected_steps = set(range(1, task_horizon + 1))
        steps_by_group: dict[DecisionKey, set[int]] = {}
        for series_key, _origin, step, model_name in forecasts.issuances:
            steps_by_group.setdefault((series_key, model_name), set()).add(step)
        incomplete = sorted(
            (group, sorted(steps))
            for group, steps in steps_by_group.items()
            if steps != expected_steps
        )
        if not incomplete:
            return
        raise _EngineError(
            "ordering forecast horizons must exactly cover the session horizon per "
            f"decision group; invalid={incomplete!r}"
        )
    missing = sorted(expected - supplied, key=str.encode)
    unexpected = sorted(supplied - expected, key=str.encode)
    raise _EngineError(
        "ordering forecasts must exactly cover configured decision series; "
        f"missing={missing!r}, unexpected={unexpected!r}"
    )


def _require_inventory_coverage(
    inventory_positions: Mapping[str, InventoryPosition],
    *,
    requested: tuple[DecisionKey, ...],
) -> None:
    expected = {series_key for series_key, _model_name in requested}
    supplied = set(inventory_positions)
    if supplied == expected:
        return
    missing = sorted(expected - supplied, key=str.encode)
    unexpected = sorted(supplied - expected, key=str.encode)
    raise _EngineError(
        "inventory positions must exactly cover requested decision series; "
        f"missing={missing!r}, unexpected={unexpected!r}"
    )


def _require_timestamp(value: object, *, name: str) -> None:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise _EngineError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise _EngineError(f"{name} must be timezone-naive")


def _validated_state_updates(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise TypeError("calibration state updates must be a mapping")
    updates: dict[str, bytes] = {}
    for partition, state in value.items():
        _require_trimmed_identifier(partition, name="calibration partition")
        if not isinstance(state, bytes):
            raise TypeError("calibration state updates must contain bytes")
        updates[partition] = state
    return updates


def _validated_state_snapshot(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise TypeError("calibration state snapshot must be a mapping")
    states: dict[str, bytes] = {}
    for label, state in value.items():
        _require_trimmed_identifier(label, name="calibration state label")
        if not isinstance(state, bytes):
            raise TypeError("calibration state snapshot must contain bytes")
        states[label] = state
    return states


def _validated_inventory_positions(
    value: Mapping[str, InventoryPosition],
) -> dict[str, InventoryPosition]:
    if not isinstance(value, Mapping):
        raise TypeError("inventory positions must be a mapping")
    positions: dict[str, InventoryPosition] = {}
    for series_key, position in value.items():
        _require_utf8_identifier(series_key, name="inventory-position series key")
        if not isinstance(position, InventoryPosition):
            raise TypeError("inventory positions must contain InventoryPosition values")
        positions[series_key] = position
    return positions


def _require_utf8_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise _EngineError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise _EngineError(f"{name} must be valid UTF-8") from error


def _require_trimmed_identifier(value: object, *, name: str) -> None:
    _require_utf8_identifier(value, name=name)
    assert isinstance(value, str)
    if value != value.strip():
        raise _EngineError(f"{name} must be a non-empty trimmed string")


__all__ = [
    "ENGINE_VERBS",
    "CalibrationResult",
    "CommitRequest",
    "CommitResult",
    "ConfiguredPolicyOrderer",
    "DecisionBatch",
    "DecisionEvidence",
    "Engine",
    "FittedTask",
    "ForecastBatch",
    "ObservationResult",
    "OrderProposal",
    "OrderRequest",
    "OriginRequest",
    "OriginResult",
    "Phase",
    "PhaseError",
    "PhaseEvent",
    "SettlementWindow",
    "Spine",
]
