"""Run the fixed chapter-03 engine spine over six abstract ports."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import islice
from types import MappingProxyType
from typing import cast, overload

import pandas as pd

from newcalibre.domain import (
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    SERIES_KEY,
    ActualsSemantics,
    Calendar,
    DecisionTiming,
    ForecastTask,
    InventoryPosition,
    Scope,
    SessionIdentity,
    StockoutRule,
    forecast_bound_groups,
    validate_forecast_frame,
)
from newcalibre.engine._ordering_runtime import (
    DecisionBatch,
    DecisionKey,
    OrderProposal,
)
from newcalibre.engine._ordering_runtime import (
    materialize_decisions as _materialize_decisions,
)
from newcalibre.engine._session import (
    SessionCosts,
)
from newcalibre.engine._session import (
    require_panel_session_binding as _require_panel_session_binding,
)
from newcalibre.engine._session import (
    require_task_session_binding as _require_task_session_binding,
)
from newcalibre.engine._session import session_decision_inputs as _session_decision_inputs
from newcalibre.engine._session import session_model_config as _session_model_config
from newcalibre.engine._session import (
    session_ordering_configuration as _session_ordering_configuration,
)
from newcalibre.engine._session import (
    session_origin_inputs as _session_origin_inputs,
)
from newcalibre.engine.errors import EngineError as _EngineError
from newcalibre.engine.ports import (
    ActualKey,
    ActualsSource,
    ArtifactStore,
    CalibrationStateStore,
    CommitReceipt,
    DispatchBackend,
    ForecastWrite,
    LedgerSink,
    OriginCommit,
    PanelSource,
    SettlementSnapshot,
)
from newcalibre.engine.settlement import SettlementRequest as _SettlementRequest
from newcalibre.engine.settlement import SettlementResult as _SettlementResult
from newcalibre.engine.settlement import settle as _settle
from newcalibre.forecasting import AdapterCapability, ForecastAdapter, resolve_adapter
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    OrderRow,
)

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
    _session: SessionIdentity | None = field(repr=False)
    _engine_token: object | None = field(repr=False, compare=False)
    issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]]

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        calendar: Calendar,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]] | None = None,
    ) -> None:
        owned_frame = pd.DataFrame(frame, copy=True)
        validated = validate_forecast_frame(owned_frame, calendar=calendar)
        keys = _forecast_keys(validated)
        if issuances is None:
            if forecast_bound_groups(validated.columns):
                raise _EngineError("forecast bound columns require explicit issuance metadata")
            supplied: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]] = {
                key: {} for key in keys
            }
        else:
            supplied = issuances
        if set(supplied) != set(keys):
            raise _EngineError("forecast issuance keys must exactly match the frame keys")
        frozen = {key: MappingProxyType(dict(supplied[key])) for key in keys}
        object.__setattr__(self, "_frame", validated)
        object.__setattr__(self, "_calendar", calendar)
        object.__setattr__(self, "_session", None)
        object.__setattr__(self, "_engine_token", None)
        object.__setattr__(self, "issuances", MappingProxyType(frozen))

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
        return self._session


@dataclass(frozen=True, slots=True)
class FittedTask:
    """Address fitted state without carrying model objects, bytes, or locations."""

    session: SessionIdentity
    task: ForecastTask

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("fitted task session must be a SessionIdentity")
        if not isinstance(self.task, ForecastTask):
            raise TypeError("fitted task task must be a ForecastTask")
        _require_task_session_binding(self.task, session=self.session)


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Stage due-row resolutions and their resulting calibration state."""

    resolutions: Mapping[ForecastKey, float]
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    prior_states: Mapping[str, bytes | None] = field(default_factory=dict)
    session: SessionIdentity | None = None
    origin: pd.Timestamp | None = None
    _engine_token: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resolutions, Mapping):
            raise TypeError("observation resolutions must be a mapping")
        object.__setattr__(self, "resolutions", MappingProxyType(dict(self.resolutions)))
        updates = _validated_state_updates(self.state_updates)
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        prior_states = _validated_state_snapshot(self.prior_states)
        object.__setattr__(self, "prior_states", MappingProxyType(prior_states))
        _validate_optional_provenance(
            self.session,
            self.origin,
            name="observation result",
        )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Return a calibrated frame plus state mutations staged for Commit."""

    forecasts: ForecastBatch
    state_updates: Mapping[str, bytes] = field(default_factory=dict)
    session: SessionIdentity | None = None
    origin: pd.Timestamp | None = None
    _engine_token: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.forecasts, ForecastBatch):
            raise TypeError("calibration result forecasts must be a ForecastBatch")
        updates = _validated_state_updates(self.state_updates)
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        _validate_optional_provenance(
            self.session,
            self.origin,
            name="calibration result",
        )
        if self.session is not None:
            if self.forecasts.session != self.session:
                raise _EngineError("calibration result forecasts must match its session")
            assert self.origin is not None
            if any(key[1] != self.origin for key in self.forecasts.issuances):
                raise _EngineError("calibration result forecasts must match its origin")


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
    calibration_partitions: tuple[str, ...]
    _model_config: bytes = field(repr=False)
    _future_exogenous: pd.DataFrame | None = field(repr=False)
    _inventory_positions: Mapping[str, InventoryPosition] = field(repr=False)

    def __init__(
        self,
        *,
        session: SessionIdentity,
        origin: pd.Timestamp,
        scope: Scope,
        calibration_partitions: Sequence[str] = (),
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
        partitions = _validated_partitions(calibration_partitions)
        if future_exogenous is not None and not isinstance(future_exogenous, pd.DataFrame):
            raise TypeError("future exogenous input must be a pandas DataFrame")
        positions = _validated_inventory_positions(inventory_positions or {})
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "calibration_partitions", partitions)
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
    """Return the committed forecast and order facts for one origin."""

    forecasts: ForecastBatch
    orders: tuple[OrderRow, ...]


@dataclass(frozen=True, slots=True)
class SettlementWindow:
    """Carry authoritative actuals and the compact ledger projection for one cycle."""

    snapshot: SettlementSnapshot
    actuals: Mapping[ActualKey, float]
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SettlementSnapshot):
            raise TypeError("settlement window snapshot must be a SettlementSnapshot")
        if not isinstance(self.actuals, Mapping):
            raise TypeError("settlement window actuals must be a mapping")
        if not isinstance(self.actuals_semantics, ActualsSemantics):
            raise TypeError("settlement window semantics must be ActualsSemantics")
        object.__setattr__(self, "actuals", MappingProxyType(dict(self.actuals)))


@dataclass(frozen=True, slots=True)
class CommitRequest:
    """Carry typed phase results into the one materializing Commit boundary."""

    session: SessionIdentity
    origin: pd.Timestamp
    observation: ObservationResult
    calibration: CalibrationResult
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    decisions: DecisionBatch | None = None
    settlement: SettlementWindow | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("commit request session must be a SessionIdentity")
        _require_timestamp(self.origin, name="commit request origin")
        if not isinstance(self.observation, ObservationResult):
            raise TypeError("commit request observation must be an ObservationResult")
        if not isinstance(self.calibration, CalibrationResult):
            raise TypeError("commit request calibration must be a CalibrationResult")
        if self.observation.session != self.session or self.observation.origin != self.origin:
            raise _EngineError("commit request observation must match its session and origin")
        if self.calibration.session != self.session or self.calibration.origin != self.origin:
            raise _EngineError("commit request calibration must match its session and origin")
        if any(key[1] != self.origin for key in self.calibration.forecasts.issuances):
            raise _EngineError("commit request forecasts must all match its origin")
        if self.decisions is not None:
            if not isinstance(self.decisions, DecisionBatch):
                raise TypeError("commit request decisions must be a DecisionBatch or None")
            if self.decisions.session != self.session or self.decisions.origin != self.origin:
                raise _EngineError("commit request decisions must match its session and origin")
        if self.settlement is not None:
            if not isinstance(self.settlement, SettlementWindow):
                raise TypeError("commit request settlement must be a SettlementWindow or None")
            if self.settlement.snapshot.session != self.session:
                raise _EngineError("commit request settlement must match its session")
            if self.settlement.snapshot.periods != (self.origin,):
                raise _EngineError("commit request settlement must contain exactly its origin")
        positions = _validated_inventory_positions(self.inventory_positions)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Return the durable receipt and the exact rows materialized by Commit."""

    receipt: CommitReceipt
    orders: tuple[OrderRow, ...]
    settlement: _SettlementResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, CommitReceipt):
            raise TypeError("commit result receipt must be a CommitReceipt")
        orders = tuple(self.orders)
        if any(not isinstance(order, OrderRow) for order in orders):
            raise TypeError("commit result orders must contain OrderRow values")
        if self.settlement is not None and not isinstance(self.settlement, _SettlementResult):
            raise TypeError("commit result settlement must be a SettlementResult or None")
        object.__setattr__(self, "orders", orders)


type AdapterResolver = Callable[[Mapping[str, object]], ForecastAdapter]
type Reconciler = Callable[[ForecastBatch], ForecastBatch]
type Calibrator = Callable[
    [ForecastBatch, Mapping[str, bytes | None]],
    CalibrationResult,
]
type Observer = Callable[
    [pd.DataFrame, Mapping[ForecastKey, float], Mapping[str, bytes | None]],
    Mapping[str, bytes],
]
type Orderer = Callable[[OrderRequest], Iterable[OrderProposal]]
type PhaseReporter = Callable[[PhaseEvent], None]


class Engine:
    """Own the exact eight-verb engine surface over six ports."""

    def __init__(
        self,
        *,
        panel_source: PanelSource,
        actuals_source: ActualsSource,
        artifact_store: ArtifactStore,
        calibration_state_store: CalibrationStateStore,
        ledger_sink: LedgerSink,
        dispatch_backend: DispatchBackend,
        adapter_resolver: AdapterResolver = resolve_adapter,
        observer: Observer | None = None,
        reconciler: Reconciler | None = None,
        calibrator: Calibrator | None = None,
        orderer: Orderer | None = None,
    ) -> None:
        ports = (
            (panel_source, PanelSource, "panel source"),
            (actuals_source, ActualsSource, "actuals source"),
            (artifact_store, ArtifactStore, "artifact store"),
            (calibration_state_store, CalibrationStateStore, "calibration-state store"),
            (ledger_sink, LedgerSink, "ledger sink"),
            (dispatch_backend, DispatchBackend, "dispatch backend"),
        )
        for adapter, port, name in ports:
            if not isinstance(adapter, port):
                raise TypeError(f"engine {name} does not satisfy its port")
        callables = (
            (adapter_resolver, "adapter resolver"),
            (observer, "observer"),
            (reconciler, "reconciler"),
            (calibrator, "calibrator"),
            (orderer, "orderer"),
        )
        for hook, name in callables:
            if hook is not None and not callable(hook):
                raise TypeError(f"engine {name} must be callable")
        # Resolve configuration before the panel port can perform any data load.
        adapter_resolver(_session_model_config(ledger_sink.session))
        ordering_configuration = _session_ordering_configuration(ledger_sink.session)
        self._ordering_configuration = ordering_configuration
        self._panel = panel_source.load()
        self._actuals_source = actuals_source
        self._artifact_store = artifact_store
        self._calibration_state_store = calibration_state_store
        self._ledger_sink = ledger_sink
        self._dispatch_backend = dispatch_backend
        self._adapter_resolver = adapter_resolver
        self._observer = observer
        self._reconciler = reconciler
        self._calibrator = calibrator
        self._orderer = orderer
        self._phase_token = object()
        _require_panel_session_binding(
            self._panel,
            session=self._ledger_sink.session,
            ledger_calendar=self._ledger_sink.calendar,
        )

    def fit(self, request: OriginRequest) -> tuple[FittedTask, ...]:
        """Build deterministic tasks and fit or restore their adapters."""
        if not isinstance(request, OriginRequest):
            raise TypeError("fit requires an OriginRequest")
        self._require_session(request.session)
        tasks = self._panel.forecast_tasks(
            origin=request.origin,
            horizon=request.horizon,
            scope=request.scope,
            model_config=request.model_config,
            future_exogenous=request.future_exogenous,
        )
        fitted_tasks = tuple(FittedTask(session=request.session, task=task) for task in tasks)
        return self._dispatch_backend.map(self._fit_one, fitted_tasks)

    def predict(self, fitted_tasks: Sequence[FittedTask]) -> ForecastBatch:
        """Predict fitted tasks in deterministic dispatch order."""
        tasks = tuple(fitted_tasks)
        if not tasks:
            raise _EngineError("predict requires at least one fitted task")
        for fitted in tasks:
            if not isinstance(fitted, FittedTask):
                raise TypeError("predict requires FittedTask values")
            self._require_session(fitted.session)
        frames = self._dispatch_backend.map(self._predict_one, tasks)
        combined = pd.concat(frames, ignore_index=True)
        return _bind_forecast_batch(
            ForecastBatch(
                combined,
                calendar=self._ledger_sink.calendar,
            ),
            session=tasks[0].session,
            engine_token=self._phase_token,
        )

    def reconcile(self, forecasts: ForecastBatch) -> ForecastBatch:
        """Apply the configured reconciler, or return the input identity."""
        if not isinstance(forecasts, ForecastBatch):
            raise TypeError("reconcile requires a ForecastBatch")
        self._require_forecast_batch(forecasts)
        if self._reconciler is None:
            return forecasts
        callback_result = self._reconciler(forecasts)
        if not isinstance(callback_result, ForecastBatch):
            raise _EngineError("reconciler must return a ForecastBatch")
        reconciled = _snapshot_forecast_batch(callback_result)
        if reconciled.calendar != forecasts.calendar:
            raise _EngineError("reconciler changed the forecast calendar")
        removed = set(forecasts.issuances) - set(reconciled.issuances)
        if removed:
            raise _EngineError("reconciler removed or changed forecast row keys")
        assert forecasts.session is not None
        return _bind_forecast_batch(
            reconciled,
            session=forecasts.session,
            engine_token=self._phase_token,
        )

    def calibrate(
        self,
        forecasts: ForecastBatch,
        *,
        session: SessionIdentity,
        partitions: Sequence[str] = (),
        observation: ObservationResult | None = None,
    ) -> CalibrationResult:
        """Apply calibration from declared state, or return the input identity."""
        if not isinstance(forecasts, ForecastBatch):
            raise TypeError("calibrate requires a ForecastBatch")
        self._require_session(session)
        self._require_forecast_batch(forecasts)
        if forecasts.session != session:
            raise _EngineError("calibration forecasts do not match its session")
        origin = _forecast_batch_origin(forecasts)
        if observation is not None and (
            observation.session != session or observation.origin != origin
        ):
            raise _EngineError("calibration observation does not match its session and origin")
        if observation is not None and observation._engine_token is not self._phase_token:
            raise _EngineError("calibration observation was not produced by this engine")
        requested = _validated_partitions(partitions)
        observed_updates = {} if observation is None else dict(observation.state_updates)
        _reject_undeclared_state_updates(observed_updates, requested, producer="observer")
        if self._calibrator is None:
            return _bind_calibration_result(
                forecasts,
                observed_updates,
                session=session,
                origin=origin,
                engine_token=self._phase_token,
            )
        if observation is not None and set(observation.prior_states) == set(requested):
            states = dict(observation.prior_states)
        else:
            states = {
                partition: self._calibration_state_store.load(session, partition)
                for partition in requested
            }
        states.update(observed_updates)
        callback_result = self._calibrator(forecasts, states)
        if not isinstance(callback_result, CalibrationResult):
            raise _EngineError("calibrator must return a CalibrationResult")
        result = CalibrationResult(
            _snapshot_forecast_batch(callback_result.forecasts),
            dict(callback_result.state_updates),
            session=callback_result.session,
            origin=callback_result.origin,
        )
        if result.session is not None and (result.session != session or result.origin != origin):
            raise _EngineError("calibrator result does not match its session and origin")
        if result.forecasts.calendar != forecasts.calendar:
            raise _EngineError("calibrator changed the forecast calendar")
        if set(result.forecasts.issuances) != set(forecasts.issuances):
            raise _EngineError("calibrator changed forecast row keys")
        calibrated = _bind_forecast_batch(
            result.forecasts,
            session=session,
            engine_token=self._phase_token,
        )
        if _forecast_batch_origin(calibrated) != origin:
            raise _EngineError("calibrator changed the forecast origin")
        _reject_undeclared_state_updates(
            result.state_updates,
            requested,
            producer="calibrator",
        )
        merged_updates = observed_updates
        merged_updates.update(result.state_updates)
        return _bind_calibration_result(
            calibrated,
            merged_updates,
            session=session,
            origin=origin,
            engine_token=self._phase_token,
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
            engine_token=self._phase_token,
        )
        task_horizon, _model_config = _session_origin_inputs(request.session)
        _require_decision_series_coverage(
            decision_request.forecasts,
            series_keys=configuration.series_keys,
            task_horizon=task_horizon,
        )
        requested = _decision_keys(decision_request.forecasts)
        _require_inventory_coverage(decision_request.inventory_positions, requested=requested)
        proposals = tuple(islice(self._orderer(decision_request), len(requested) + 1))
        decisions = DecisionBatch(
            session=request.session,
            origin=request.origin,
            requested=requested,
            proposals=proposals,
        )
        object.__setattr__(decisions, "_engine_token", self._phase_token)
        return decisions

    def observe(
        self,
        origin: pd.Timestamp,
        *,
        session: SessionIdentity,
        partitions: Sequence[str] = (),
    ) -> ObservationResult:
        """Match due ledger rows to actuals admissible at this origin."""
        self._require_session(session)
        requested = _validated_partitions(partitions)
        due = self._ledger_sink.due_frame(origin)
        due_rows = tuple(due.itertuples(index=False))
        actual_keys = tuple(
            (cast(str, row.series_key), cast(pd.Timestamp, row.target_timestamp))
            for row in due_rows
        )
        actuals = self._actuals_source.for_keys(actual_keys, before=origin)
        resolutions: dict[ForecastKey, float] = {}
        for row in due_rows:
            series_key = cast(str, row.series_key)
            target_timestamp = cast(pd.Timestamp, row.target_timestamp)
            actual = actuals.get((series_key, target_timestamp))
            if actual is not None:
                key: ForecastKey = (
                    series_key,
                    cast(pd.Timestamp, row.origin),
                    cast(int, row.horizon_step),
                    cast(str, row.model_name),
                )
                resolutions[key] = actual
        if self._observer is None:
            return _bind_observation_result(
                resolutions,
                session=session,
                origin=origin,
                engine_token=self._phase_token,
            )
        states = MappingProxyType(
            {
                partition: self._calibration_state_store.load(session, partition)
                for partition in requested
            }
        )
        updates = self._observer(due, MappingProxyType(resolutions), states)
        result = _bind_observation_result(
            resolutions,
            updates,
            states,
            session=session,
            origin=origin,
            engine_token=self._phase_token,
        )
        _reject_undeclared_state_updates(
            result.state_updates,
            requested,
            producer="observer",
        )
        return result

    def settle(self, request: _SettlementRequest) -> _SettlementResult:
        """Apply the engine's single pure settlement implementation."""
        if not isinstance(request, _SettlementRequest):
            raise TypeError("settle requires a SettlementRequest")
        self._require_session(request.session)
        if request.snapshot.calendar != self._ledger_sink.calendar:
            raise _EngineError("settlement snapshot calendar does not match the engine ledger")
        owned_snapshot = self._ledger_sink.settlement_snapshot(request.snapshot.periods)
        if request.snapshot != owned_snapshot:
            raise _EngineError("settlement snapshot does not match the engine ledger")
        return _settle(request)

    @overload
    def commit(self, request: CommitRequest) -> CommitResult: ...

    @overload
    def commit(self, request: OriginCommit | CommitReceipt) -> CommitReceipt: ...

    def commit(
        self,
        request: CommitRequest | OriginCommit | CommitReceipt,
    ) -> CommitResult | CommitReceipt:
        """Persist an origin's ledger and monotone calibration-state mutations."""
        if isinstance(request, CommitRequest):
            return self._commit_request(request)
        if not isinstance(request, (OriginCommit, CommitReceipt)):
            raise TypeError("commit requires a CommitRequest, OriginCommit, or CommitReceipt")
        self._require_session(request.session)
        if isinstance(request, OriginCommit):
            expected = CommitReceipt.from_commit(request)
            receipt = _snapshot_commit_receipt(self._ledger_sink.commit(request))
            if receipt != expected:
                raise _EngineError("ledger sink returned a mismatched commit receipt")
        else:
            callback_receipt = self._ledger_sink.receipt(request.origin)
            if callback_receipt is None:
                raise _EngineError("commit receipt is not journaled by the ledger sink")
            receipt = _snapshot_commit_receipt(callback_receipt)
            if receipt != request:
                raise _EngineError("commit receipt does not match the ledger journal")
        for partition, value in receipt.state_updates.items():
            self._calibration_state_store.save(
                receipt.session,
                partition,
                value,
                origin=receipt.origin,
            )
        return receipt

    def _commit_request(self, request: CommitRequest) -> CommitResult:
        self._require_session(request.session)
        if request.observation._engine_token is not self._phase_token:
            raise _EngineError("commit observation was not produced by this engine")
        if request.calibration._engine_token is not self._phase_token:
            raise _EngineError("commit calibration was not produced by this engine")
        if (
            request.decisions is not None
            and request.decisions._engine_token is not self._phase_token
        ):
            raise _EngineError("commit decisions were not produced by this engine")
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
                engine_token=self._phase_token,
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
        orders = _materialize_decisions(
            request.decisions,
            configuration=configuration,
            calendar=self._ledger_sink.calendar,
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
            )
            settlement_result = (
                self.settle(settlement_request)
                if self._ledger_sink.receipt(request.origin) is None
                else _settle(settlement_request)
            )
        receipt = self.commit(
            OriginCommit(
                session=request.session,
                origin=request.origin,
                resolutions=request.observation.resolutions,
                forecasts=(
                    ForecastWrite(
                        request.calibration.forecasts.frame,
                        request.calibration.forecasts.issuances,
                    ),
                ),
                orders=orders,
                settlements=() if settlement_result is None else settlement_result.records,
                state_updates=request.calibration.state_updates,
            )
        )
        return CommitResult(receipt=receipt, orders=orders, settlement=settlement_result)

    def _require_session(self, session: SessionIdentity) -> None:
        if session != self._ledger_sink.session:
            raise _EngineError("engine session does not match its ledger sink")

    def _require_forecast_batch(self, forecasts: ForecastBatch) -> None:
        if forecasts._engine_token is not self._phase_token or forecasts.session is None:
            raise _EngineError("forecast batch was not produced by this engine")
        self._require_session(forecasts.session)
        if forecasts.calendar != self._ledger_sink.calendar:
            raise _EngineError("forecast batch calendar does not match the engine ledger")

    def _require_time_loop_ports(
        self,
        *,
        actuals_source: ActualsSource,
        ledger_sink: LedgerSink,
    ) -> None:
        """Reject a driver wired to different ports than this engine."""
        if actuals_source is not self._actuals_source:
            raise _EngineError("time loop actuals source does not belong to the engine")
        if ledger_sink is not self._ledger_sink:
            raise _EngineError("time loop ledger sink does not belong to the engine")

    def _fit_one(self, fitted: FittedTask) -> FittedTask:
        adapter = self._adapter_resolver(fitted.task.model_config)
        if AdapterCapability.ARTIFACT_PERSISTENCE in adapter.capabilities:
            artifact_key = _artifact_key(fitted.session, fitted.task)
            stored = self._artifact_store.load(artifact_key)
            if stored is None:
                adapter.fit(fitted.task)
                self._artifact_store.save(artifact_key, adapter.dump_state())
            else:
                adapter.load_state(stored)
        else:
            adapter.fit(fitted.task)
        return fitted

    def _predict_one(self, fitted: FittedTask) -> pd.DataFrame:
        if not isinstance(fitted, FittedTask):
            raise TypeError("dispatch prediction item must be a FittedTask")
        adapter = self._adapter_resolver(fitted.task.model_config)
        if AdapterCapability.ARTIFACT_PERSISTENCE in adapter.capabilities:
            artifact_key = _artifact_key(fitted.session, fitted.task)
            stored = self._artifact_store.load(artifact_key)
            if stored is None:
                raise _EngineError("fitted model artifact is missing")
            adapter.load_state(stored)
        else:
            adapter.fit(fitted.task)
        return adapter.predict(fitted.task)


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
        decision_origin: bool = True,
        settlement: SettlementWindow | None = None,
    ) -> OriginResult:
        """Run Resolve through Commit once for a declared origin."""
        if not isinstance(request, OriginRequest):
            raise TypeError("spine requires an OriginRequest")
        if not isinstance(decision_origin, bool):
            raise TypeError("decision_origin must be boolean")
        if settlement is not None and not isinstance(settlement, SettlementWindow):
            raise TypeError("settlement must be a SettlementWindow or None")
        if settlement is not None and settlement.snapshot.periods != (request.origin,):
            raise ValueError("spine settlement window must contain exactly its origin")
        observation = self._phase(
            Phase.RESOLVE,
            request.origin,
            lambda: self._engine.observe(
                request.origin,
                session=request.session,
                partitions=request.calibration_partitions,
            ),
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
                partitions=request.calibration_partitions,
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
        )

        def commit_phase() -> tuple[OrderRow, ...]:
            committed = self._engine.commit(
                CommitRequest(
                    session=request.session,
                    origin=request.origin,
                    observation=observation,
                    calibration=calibrated,
                    inventory_positions=request.inventory_positions,
                    decisions=decisions,
                    settlement=settlement,
                )
            )
            return committed.orders

        orders = self._phase(
            Phase.COMMIT,
            request.origin,
            commit_phase,
        )
        return OriginResult(forecasts=calibrated.forecasts, orders=orders)

    def _phase(self, phase: Phase, origin: pd.Timestamp, action):
        started = time.perf_counter()
        try:
            result = action()
        except Exception as error:
            self._report(phase, origin, started, error)
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
    session: SessionIdentity,
    engine_token: object,
) -> ForecastBatch:
    if forecasts._engine_token is not None and forecasts._engine_token is not engine_token:
        raise _EngineError("forecast batch was produced by another engine")
    if forecasts.session is not None and forecasts.session != session:
        raise _EngineError("forecast batch belongs to another engine session")
    if forecasts.session == session and forecasts._engine_token is engine_token:
        return forecasts
    bound = ForecastBatch(
        forecasts.frame,
        calendar=forecasts.calendar,
        issuances=forecasts.issuances,
    )
    object.__setattr__(bound, "_session", session)
    object.__setattr__(bound, "_engine_token", engine_token)
    return bound


def _snapshot_forecast_batch(forecasts: ForecastBatch) -> ForecastBatch:
    """Collapse callback subclasses into one stable, exact forecast value."""
    frame = forecasts.frame
    calendar = forecasts.calendar
    issuances = {key: dict(bounds) for key, bounds in forecasts.issuances.items()}
    session = forecasts.session
    engine_token = forecasts._engine_token
    snapshot = ForecastBatch(frame, calendar=calendar, issuances=issuances)
    object.__setattr__(snapshot, "_session", session)
    object.__setattr__(snapshot, "_engine_token", engine_token)
    return snapshot


def _snapshot_commit_receipt(receipt: object) -> CommitReceipt:
    """Collapse a ledger callback result into one stable, exact receipt."""
    if not isinstance(receipt, CommitReceipt):
        raise _EngineError("ledger sink must return a CommitReceipt")
    return CommitReceipt(
        session=receipt.session,
        origin=receipt.origin,
        digest=receipt.digest,
        state_updates=dict(receipt.state_updates),
        settlement_periods=tuple(receipt.settlement_periods),
    )


def _bind_observation_result(
    resolutions: Mapping[ForecastKey, float],
    state_updates: Mapping[str, bytes] | None = None,
    prior_states: Mapping[str, bytes | None] | None = None,
    *,
    session: SessionIdentity,
    origin: pd.Timestamp,
    engine_token: object,
) -> ObservationResult:
    result = ObservationResult(
        resolutions,
        {} if state_updates is None else state_updates,
        {} if prior_states is None else prior_states,
        session=session,
        origin=origin,
    )
    object.__setattr__(result, "_engine_token", engine_token)
    return result


def _bind_calibration_result(
    forecasts: ForecastBatch,
    state_updates: Mapping[str, bytes],
    *,
    session: SessionIdentity,
    origin: pd.Timestamp,
    engine_token: object,
) -> CalibrationResult:
    result = CalibrationResult(
        forecasts,
        state_updates,
        session=session,
        origin=origin,
    )
    object.__setattr__(result, "_engine_token", engine_token)
    return result


def _forecast_batch_origin(forecasts: ForecastBatch) -> pd.Timestamp:
    origins = {key[1] for key in forecasts.issuances}
    if len(origins) != 1:
        raise _EngineError("forecast batch must contain exactly one origin")
    return next(iter(origins))


def _validate_optional_provenance(
    session: SessionIdentity | None,
    origin: pd.Timestamp | None,
    *,
    name: str,
) -> None:
    if session is None and origin is None:
        return
    if not isinstance(session, SessionIdentity):
        raise TypeError(f"{name} session must be a SessionIdentity")
    _require_timestamp(origin, name=f"{name} origin")


def _artifact_key(session: SessionIdentity, task: ForecastTask) -> str:
    digest = hashlib.sha256()
    for payload in (
        b"newcalibre.forecast-model/v1",
        session.value.encode(),
        task.to_bytes(),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"forecast-model:{digest.hexdigest()}"


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


def _bottom_node_order_request(
    request: OrderRequest,
    *,
    series_keys: tuple[str, ...],
    engine_token: object,
) -> OrderRequest:
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
    filtered_forecasts = _bind_forecast_batch(
        ForecastBatch(
            filtered,
            calendar=request.forecasts.calendar,
            issuances=issuances,
        ),
        session=request.session,
        engine_token=engine_token,
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


def _validated_partitions(partitions: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(partitions)
    if len(set(requested)) != len(requested):
        raise _EngineError("calibration partitions must be unique")
    for partition in requested:
        _require_trimmed_identifier(partition, name="calibration partition")
    return requested


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


def _validated_state_snapshot(
    value: Mapping[str, bytes | None],
) -> dict[str, bytes | None]:
    if not isinstance(value, Mapping):
        raise TypeError("calibration state snapshot must be a mapping")
    states: dict[str, bytes | None] = {}
    for partition, state in value.items():
        _require_trimmed_identifier(partition, name="calibration partition")
        if state is not None and not isinstance(state, bytes):
            raise TypeError("calibration state snapshot must contain bytes or None")
        states[partition] = state
    return states


def _reject_undeclared_state_updates(
    updates: Mapping[str, bytes],
    partitions: Sequence[str],
    *,
    producer: str,
) -> None:
    unknown = set(updates) - set(partitions)
    if unknown:
        raise _EngineError(f"{producer} updated undeclared partitions: {sorted(unknown)!r}")


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
    "DecisionBatch",
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
