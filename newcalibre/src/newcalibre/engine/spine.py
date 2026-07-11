"""Run the fixed chapter-03 engine spine over six abstract ports."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from types import MappingProxyType
from typing import cast

import pandas as pd

from newcalibre.domain import (
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    SERIES_KEY,
    Calendar,
    CostStructure,
    DecisionTiming,
    ForecastTask,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    forecast_bound_groups,
    validate_forecast_frame,
)
from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.engine.ports import (
    ActualKey,
    ActualsSource,
    ArtifactStore,
    CalibrationStateStore,
    CommitReceipt,
    DispatchBackend,
    ForecastWrite,
    LedgerSink,
    LedgerSnapshot,
    OriginCommit,
    PanelSource,
)
from newcalibre.forecasting import AdapterCapability, ForecastAdapter, resolve_adapter
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    OrderRow,
    SettlementRecord,
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


class EngineError(ValueError):
    """Report an invalid engine input or stage result."""


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
    issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]]

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        calendar: Calendar,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]] | None = None,
    ) -> None:
        validated = validate_forecast_frame(frame, calendar=calendar)
        keys = _forecast_keys(validated)
        if issuances is None:
            if forecast_bound_groups(validated.columns):
                raise EngineError("forecast bound columns require explicit issuance metadata")
            supplied: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]] = {
                key: {} for key in keys
            }
        else:
            supplied = issuances
        if set(supplied) != set(keys):
            raise EngineError("forecast issuance keys must exactly match the frame keys")
        frozen = {key: MappingProxyType(dict(supplied[key])) for key in keys}
        object.__setattr__(self, "_frame", validated)
        object.__setattr__(self, "_calendar", calendar)
        object.__setattr__(self, "issuances", MappingProxyType(frozen))

    @property
    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the validated forecast frame."""
        return self._frame.copy(deep=True)

    @property
    def calendar(self) -> Calendar:
        """Return the bound calendar used to validate this batch."""
        return self._calendar


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

    def __post_init__(self) -> None:
        if not isinstance(self.resolutions, Mapping):
            raise TypeError("observation resolutions must be a mapping")
        object.__setattr__(self, "resolutions", MappingProxyType(dict(self.resolutions)))
        updates = _validated_state_updates(self.state_updates)
        object.__setattr__(self, "state_updates", MappingProxyType(updates))
        prior_states = _validated_state_snapshot(self.prior_states)
        object.__setattr__(self, "prior_states", MappingProxyType(prior_states))


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Return a calibrated frame plus state mutations staged for Commit."""

    forecasts: ForecastBatch
    state_updates: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.forecasts, ForecastBatch):
            raise TypeError("calibration result forecasts must be a ForecastBatch")
        updates = _validated_state_updates(self.state_updates)
        object.__setattr__(self, "state_updates", MappingProxyType(updates))


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Give an ordering policy every declared decision-time fact."""

    session: SessionIdentity
    origin: pd.Timestamp
    forecasts: ForecastBatch
    inventory_positions: Mapping[str, InventoryPosition] = field(default_factory=dict)
    cost_structure: CostStructure | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("order request session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("order request origin must be a pandas Timestamp")
        if not isinstance(self.forecasts, ForecastBatch):
            raise TypeError("order request forecasts must be a ForecastBatch")
        positions = _validated_inventory_positions(self.inventory_positions)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        object.__setattr__(self, "cost_structure", _session_cost_structure(self.session))


@dataclass(frozen=True, slots=True)
class SettlementRequest:
    """Give the settlement module immutable facts and declared configuration."""

    session: SessionIdentity
    origin: pd.Timestamp
    ledger: LedgerSnapshot
    actuals: Mapping[ActualKey, float]
    inventory_positions: Mapping[str, InventoryPosition]
    timing: DecisionTiming
    cost_structure: CostStructure = field(init=False)
    stockout_rule: str
    actuals_semantics: str

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("settlement request session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp):
            raise TypeError("settlement request origin must be a pandas Timestamp")
        if not isinstance(self.ledger, LedgerSnapshot):
            raise TypeError("settlement request ledger must be a LedgerSnapshot")
        if self.ledger.session != self.session:
            raise EngineError("settlement request session must match its ledger snapshot")
        actuals = _validated_actuals(self.actuals)
        object.__setattr__(self, "actuals", MappingProxyType(actuals))
        positions = _validated_inventory_positions(self.inventory_positions)
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        if not isinstance(self.timing, DecisionTiming):
            raise TypeError("settlement request timing must be DecisionTiming")
        cost_structure = _session_cost_structure(self.session)
        if cost_structure is None:
            raise EngineError("settlement request session has no decision cost structure")
        object.__setattr__(self, "cost_structure", cost_structure)
        _require_identifier(self.stockout_rule, name="stock-out transition rule")
        _require_identifier(self.actuals_semantics, name="actuals semantics")


@dataclass(frozen=True, slots=True, init=False)
class OriginRequest:
    """Declare every input needed to compute one origin."""

    session: SessionIdentity
    origin: pd.Timestamp
    horizon: int
    scope: Scope
    calibration_partitions: tuple[str, ...]
    cost_structure: CostStructure | None
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
        horizon, encoded_config, cost_structure = _session_origin_inputs(session)
        partitions = _validated_partitions(calibration_partitions)
        if future_exogenous is not None and not isinstance(future_exogenous, pd.DataFrame):
            raise TypeError("future exogenous input must be a pandas DataFrame")
        positions = _validated_inventory_positions(inventory_positions or {})
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "calibration_partitions", partitions)
        object.__setattr__(self, "cost_structure", cost_structure)
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
type Orderer = Callable[[OrderRequest], Sequence[OrderRow]]
type Settler = Callable[[SettlementRequest], Sequence[SettlementRecord]]
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
        settler: Settler | None = None,
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
            (settler, "settler"),
        )
        for hook, name in callables:
            if hook is not None and not callable(hook):
                raise TypeError(f"engine {name} must be callable")
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
        self._settler = settler
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
            raise EngineError("predict requires at least one fitted task")
        for fitted in tasks:
            if not isinstance(fitted, FittedTask):
                raise TypeError("predict requires FittedTask values")
            self._require_session(fitted.session)
        frames = self._dispatch_backend.map(self._predict_one, tasks)
        combined = pd.concat(frames, ignore_index=True)
        return ForecastBatch(combined, calendar=self._ledger_sink.calendar)

    def reconcile(self, forecasts: ForecastBatch) -> ForecastBatch:
        """Apply the configured reconciler, or return the input identity."""
        if not isinstance(forecasts, ForecastBatch):
            raise TypeError("reconcile requires a ForecastBatch")
        if self._reconciler is None:
            return forecasts
        reconciled = self._reconciler(forecasts)
        if not isinstance(reconciled, ForecastBatch):
            raise EngineError("reconciler must return a ForecastBatch")
        return reconciled

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
        requested = _validated_partitions(partitions)
        observed_updates = {} if observation is None else dict(observation.state_updates)
        _reject_undeclared_state_updates(observed_updates, requested, producer="observer")
        if self._calibrator is None:
            return CalibrationResult(forecasts, observed_updates)
        if observation is not None and set(observation.prior_states) == set(requested):
            states = dict(observation.prior_states)
        else:
            states = {
                partition: self._calibration_state_store.load(session, partition)
                for partition in requested
            }
        states.update(observed_updates)
        result = self._calibrator(forecasts, states)
        if not isinstance(result, CalibrationResult):
            raise EngineError("calibrator must return a CalibrationResult")
        _reject_undeclared_state_updates(
            result.state_updates,
            requested,
            producer="calibrator",
        )
        merged_updates = observed_updates
        merged_updates.update(result.state_updates)
        return CalibrationResult(result.forecasts, merged_updates)

    def order(self, request: OrderRequest) -> tuple[OrderRow, ...]:
        """Apply the configured orderer, or emit no orders."""
        if not isinstance(request, OrderRequest):
            raise TypeError("order requires an OrderRequest")
        self._require_session(request.session)
        if self._orderer is None:
            return ()
        orders = tuple(self._orderer(request))
        if any(not isinstance(order, OrderRow) for order in orders):
            raise EngineError("orderer must return only OrderRow values")
        return orders

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
            return ObservationResult(resolutions)
        states = MappingProxyType(
            {
                partition: self._calibration_state_store.load(session, partition)
                for partition in requested
            }
        )
        updates = self._observer(due, MappingProxyType(resolutions), states)
        result = ObservationResult(resolutions, updates, states)
        _reject_undeclared_state_updates(
            result.state_updates,
            requested,
            producer="observer",
        )
        return result

    def settle(self, request: SettlementRequest) -> tuple[SettlementRecord, ...]:
        """Invoke the configured settlement hook, or emit no records."""
        if not isinstance(request, SettlementRequest):
            raise TypeError("settle requires a SettlementRequest")
        self._require_session(request.session)
        if self._settler is None:
            return ()
        settlements = tuple(self._settler(request))
        if any(not isinstance(record, SettlementRecord) for record in settlements):
            raise EngineError("settler must return only SettlementRecord values")
        return settlements

    def commit(self, request: OriginCommit | CommitReceipt) -> CommitReceipt:
        """Persist an origin's ledger and monotone calibration-state mutations."""
        if not isinstance(request, (OriginCommit, CommitReceipt)):
            raise TypeError("commit requires an OriginCommit or CommitReceipt")
        self._require_session(request.session)
        if isinstance(request, OriginCommit):
            expected = CommitReceipt.from_commit(request)
            receipt = self._ledger_sink.commit(request)
            if receipt != expected:
                raise EngineError("ledger sink returned a mismatched commit receipt")
        else:
            receipt = self._ledger_sink.receipt(request.origin)
            if receipt is None:
                raise EngineError("commit receipt is not journaled by the ledger sink")
            if receipt != request:
                raise EngineError("commit receipt does not match the ledger journal")
        for partition, value in receipt.state_updates.items():
            self._calibration_state_store.save(
                receipt.session,
                partition,
                value,
                origin=receipt.origin,
            )
        return receipt

    def _require_session(self, session: SessionIdentity) -> None:
        if session != self._ledger_sink.session:
            raise EngineError("engine session does not match its ledger sink")

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
                raise EngineError("fitted model artifact is missing")
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

    def run_origin(self, request: OriginRequest) -> OriginResult:
        """Run Resolve through Commit once for a declared origin."""
        if not isinstance(request, OriginRequest):
            raise TypeError("spine requires an OriginRequest")
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
        orders = self._phase(
            Phase.ORDER,
            request.origin,
            lambda: self._engine.order(order_request),
        )
        commit_request = OriginCommit(
            session=request.session,
            origin=request.origin,
            resolutions=observation.resolutions,
            forecasts=(
                ForecastWrite(
                    calibrated.forecasts.frame,
                    calibrated.forecasts.issuances,
                ),
            ),
            orders=orders,
            state_updates=calibrated.state_updates,
        )
        self._phase(
            Phase.COMMIT,
            request.origin,
            lambda: self._engine.commit(commit_request),
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


def _session_definition(session: SessionIdentity) -> dict[str, object]:
    try:
        definition = json.loads(session.to_bytes())
    except (TypeError, ValueError) as error:
        raise EngineError("session identity has an invalid canonical definition") from error
    if not isinstance(definition, dict):
        raise EngineError("session identity definition must be an object")
    return definition


def _session_origin_inputs(
    session: SessionIdentity,
) -> tuple[int, bytes, CostStructure | None]:
    definition = _session_definition(session)
    horizon = definition.get("horizon")
    model_config = definition.get("model_config")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise EngineError("session identity has an invalid horizon")
    if not isinstance(model_config, Mapping):
        raise EngineError("session identity has an invalid model configuration")
    encoded_config = canonical_json_bytes(
        dict(model_config),
        path="session model configuration",
    )
    return horizon, encoded_config, _cost_from_definition(definition)


def _session_cost_structure(session: SessionIdentity) -> CostStructure | None:
    return _cost_from_definition(_session_definition(session))


def _cost_from_definition(definition: Mapping[str, object]) -> CostStructure | None:
    decision = definition.get("decision")
    if decision is None:
        return None
    if not isinstance(decision, Mapping):
        raise EngineError("session identity has an invalid decision configuration")
    cost = decision.get("cost_structure")
    if not isinstance(cost, Mapping):
        raise EngineError("session identity has an invalid cost structure")
    cost_values = dict(cost)
    try:
        return CostStructure(
            underage=cast(float, cost_values["underage_cost"]),
            overage=cast(float, cost_values["overage_cost"]),
            holding=cast(float, cost_values["holding_cost"]),
            shortage=cast(float, cost_values["shortage_cost"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EngineError("session identity has an invalid cost structure") from error


def _require_panel_session_binding(
    panel: Panel,
    *,
    session: SessionIdentity,
    ledger_calendar: Calendar,
) -> None:
    definition = _session_definition(session)
    series_set = definition.get("series_set")
    frequency = definition.get("calendar_frequency")
    if not isinstance(series_set, list) or any(
        not isinstance(series_key, str) for series_key in series_set
    ):
        raise EngineError("session identity has an invalid series set")
    if tuple(series_set) != panel.series_keys:
        raise EngineError("panel series set does not match the engine session")
    if frequency != ledger_calendar.frequency:
        raise EngineError("ledger calendar does not match the engine session")
    if panel.calendar != ledger_calendar:
        raise EngineError("panel calendar does not match the ledger calendar")


def _require_task_session_binding(
    task: ForecastTask,
    *,
    session: SessionIdentity,
) -> None:
    definition = _session_definition(session)
    expected_horizon = definition.get("horizon")
    expected_model = definition.get("model_config")
    expected_series = definition.get("series_set")
    expected_frequency = definition.get("calendar_frequency")
    if task.horizon != expected_horizon or task.model_config != expected_model:
        raise EngineError("fitted task configuration does not match its session")
    if not isinstance(expected_series, list) or not set(task.series_keys) <= set(expected_series):
        raise EngineError("fitted task series do not belong to its session")
    if task.scope is Scope.GLOBAL and tuple(expected_series) != task.series_keys:
        raise EngineError("global fitted task must cover its session series set")
    if task.calendar.frequency != expected_frequency:
        raise EngineError("fitted task calendar does not match its session")


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


def _validated_partitions(partitions: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(partitions)
    if len(set(requested)) != len(requested):
        raise EngineError("calibration partitions must be unique")
    for partition in requested:
        _require_identifier(partition, name="calibration partition")
    return requested


def _validated_state_updates(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise TypeError("calibration state updates must be a mapping")
    updates: dict[str, bytes] = {}
    for partition, state in value.items():
        _require_identifier(partition, name="calibration partition")
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
        _require_identifier(partition, name="calibration partition")
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
        raise EngineError(f"{producer} updated undeclared partitions: {sorted(unknown)!r}")


def _validated_inventory_positions(
    value: Mapping[str, InventoryPosition],
) -> dict[str, InventoryPosition]:
    if not isinstance(value, Mapping):
        raise TypeError("inventory positions must be a mapping")
    positions: dict[str, InventoryPosition] = {}
    for series_key, position in value.items():
        _require_identifier(series_key, name="inventory-position series key")
        if not isinstance(position, InventoryPosition):
            raise TypeError("inventory positions must contain InventoryPosition values")
        positions[series_key] = position
    return positions


def _validated_actuals(value: Mapping[ActualKey, float]) -> dict[ActualKey, float]:
    if not isinstance(value, Mapping):
        raise TypeError("settlement request actuals must be a mapping")
    actuals: dict[ActualKey, float] = {}
    for key, raw_actual in value.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not isinstance(key[1], pd.Timestamp)
        ):
            raise TypeError("settlement actual keys must be (series key, Timestamp) pairs")
        _require_identifier(key[0], name="settlement actual series key")
        if isinstance(raw_actual, bool) or not isinstance(raw_actual, Real):
            raise TypeError("settlement actuals must be real numbers")
        actual = float(raw_actual)
        if not math.isfinite(actual):
            raise EngineError("settlement actuals must be finite")
        actuals[key] = actual
    return actuals


def _require_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EngineError(f"{name} must be a non-empty trimmed string")


__all__ = [
    "ENGINE_VERBS",
    "CalibrationResult",
    "Engine",
    "EngineError",
    "FittedTask",
    "ForecastBatch",
    "ObservationResult",
    "OrderRequest",
    "OriginRequest",
    "OriginResult",
    "Phase",
    "PhaseError",
    "PhaseEvent",
    "SettlementRequest",
    "Spine",
]
