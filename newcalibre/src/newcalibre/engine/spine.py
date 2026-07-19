"""Run the fixed chapter-03 engine spine over six abstract ports."""

from __future__ import annotations

import hashlib
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
    CalibrationContext,
    ConformalRuntime,
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
    SERIES_KEY,
    ActualsSemantics,
    Calendar,
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
    _session: SessionIdentity | None = field(repr=False)
    _engine_token: object | None = field(repr=False, compare=False)
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
        object.__setattr__(self, "_session", None)
        object.__setattr__(self, "_engine_token", None)
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
    """Authenticate one immutable staged observe cycle and its prior state."""

    cycle: ObserveCycle
    prior_states: Mapping[str, bytes] = field(default_factory=dict)
    session: SessionIdentity | None = None
    origin: pd.Timestamp | None = None
    _engine_token: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, ObserveCycle):
            raise TypeError("observation result cycle must be an ObserveCycle")
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
    input_fingerprint: str | None = None

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
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str)
            or len(self.input_fingerprint) != 64
            or self.input_fingerprint != self.input_fingerprint.lower()
            or any(character not in "0123456789abcdef" for character in self.input_fingerprint)
        ):
            raise ValueError("commit request input fingerprint must be a SHA-256 hex string")
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
type CustomOrderer = Callable[[OrderRequest], Iterable[OrderProposal]]
type Orderer = CustomOrderer | ConfiguredPolicyOrderer
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
        hierarchy: HierarchyIndex | None,
        adapter_resolver: AdapterResolver = resolve_adapter,
        reconciliation_strategy: str = "none",
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
        if hierarchy is not None and not isinstance(hierarchy, HierarchyIndex):
            raise TypeError("engine hierarchy must be a HierarchyIndex or None")
        if not callable(adapter_resolver):
            raise TypeError("engine adapter resolver must be callable")
        if orderer is not None and not (
            callable(orderer) or isinstance(orderer, ConfiguredPolicyOrderer)
        ):
            raise TypeError("engine orderer must be callable or a ConfiguredPolicyOrderer")
        # Resolve configuration before the panel port can perform any data load.
        adapter_resolver(_session_model_config(ledger_sink.session))
        reconciler = resolve_strategy(reconciliation_strategy)
        ordering_configuration = _session_ordering_configuration(ledger_sink.session)
        state_snapshot = calibration_state_store.snapshot(ledger_sink.session)
        conformal_configuration = _session_conformal_config(ledger_sink.session)
        runtime = (
            None
            if conformal_configuration is None
            else resolve_method(conformal_configuration, states=state_snapshot)
        )
        self._ordering_configuration = ordering_configuration
        self._runtime: ConformalRuntime | None = runtime
        self._reconciliation_hierarchy = hierarchy
        self._panel = panel_source.load()
        self._hierarchy = hierarchy or HierarchyIndex.flat(self._panel.series_keys)
        self._actuals_source = actuals_source
        self._artifact_store = artifact_store
        self._calibration_state_store = calibration_state_store
        self._ledger_sink = ledger_sink
        self._dispatch_backend = dispatch_backend
        self._adapter_resolver = adapter_resolver
        self._reconciler = reconciler
        self._orderer = orderer
        self._phase_token = object()
        _require_panel_session_binding(
            self._panel,
            session=self._ledger_sink.session,
            ledger_calendar=self._ledger_sink.calendar,
        )
        if self._hierarchy.bottom_series != self._panel.series_keys:
            raise _EngineError("hierarchy bottom series do not match the engine session panel")

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
        """Apply the configured registered point-reconciliation strategy."""
        if not isinstance(forecasts, ForecastBatch):
            raise TypeError("reconcile requires a ForecastBatch")
        self._require_forecast_batch(forecasts)
        result = self._reconciler(
            forecasts.frame,
            self._reconciliation_hierarchy,
            ReconciliationContext(),
        )
        if not isinstance(result, pd.DataFrame):
            raise _EngineError("reconciliation strategy must return a pandas DataFrame")
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
        if observation._engine_token is not self._phase_token:
            raise _EngineError("calibration observation was not produced by this engine")

        observed_updates = dict(observation.cycle.state_updates)
        states = dict(observation.prior_states)
        states.update(observed_updates)
        if self._runtime is None:
            return _bind_calibration_result(
                forecasts,
                observed_updates,
                session=session,
                origin=origin,
                engine_token=self._phase_token,
            )

        context = _calibration_context(
            self._runtime,
            forecasts._frame,
            hierarchy=self._hierarchy,
        )
        try:
            runtime_result = self._runtime.apply(
                forecasts.frame,
                MappingProxyType(states),
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
            session=session,
            engine_token=self._phase_token,
        )
        if _forecast_batch_origin(calibrated) != origin:
            raise _EngineError("conformal apply changed the forecast origin")

        apply_updates = _validated_state_updates(runtime_result.state_updates)
        conflicts = {
            label
            for label in observed_updates.keys() & apply_updates.keys()
            if observed_updates[label] != apply_updates[label]
        }
        if conflicts:
            raise _EngineError(
                f"observe and apply emitted conflicting states: {sorted(conflicts)!r}"
            )
        merged_updates = dict(observed_updates)
        merged_updates.update(apply_updates)
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
        submission: ActualsSubmission | None = None,
    ) -> ObservationResult:
        """Stage one complete durable-snapshot observe cycle without persistence."""
        self._require_session(session)
        if submission is not None and not isinstance(submission, ActualsSubmission):
            raise TypeError("observe submission must be an ActualsSubmission or None")
        prior_states = dict(self._calibration_state_store.snapshot(session))
        loop = ObserveLoop(
            hierarchy=self._hierarchy,
            observed_history=self._ledger_sink.observed_history,
            pending_observations=self._ledger_sink.pending_observations,
            conformal_states=prior_states,
            runtime=self._runtime,
        )
        accepted = self._actuals_source.reveal(before=origin) if submission is None else submission
        loop.accept(accepted)
        cycle = loop.cycle(origin)
        return _bind_observation_result(
            cycle,
            prior_states,
            session=session,
            origin=origin,
            engine_token=self._phase_token,
        )

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
            receipt = _snapshot_commit_receipt(self._ledger_sink.commit(request))
            expected = CommitReceipt.from_commit(request, sequence=receipt.sequence)
            if receipt != expected:
                raise _EngineError("ledger sink returned a mismatched commit receipt")
        else:
            callback_receipt = self._ledger_sink.receipt(request.commit_key)
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
                sequence=receipt.sequence,
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
                input_fingerprint=request.input_fingerprint,
                inventory_positions=request.inventory_positions,
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

    def _require_event_driver_port(self, *, ledger_sink: LedgerSink) -> None:
        """Reject a driver wired to a different ledger than this engine."""
        if ledger_sink is not self._ledger_sink:
            raise _EngineError("event driver ledger sink does not belong to the engine")

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
        submission: ActualsSubmission | None = None,
        input_fingerprint: str | None = None,
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
        if submission is not None and not isinstance(submission, ActualsSubmission):
            raise TypeError("spine submission must be an ActualsSubmission or None")
        observation = self._phase(
            Phase.RESOLVE,
            request.origin,
            lambda: self._engine.observe(
                request.origin,
                session=request.session,
                submission=submission,
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
                    observation=observation,
                    calibration=calibrated,
                    inventory_positions=request.inventory_positions,
                    decisions=decisions,
                    settlement=settlement,
                    input_fingerprint=input_fingerprint,
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
        observation_issuances=forecasts.observation_issuances,
    )
    object.__setattr__(bound, "_session", session)
    object.__setattr__(bound, "_engine_token", engine_token)
    return bound


def _snapshot_commit_receipt(receipt: object) -> CommitReceipt:
    """Collapse a ledger callback result into one stable, exact receipt."""
    if not isinstance(receipt, CommitReceipt):
        raise _EngineError("ledger sink must return a CommitReceipt")
    return CommitReceipt(
        session=receipt.session,
        origin=receipt.origin,
        digest=receipt.digest,
        state_updates=dict(receipt.state_updates),
        observe_cycle=receipt.observe_cycle,
        settlement_periods=tuple(receipt.settlement_periods),
        sequence=receipt.sequence,
        actual_keys=tuple(receipt.actual_keys),
        input_fingerprint=receipt.input_fingerprint,
        orders=tuple(receipt.orders),
        inventory_positions=dict(receipt.inventory_positions),
    )


def _bind_observation_result(
    cycle: ObserveCycle,
    prior_states: Mapping[str, bytes],
    *,
    session: SessionIdentity,
    origin: pd.Timestamp,
    engine_token: object,
) -> ObservationResult:
    result = ObservationResult(
        cycle,
        prior_states,
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
