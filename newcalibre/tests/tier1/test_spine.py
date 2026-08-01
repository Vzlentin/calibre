"""Exercise the fixed engine spine through its exact public verb surface."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionScope,
    DecisionScopeKind,
    DecisionTiming,
    EmissionScope,
    ForecastFrameError,
    ForecastTask,
    GuaranteeClaim,
    GuaranteeDescriptor,
    GuaranteeType,
    HierarchyIndex,
    InventoryPosition,
    Panel,
    Scope,
    ScoredSeries,
    SessionIdentity,
    StockoutRule,
    TargetSupport,
    interval_columns,
    quantile_column,
    target_timestamp,
)
from newcalibre.engine import (
    ENGINE_VERBS,
    Engine,
    EngineError,
    ForecastBatch,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    OrderProposal,
    OrderRequest,
    OriginCommit,
    OriginIntent,
    OriginRequest,
    OriginSnapshot,
    Phase,
    PhaseError,
    PhaseEvent,
    SettlementSnapshot,
    SettlementWindow,
    Spine,
)
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError
from newcalibre.ledger import ForecastIssuance
from newcalibre.ordering import OrderingConfigError
from newcalibre.reconcile import ReconciliationRegistryError

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
MODEL_CONFIG = {"backend": "fixture", "name": "fixture"}
COST_STRUCTURE = CostStructure(1.0, 1.0, 1.0, 1.0)
ORDERING_POLICY = {"name": "newsvendor"}
TIMING = DecisionTiming(lead_time=1, review_period=1)


def _panel(
    *,
    series_keys: tuple[str, ...] = ("a",),
    target_support: TargetSupport = TargetSupport.REAL,
) -> Panel:
    timestamps = pd.date_range("2026-01-01", periods=7, freq="D")
    return Panel.from_frame(
        pd.DataFrame.from_records(
            [
                {
                    SERIES_KEY: key,
                    TIMESTAMP: timestamp,
                    OBSERVED_VALUE: float(index),
                }
                for key in series_keys
                for index, timestamp in enumerate(timestamps, start=1)
            ]
        ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"}),
        calendar=CALENDAR,
        target_support=target_support,
    )


def _session(
    *,
    tenant: str = "tenant-a",
    model_config: Mapping[str, object] = MODEL_CONFIG,
    horizon: int | None = None,
    with_decision: bool = False,
    series_keys: tuple[str, ...] = ("a",),
    conformal_config: Mapping[str, object] | None = None,
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=(TIMING.protection_period if with_decision else 1) if horizon is None else horizon,
        model_config=model_config,
        conformal_config=conformal_config,
        ordering_policy=ORDERING_POLICY if with_decision else None,
        decision_series_keys=series_keys if with_decision else None,
        cost_structure=COST_STRUCTURE if with_decision else None,
        decision_timing=TIMING if with_decision else None,
        stockout_rule=StockoutRule.LOST_SALES if with_decision else None,
    )


class PersistentFixtureAdapter:
    """Persist one last-value point and expose lifecycle events."""

    def __init__(self, events: list[str], *, fail_prediction: bool = False) -> None:
        self._events = events
        self._fail_prediction = fail_prediction
        self._point: float | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset(
            {AdapterCapability.ARTIFACT_PERSISTENCE, AdapterCapability.INCREMENTAL_UPDATE}
        )

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask) -> None:
        self._events.append("fit")
        self._point = float(task.history.materialize()[OBSERVED_VALUE].iloc[-1])

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        self._events.append("predict")
        if self._fail_prediction:
            raise RuntimeError("fixture prediction failed")
        assert self._point is not None
        frame = pd.DataFrame.from_records(
            [
                {
                    SERIES_KEY: key,
                    TARGET_TIMESTAMP: target_timestamp(
                        task.origin,
                        step,
                        calendar=task.calendar,
                    ),
                    ACTUAL_VALUE: float("nan"),
                    POINT_FORECAST: self._point,
                    HORIZON_STEP: step,
                    ORIGIN: task.origin,
                    MODEL_NAME: "fixture",
                }
                for key in task.series_keys
                for step in range(1, task.horizon + 1)
            ]
        )
        return frame.astype(
            {
                SERIES_KEY: "string",
                ACTUAL_VALUE: "float64",
                POINT_FORECAST: "float64",
                HORIZON_STEP: "int64",
                MODEL_NAME: "string",
            }
        )

    def dump_state(self) -> bytes:
        assert self._point is not None
        return repr(self._point).encode()

    def load_state(self, state: bytes) -> None:
        self._events.append("load")
        self._point = float(state.decode())

    def fitted_values(self):
        raise AdapterCapabilityError("fixture has no fitted values")

    def update(self, delta) -> None:
        self._events.append("update")
        materialized = delta.materialize()
        if not materialized.empty:
            self._point = float(materialized[OBSERVED_VALUE].iloc[-1])


class DataFrameSubclassFixtureAdapter(PersistentFixtureAdapter):
    """Return a pandas subclass to exercise the forecast value boundary."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        class CallbackFrame(pd.DataFrame):
            @property
            def _constructor(self):
                return CallbackFrame

        return CallbackFrame(super().predict(task))


class RecordingDispatch:
    """Record each deterministic dispatch batch size."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def map(
        self,
        function: Callable[[object], object],
        items: Sequence[object],
    ) -> tuple[object, ...]:
        self.batch_sizes.append(len(items))
        return tuple(function(item) for item in items)


class RecordingPanelSource(InMemoryPanelSource):
    """Count immutable panel loads across engine construction."""

    def __init__(self, panel: Panel) -> None:
        super().__init__(panel)
        self.loads = 0

    def load(self) -> Panel:
        self.loads += 1
        return super().load()


def _store(session: SessionIdentity, panel: Panel) -> InMemoryIndexedRunStore:
    return InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )


def _engine(
    *,
    session: SessionIdentity,
    panel: Panel,
    store: InMemoryIndexedRunStore,
    events: list[str] | None = None,
    dispatch: RecordingDispatch | None = None,
    reconciliation_strategy: str = "none",
    orderer=None,
    panel_source: InMemoryPanelSource | None = None,
    adapter_resolver=None,
    hierarchy: HierarchyIndex | None = None,
) -> Engine:
    event_log = [] if events is None else events
    return Engine(
        session=session,
        panel_source=panel_source or InMemoryPanelSource(panel),
        run_store=store,
        dispatch_backend=dispatch or RecordingDispatch(),
        hierarchy=hierarchy or HierarchyIndex.flat(panel.series_keys),
        adapter_resolver=(
            (lambda _config: PersistentFixtureAdapter(event_log))
            if adapter_resolver is None
            else adapter_resolver
        ),
        reconciliation_strategy=reconciliation_strategy,
        orderer=orderer,
    )


def _snapshot(
    store: InMemoryIndexedRunStore,
    session: SessionIdentity,
    origin: pd.Timestamp,
) -> OriginSnapshot:
    snapshot = store.open(OriginIntent(session, origin))
    assert isinstance(snapshot, OriginSnapshot)
    return snapshot


def _run_origin(
    engine: Engine,
    store: InMemoryIndexedRunStore,
    request: OriginRequest,
    *,
    reporter=None,
):
    return Spine(engine, reporter=reporter).run_origin(
        request,
        snapshot=_snapshot(store, request.session, request.origin),
    )


def _predict(
    engine: Engine,
    store: InMemoryIndexedRunStore,
    session: SessionIdentity,
    origin: pd.Timestamp,
) -> ForecastBatch:
    engine.observe(origin, session=session, snapshot=_snapshot(store, session, origin))
    return engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )


def test_adapter_capabilities_refuse_before_panel_load() -> None:
    """Resolve adapter contracts before invoking panel I/O."""
    panel = _panel()
    source = RecordingPanelSource(panel)
    session = _session(model_config={"backend": "seasonal-naive", "m": 1, "quantile_levels": [0.5]})
    with pytest.raises(AdapterCapabilityError, match="native_quantiles"):
        Engine(
            session=session,
            panel_source=source,
            run_store=_store(session, panel),
            dispatch_backend=RecordingDispatch(),
            hierarchy=HierarchyIndex.flat(panel.series_keys),
        )
    assert source.loads == 0


def test_ordering_configuration_refuses_before_panel_load() -> None:
    """Resolve ordering configuration before invoking panel I/O."""
    panel = _panel()
    source = RecordingPanelSource(panel)
    session = SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("a",),
        calendar=CALENDAR,
        horizon=TIMING.protection_period,
        model_config={"backend": "seasonal-naive", "m": 1},
        ordering_policy={"name": "rs"},
        decision_series_keys=("a",),
        cost_structure=COST_STRUCTURE,
        decision_timing=TIMING,
        stockout_rule=StockoutRule.LOST_SALES,
    )
    with pytest.raises(OrderingConfigError, match="requires conformal coverage"):
        Engine(
            session=session,
            panel_source=source,
            run_store=_store(session, panel),
            dispatch_backend=RecordingDispatch(),
            hierarchy=HierarchyIndex.flat(panel.series_keys),
        )
    assert source.loads == 0


def test_unknown_reconciliation_strategy_refuses_before_panel_load() -> None:
    """Resolve reconciliation registration before invoking panel I/O."""
    panel = _panel()
    source = RecordingPanelSource(panel)
    session = _session()
    with pytest.raises(ReconciliationRegistryError):
        _engine(
            session=session,
            panel=panel,
            store=_store(session, panel),
            panel_source=source,
            reconciliation_strategy="unknown",
        )
    assert source.loads == 0


def test_engine_supplies_panel_target_support_to_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carry target support through the registered reconciliation seam."""
    seen: list[TargetSupport] = []

    class RecordingReconciler:
        def __call__(self, frame, hierarchy, context):
            del hierarchy
            seen.append(context.target_support)
            return frame

    monkeypatch.setattr(
        "newcalibre.engine.spine.resolve_strategy",
        lambda _name: RecordingReconciler(),
    )
    panel = _panel(target_support=TargetSupport.NONNEGATIVE)
    session = _session()
    store = _store(session, panel)
    engine = _engine(
        session=session,
        panel=panel,
        store=store,
        reconciliation_strategy="recording",
    )
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    engine.observe(origin, session=session, snapshot=_snapshot(store, session, origin))
    engine.reconcile(engine.predict(engine.fit(request)))
    assert seen == [TargetSupport.NONNEGATIVE]


def test_spine_runs_fixed_phases_and_publishes_one_atomic_transaction() -> None:
    """Run the six orchestration phases and publish all durable effects together."""
    panel = _panel()
    session = _session(
        with_decision=True,
        conformal_config={
            "method": "split-per-step",
            "coverage": 0.5,
            "calibration_window": 20,
        },
    )
    store = _store(session, panel)
    events: list[str] = []
    dispatch = RecordingDispatch()

    def order(request: OrderRequest) -> tuple[OrderProposal, ...]:
        events.append("order")
        assert request.inventory_positions["a"].value == 0.0
        assert request.costs_by_series == {"a": COST_STRUCTURE}
        return (OrderProposal("a", "fixture", 1.0),)

    engine = _engine(
        session=session,
        panel=panel,
        store=store,
        events=events,
        dispatch=dispatch,
        orderer=order,
    )
    phase_events: list[PhaseEvent] = []
    origin = pd.Timestamp("2026-01-05")
    result = _run_origin(
        engine,
        store,
        OriginRequest(
            session=session,
            origin=origin,
            scope=Scope.LOCAL,
            inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
        ),
        reporter=phase_events.append,
    )

    assert [event.phase for event in phase_events] == [
        Phase.RESOLVE,
        Phase.PREDICT,
        Phase.RECONCILE,
        Phase.CALIBRATE,
        Phase.ORDER,
        Phase.COMMIT,
    ]
    assert events == ["fit", "predict", "order"]
    assert dispatch.batch_sizes == [1]
    assert store.forecasts
    assert store.orders
    assert store.states
    assert store.checkpoints
    assert store.checkpoint_indexes
    assert result.receipt.revision == 2
    assert store.receipt(origin) == result.receipt


def test_unconfigured_reconcile_calibrate_and_order_are_identities() -> None:
    """Keep unconfigured stages explicit without changing forecast facts."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    observation = engine.observe(
        origin,
        session=session,
        snapshot=_snapshot(store, session, origin),
    )
    predicted = engine.predict(engine.fit(request))
    reconciled = engine.reconcile(predicted)
    calibrated = engine.calibrate(reconciled, session=session, observation=observation)

    pd.testing.assert_frame_equal(reconciled.frame, predicted.frame)
    pd.testing.assert_frame_equal(calibrated.forecasts.frame, predicted.frame)
    assert (
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=calibrated.forecasts,
                inventory_positions={},
            )
        )
        is None
    )


def test_predict_collapses_dataframe_subclasses_before_phase_validation() -> None:
    """Own adapter frames before applying the engine forecast contract."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(
        session=session,
        panel=panel,
        store=store,
        adapter_resolver=lambda _config: DataFrameSubclassFixtureAdapter([]),
    )

    assert type(_predict(engine, store, session, pd.Timestamp("2026-01-05")).frame) is pd.DataFrame


def test_forecast_batch_collapses_dataframe_subclasses_before_validation() -> None:
    """Prevent dataframe subclasses from masking malformed forecast columns."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)
    malformed = _predict(
        engine,
        store,
        session,
        pd.Timestamp("2026-01-05"),
    ).frame
    malformed.loc[malformed.index[-1], HORIZON_STEP] = 2

    class MaskingFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return MaskingFrame

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if key == HORIZON_STEP:
                masked = value.copy(deep=True)
                masked.iloc[-1] = 1
                return masked
            if isinstance(key, list) and HORIZON_STEP in key:
                masked = pd.DataFrame(value, copy=True)
                masked.loc[masked.index[-1], HORIZON_STEP] = 1
                return masked
            return value

    with pytest.raises(ForecastFrameError, match="target timestamp must equal"):
        ForecastBatch(MaskingFrame(malformed), calendar=CALENDAR)


def test_forecast_batch_materializes_empty_native_quantile_issuances() -> None:
    """Represent unguaranteed native quantiles with explicit empty row maps."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    predicted = _predict(
        _engine(session=session, panel=panel, store=store),
        store,
        session,
        pd.Timestamp("2026-01-05"),
    )
    frame = predicted.frame
    quantile = quantile_column(0.5)
    frame[quantile] = frame[POINT_FORECAST] + 1.0

    forecasts = ForecastBatch(frame, calendar=CALENDAR)

    assert forecasts.frame[quantile].tolist() == [5.0]
    assert all(not values for values in forecasts.issuances.values())


def test_forecast_batch_rejects_partial_native_quantile_issuance() -> None:
    """Require a native quantile guarantee to cover every forecast row."""
    series_keys = ("a", "b")
    panel = _panel(series_keys=series_keys)
    session = _session(series_keys=series_keys)
    store = _store(session, panel)
    origin = pd.Timestamp("2026-01-05")
    predicted = _predict(
        _engine(session=session, panel=panel, store=store),
        store,
        session,
        origin,
    )
    frame = predicted.frame
    quantile = quantile_column(0.5)
    frame[quantile] = frame[POINT_FORECAST]
    keys = tuple(predicted.issuances)
    issuances = {key: {} for key in keys}
    descriptor = GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.NONE,
            currency=None,
            declared_slack=None,
        ),
        level=0.5,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
    )
    issuances[keys[0]] = {
        (quantile,): ForecastIssuance(
            descriptor=descriptor,
            guaranteed_side=None,
            calibration_ready=False,
            bounds_finite=True,
            bounds_null_reason=None,
        )
    }

    with pytest.raises(EngineError, match="quantile issuance.*every row or no rows"):
        ForecastBatch(frame, calendar=CALENDAR, issuances=issuances)


def test_forecast_batch_rejects_interval_without_issuance_evidence() -> None:
    """Reject interval columns that have no explicit guarantee metadata."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    predicted = _predict(
        _engine(session=session, panel=panel, store=store),
        store,
        session,
        pd.Timestamp("2026-01-05"),
    )
    frame = predicted.frame
    lower, upper = interval_columns(0.8)
    frame[lower] = frame[POINT_FORECAST] - 1.0
    frame[upper] = frame[POINT_FORECAST] + 1.0

    with pytest.raises(EngineError, match="interval.*issuance"):
        ForecastBatch(frame, calendar=CALENDAR, issuances=predicted.issuances)


def test_settlement_window_requires_explicit_actuals_semantics() -> None:
    """Reject settlement input whose observation interpretation is implicit."""
    origin = pd.Timestamp("2026-01-05")
    snapshot = SettlementSnapshot(
        session=_session(with_decision=True),
        calendar=CALENDAR,
        periods=(origin,),
        frontier=None,
        latest_positions={},
        open_order_quantities={"a": 0.0},
        due_arrivals={},
        actuals_semantics=None,
    )

    with pytest.raises(TypeError, match="actuals_semantics"):
        SettlementWindow(
            snapshot=snapshot,
            actuals={("a", origin): 0.0},
        )  # type: ignore[call-arg]


def test_order_request_rejects_forecasts_from_another_origin() -> None:
    """Bind every decision row to the request's one origin."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    origin = pd.Timestamp("2026-01-05")
    forecasts = _predict(
        _engine(session=session, panel=panel, store=store),
        store,
        session,
        origin,
    )
    foreign = forecasts.frame
    foreign[ORIGIN] = foreign[ORIGIN] + pd.Timedelta(days=1)
    foreign[TARGET_TIMESTAMP] = foreign[TARGET_TIMESTAMP] + pd.Timedelta(days=1)

    for frame in (foreign, pd.concat([forecasts.frame, foreign], ignore_index=True)):
        with pytest.raises(EngineError, match="all match its origin"):
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=ForecastBatch(frame, calendar=CALENDAR),
            )


def test_engine_constructor_has_no_provisional_reconciler_callback() -> None:
    """Keep reconciliation selection behind the registered strategy name."""
    parameters = inspect.signature(Engine).parameters

    assert "reconciler" not in parameters
    assert parameters["reconciliation_strategy"].default == "none"


def test_cycle_token_uses_store_revision_and_rejects_stale_fitted_work() -> None:
    """Retire work prepared from a prior store revision when a new snapshot opens."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    first_snapshot = _snapshot(store, session, origin)
    observation = engine.observe(origin, session=session, snapshot=first_snapshot)
    fitted = engine.fit(request)
    assert observation.token is not None
    assert observation.token.revision == first_snapshot.revision

    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-04"),
            expected_revision=store.revision,
        )
    )
    engine.observe(origin, session=session, snapshot=_snapshot(store, session, origin))
    with pytest.raises(EngineError, match="stale"):
        engine.predict(fitted)


def test_reopening_the_same_revision_retires_prior_attempt_values() -> None:
    """Reject values staged by an aborted attempt even without a store commit."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    snapshot = _snapshot(store, session, origin)
    first = engine.observe(origin, session=session, snapshot=snapshot)
    fitted = engine.fit(request)

    second = engine.observe(origin, session=session, snapshot=snapshot)

    assert first.token is not None
    assert second.token is not None
    assert first.token.revision == second.token.revision
    assert first.token.attempt != second.token.attempt
    with pytest.raises(EngineError, match="stale"):
        engine.predict(fitted)


def test_prediction_failure_publishes_no_checkpoint_or_ledger_state() -> None:
    """Abort a failed phase without exposing any staged lifecycle effect."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(
        session=session,
        panel=panel,
        store=store,
        adapter_resolver=lambda _config: PersistentFixtureAdapter(
            [],
            fail_prediction=True,
        ),
    )
    origin = pd.Timestamp("2026-01-05")
    with pytest.raises(PhaseError, match="Predict.*fixture prediction failed"):
        _run_origin(
            engine,
            store,
            OriginRequest(session=session, origin=origin, scope=Scope.LOCAL),
        )

    assert store.revision == 1
    assert store.checkpoints == {}
    assert store.checkpoint_indexes == {}
    assert store.states == {}
    assert store.forecasts == ()
    assert store.receipt(origin) is None


def test_failed_phase_is_retryable_from_a_fresh_snapshot() -> None:
    """Reconstruct the engine after failure and commit exactly one origin."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    origin = pd.Timestamp("2026-01-05")
    failing = _engine(
        session=session,
        panel=panel,
        store=store,
        adapter_resolver=lambda _config: PersistentFixtureAdapter(
            [],
            fail_prediction=True,
        ),
    )
    with pytest.raises(PhaseError):
        _run_origin(
            failing,
            store,
            OriginRequest(session=session, origin=origin, scope=Scope.LOCAL),
        )

    retry = _engine(session=session, panel=panel, store=store)
    result = _run_origin(
        retry,
        store,
        OriginRequest(session=session, origin=origin, scope=Scope.LOCAL),
    )
    assert store.forecast_origin_count == 1
    assert store.receipt(origin) == result.receipt


def test_fit_results_are_process_safe_and_store_scoped() -> None:
    """Keep dispatched fitted tasks free of live adapters or store handles."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    engine.observe(origin, session=session, snapshot=_snapshot(store, session, origin))
    fitted = engine.fit(request)

    restored = pickle.loads(pickle.dumps(fitted))
    assert tuple(value.token for value in restored) == tuple(value.token for value in fitted)
    assert tuple(value.task.identity for value in restored) == tuple(
        value.task.identity for value in fitted
    )
    assert all(not hasattr(value, "adapter") for value in fitted)
    assert all(not hasattr(value, "store") for value in fitted)


def test_reporter_failure_does_not_mask_a_successful_phase() -> None:
    """Treat phase reporting as diagnostic and keep committed outcomes authoritative."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)

    def broken_reporter(_event: PhaseEvent) -> None:
        raise RuntimeError("reporter unavailable")

    origin = pd.Timestamp("2026-01-05")
    result = _run_origin(
        engine,
        store,
        OriginRequest(session=session, origin=origin, scope=Scope.LOCAL),
        reporter=broken_reporter,
    )
    assert store.receipt(origin) == result.receipt


def test_origin_request_owns_nested_inputs() -> None:
    """Snapshot caller-owned exogenous and inventory inputs."""
    session = _session()
    future = pd.DataFrame({"feature": [1.0]})
    positions = {"a": InventoryPosition(1.0, 0.0, 0.0)}
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
        future_exogenous=future,
        inventory_positions=positions,
    )
    future.loc[0, "feature"] = 9.0
    positions["a"] = InventoryPosition(9.0, 0.0, 0.0)

    assert request.future_exogenous is not None
    assert request.future_exogenous.loc[0, "feature"] == 1.0
    assert request.inventory_positions["a"].on_hand == 1.0


def test_engine_exposes_exactly_the_closed_eight_verbs() -> None:
    """Keep the public execution vocabulary unchanged by the store cutover."""
    public_methods = {
        name
        for name, value in inspect.getmembers(Engine, inspect.isfunction)
        if not name.startswith("_")
    }
    assert ENGINE_VERBS == (
        "fit",
        "predict",
        "reconcile",
        "calibrate",
        "order",
        "observe",
        "settle",
        "commit",
    )
    assert public_methods == set(ENGINE_VERBS)


def test_forecast_batch_provenance_cannot_be_forged_publicly() -> None:
    """Reject direct construction without engine-owned cycle provenance."""
    panel = _panel()
    session = _session()
    store = _store(session, panel)
    engine = _engine(session=session, panel=panel, store=store)
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    engine.observe(origin, session=session, snapshot=_snapshot(store, session, origin))
    issued = engine.predict(engine.fit(request))
    forged = ForecastBatch(issued.frame, calendar=CALENDAR)

    with pytest.raises(EngineError, match="produced by this engine"):
        engine.reconcile(forged)


def test_order_bounds_infinite_policy_output_before_rejection() -> None:
    """Read at most one proposal beyond the requested decision set."""
    panel = _panel()
    session = _session(with_decision=True)
    store = _store(session, panel)

    def proposals():
        while True:
            yield OrderProposal("a", "fixture", 1.0)

    engine = _engine(
        session=session,
        panel=panel,
        store=store,
        orderer=lambda _request: proposals(),
    )
    origin = pd.Timestamp("2026-01-05")
    forecasts = _predict(engine, store, session, origin)

    with pytest.raises(EngineError, match="more proposals than requested"):
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=forecasts,
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
            )
        )


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), True])
def test_order_proposal_requires_a_finite_real_quantity(quantity: object) -> None:
    """Reject quantities that cannot be bounded into a durable whole-unit order."""
    with pytest.raises(EngineError, match="finite real"):
        OrderProposal("a", "fixture", cast(Any, quantity))


@pytest.mark.parametrize(
    ("series_key", "model_name", "match"),
    [
        ("", "fixture", "non-empty"),
        ("a", "", "non-empty"),
        (chr(0xD800), "fixture", "valid UTF-8"),
        ("a", chr(0xD800), "valid UTF-8"),
    ],
)
def test_order_proposal_requires_nonempty_utf8_key_parts(
    series_key: str,
    model_name: str,
    match: str,
) -> None:
    """Keep decision natural-key components nonempty and encodable."""
    with pytest.raises(EngineError, match=match):
        OrderProposal(series_key, model_name, 1.0)


def test_engine_refuses_a_panel_outside_its_session_definition() -> None:
    """Reject a loaded series set that differs from the bound session."""
    panel = Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["b"], dtype="string"),
                TIMESTAMP: pd.to_datetime(["2026-01-01"]),
                OBSERVED_VALUE: pd.Series([1.0], dtype="float64"),
            }
        ),
        calendar=CALENDAR,
        target_support=TargetSupport.REAL,
    )
    session = _session()
    with pytest.raises(EngineError, match="panel series set does not match"):
        _engine(
            session=session,
            panel=panel,
            store=_store(session, panel),
        )
