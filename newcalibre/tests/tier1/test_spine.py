"""Exercise the fixed engine spine through its exact public verb surface."""

from __future__ import annotations

import builtins
import inspect
import pickle
import socket
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pandas as pd
import pytest

from newcalibre.conformal import (
    SPLIT_PER_STEP,
    CalibrationSeedBatch,
    derive_partition_label,
    resolve_method,
)
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
    CycleToken,
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
    CalibrationResult,
    CommitReceipt,
    CommitRequest,
    CommitResult,
    DecisionBatch,
    Engine,
    EngineError,
    ForecastBatch,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    OrderProposal,
    OrderRequest,
    OriginCommit,
    OriginRequest,
    Phase,
    PhaseError,
    PhaseEvent,
    SettlementSnapshot,
    SettlementWindow,
    Spine,
)
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError
from newcalibre.ledger import ForecastIssuance, OrderRow
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
                    SERIES_KEY: series_key,
                    TIMESTAMP: timestamp,
                    OBSERVED_VALUE: float(index),
                }
                for series_key in series_keys
                for index, timestamp in enumerate(timestamps, start=1)
            ]
        ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"}),
        calendar=CALENDAR,
        target_support=target_support,
    )


def _all_node_panel(panel: Panel, hierarchy: HierarchyIndex) -> Panel:
    records: list[dict[str, object]] = []
    for timestamp, section in panel.frame.groupby(TIMESTAMP, sort=True, observed=True):
        values = dict(zip(section[SERIES_KEY], section[OBSERVED_VALUE], strict=True))
        aggregated = hierarchy.aggregate(values)
        records.extend(
            {
                SERIES_KEY: label,
                TIMESTAMP: pd.Timestamp(timestamp),
                OBSERVED_VALUE: float(aggregated[label]),
            }
            for label in hierarchy.node_labels
        )
    frame = pd.DataFrame.from_records(records)
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
    frame[OBSERVED_VALUE] = frame[OBSERVED_VALUE].astype("float64")
    return Panel.from_frame(
        frame,
        calendar=panel.calendar,
        target_support=panel.target_support,
    )


def _session(
    *,
    tenant: str = "tenant-a",
    model_config: Mapping[str, object] = MODEL_CONFIG,
    horizon: int | None = None,
    with_decision: bool = False,
    series_keys: tuple[str, ...] = ("a",),
    decision_series_keys: tuple[str, ...] | None = None,
    ordering_policy: Mapping[str, object] = ORDERING_POLICY,
    cost_structure: CostStructure | Mapping[str, CostStructure] = COST_STRUCTURE,
    conformal_config: Mapping[str, object] | None = None,
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=(TIMING.protection_period if with_decision else 1) if horizon is None else horizon,
        model_config=model_config,
        conformal_config=conformal_config,
        ordering_policy=ordering_policy if with_decision else None,
        decision_series_keys=(
            series_keys if with_decision and decision_series_keys is None else decision_series_keys
        ),
        cost_structure=cost_structure if with_decision else None,
        decision_timing=TIMING if with_decision else None,
        stockout_rule=StockoutRule.LOST_SALES if with_decision else None,
    )


class PersistentFixtureAdapter:
    """Native-persistence fixture that makes every engine port observable."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._point: float | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset(
            {
                AdapterCapability.ARTIFACT_PERSISTENCE,
                AdapterCapability.INCREMENTAL_UPDATE,
            }
        )

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask) -> None:
        self._events.append("fit")
        self._point = float(task.history.materialize()[OBSERVED_VALUE].iloc[-1])

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        self._events.append("predict")
        assert self._point is not None
        rows = [
            {
                SERIES_KEY: series_key,
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
            for series_key in task.series_keys
            for step in range(1, task.horizon + 1)
        ]
        frame = pd.DataFrame.from_records(rows)
        frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
        frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
        frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
        frame[POINT_FORECAST] = frame[POINT_FORECAST].astype("float64")
        frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int64")
        return frame

    def dump_state(self) -> bytes:
        assert self._point is not None
        return repr(self._point).encode()

    def load_state(self, state: bytes) -> None:
        self._events.append("load")
        self._point = float(state.decode())

    def fitted_values(self):
        raise AdapterCapabilityError("fixture has no fitted-values capability")

    def update(self, delta) -> None:
        self._events.append("update")
        materialized = delta.materialize()
        if not materialized.empty:
            self._point = float(materialized[OBSERVED_VALUE].iloc[-1])


class OmitsSeriesFixtureAdapter(PersistentFixtureAdapter):
    """Return no rows for series `b` to exercise decision-scope coverage."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        frame = super().predict(task)
        return frame.loc[frame[SERIES_KEY] != "b"].reset_index(drop=True)


class MalformedHorizonFixtureAdapter(PersistentFixtureAdapter):
    """Replace the final requested horizon with one outside the session."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        frame = super().predict(task)
        invalid_step = task.horizon + 1
        frame.loc[frame.index[-1], HORIZON_STEP] = invalid_step
        frame.loc[frame.index[-1], TARGET_TIMESTAMP] = target_timestamp(
            task.origin,
            invalid_step,
            calendar=task.calendar,
        )
        return frame


class MalformedLocalBatchFixtureAdapter(PersistentFixtureAdapter):
    """Corrupt one local result so whole-batch validation rejects it."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        frame = super().predict(task)
        if task.series_keys == ("b",):
            frame.loc[:, SERIES_KEY] = "a"
        return frame


class FailingLocalBatchFixtureAdapter(PersistentFixtureAdapter):
    """Fail one local prediction after an earlier dispatch item completed."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if task.series_keys == ("b",):
            raise RuntimeError("second local prediction failed")
        return super().predict(task)


class ForeignSeriesFixtureAdapter(PersistentFixtureAdapter):
    """Emit a series outside the canonical hierarchy to fail Reconcile."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        frame = super().predict(task)
        frame[SERIES_KEY] = frame[SERIES_KEY].mask(frame[SERIES_KEY] == "a", "foreign")
        return frame


class DataFrameSubclassFixtureAdapter(PersistentFixtureAdapter):
    """Return a pandas subclass to exercise the forecast value boundary."""

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        class CallbackFrame(pd.DataFrame):
            @property
            def _constructor(self):
                return CallbackFrame

        return CallbackFrame(super().predict(task))


class RecordingDispatch:
    """Expose that fit and predict both traverse the dispatch port."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def map(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
    ) -> tuple[Any, ...]:
        self.batch_sizes.append(len(items))
        return tuple(function(item) for item in items)


class RecordingPanelSource(InMemoryPanelSource):
    """Count immutable panel loads across origins."""

    def __init__(self, panel: Panel) -> None:
        super().__init__(panel)
        self.loads = 0

    def load(self) -> Panel:
        self.loads += 1
        return super().load()


def _engine_with_default_resolver(
    *,
    session: SessionIdentity,
    panel: Panel,
    panel_source: RecordingPanelSource,
) -> Engine:
    return Engine(
        panel_source=panel_source,
        actuals_source=InMemoryActualsSource(
            panel,
            actuals_semantics=ActualsSemantics.DEMAND,
        ),
        artifact_store=InMemoryArtifactStore(),
        calibration_state_store=InMemoryCalibrationStateStore(),
        ledger_sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch_backend=RecordingDispatch(),
        hierarchy=HierarchyIndex.flat(panel.series_keys),
    )


def test_adapter_capabilities_refuse_before_panel_load_or_fit() -> None:
    panel = _panel()
    panel_source = RecordingPanelSource(panel)
    session = _session(
        model_config={
            "backend": "seasonal-naive",
            "m": 1,
            "quantile_levels": [0.5],
        }
    )

    with pytest.raises(AdapterCapabilityError, match="native_quantiles"):
        _engine_with_default_resolver(
            session=session,
            panel=panel,
            panel_source=panel_source,
        )

    assert panel_source.loads == 0


def test_ordering_configuration_refuses_before_panel_load_or_fit() -> None:
    panel = _panel()
    panel_source = RecordingPanelSource(panel)
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
        _engine_with_default_resolver(
            session=session,
            panel=panel,
            panel_source=panel_source,
        )

    assert panel_source.loads == 0


def test_heterogeneous_cost_fractiles_refuse_before_panel_load_or_fit() -> None:
    panel = _panel(series_keys=("a", "b"))
    panel_source = RecordingPanelSource(panel)
    session = _session(
        model_config={"backend": "seasonal-naive", "m": 1},
        with_decision=True,
        series_keys=("a", "b"),
        cost_structure={
            "a": CostStructure(1.0, 1.0, 0.0, 0.0),
            "b": CostStructure(3.0, 1.0, 0.0, 0.0),
        },
    )

    with pytest.raises(OrderingConfigError, match="homogeneous critical ratios"):
        _engine_with_default_resolver(
            session=session,
            panel=panel,
            panel_source=panel_source,
        )

    assert panel_source.loads == 0


class FailOnceArtifactStore(InMemoryArtifactStore):
    """Raise before the first artifact write, then behave normally."""

    def __init__(self) -> None:
        super().__init__()
        self._fail = True

    def save(self, key: str, value: bytes) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("artifact store unavailable")
        super().save(key, value)

    def publish(
        self,
        artifacts: Mapping[str, bytes],
        indexes: Mapping[str, bytes],
    ) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("artifact store unavailable")
        super().publish(artifacts, indexes)


class FailOnceStateStore(InMemoryCalibrationStateStore):
    """Raise before the first state write, then behave normally."""

    def __init__(self) -> None:
        super().__init__()
        self._fail = True

    def save(
        self,
        session: SessionIdentity,
        partition: str,
        value: bytes,
        *,
        sequence: int,
    ) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("calibration state store unavailable")
        super().save(session, partition, value, sequence=sequence)


class RecordingStateStore(InMemoryCalibrationStateStore):
    """Count complete state snapshots across engine construction and Resolve."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshots = 0

    def snapshot(self, session: SessionIdentity) -> Mapping[str, bytes]:
        self.snapshots += 1
        return super().snapshot(session)


class FailAfterCommitLedgerSink(InMemoryLedgerSink):
    """Lose the first response after the atomic ledger receipt exists."""

    def __init__(self, *, session: SessionIdentity, calendar: Calendar) -> None:
        super().__init__(session=session, calendar=calendar)
        self._fail = True

    def commit(self, write: OriginCommit) -> CommitReceipt:
        receipt = super().commit(write)
        if self._fail:
            self._fail = False
            raise RuntimeError("ledger commit response lost")
        return receipt


def _engine(
    *,
    panel: Panel,
    events: list[str],
    artifacts: InMemoryArtifactStore,
    states: InMemoryCalibrationStateStore,
    sink: InMemoryLedgerSink,
    dispatch: RecordingDispatch,
    reconciliation_strategy: str = "none",
    orderer=None,
    panel_source=None,
    adapter_resolver=None,
    hierarchy: HierarchyIndex | None = None,
) -> Engine:
    return Engine(
        panel_source=panel_source or InMemoryPanelSource(panel),
        actuals_source=InMemoryActualsSource(
            panel,
            actuals_semantics=ActualsSemantics.DEMAND,
        ),
        artifact_store=artifacts,
        calibration_state_store=states,
        ledger_sink=sink,
        dispatch_backend=dispatch,
        hierarchy=hierarchy or HierarchyIndex.flat(panel.series_keys),
        adapter_resolver=(
            (lambda _config: PersistentFixtureAdapter(events))
            if adapter_resolver is None
            else adapter_resolver
        ),
        reconciliation_strategy=reconciliation_strategy,
        orderer=orderer,
    )


def test_engine_supplies_panel_target_support_to_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[TargetSupport] = []

    class RecordingReconciler:
        def __call__(
            self,
            frame: pd.DataFrame,
            hierarchy: HierarchyIndex | None,
            context,
        ) -> pd.DataFrame:
            del hierarchy
            seen.append(context.target_support)
            return frame

    monkeypatch.setattr(
        "newcalibre.engine.spine.resolve_strategy",
        lambda _name: RecordingReconciler(),
    )
    panel = _panel(target_support=TargetSupport.NONNEGATIVE)
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        reconciliation_strategy="recording",
    )
    request = OriginRequest(session=session, origin=pd.Timestamp("2026-01-05"), scope=Scope.LOCAL)

    predicted = engine.predict(engine.fit(request))
    engine.reconcile(predicted)

    assert seen == [TargetSupport.NONNEGATIVE]


def test_spine_runs_fixed_phases_and_uses_every_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    session = _session(with_decision=True)
    artifacts = InMemoryArtifactStore()
    states = RecordingStateStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    dispatch = RecordingDispatch()
    events: list[str] = []
    panel_source = RecordingPanelSource(panel)

    def order(request: OrderRequest) -> tuple[OrderProposal, ...]:
        events.append("order")
        assert request.session == session
        assert request.inventory_positions["a"].value == 0.0
        assert request.costs_by_series == {"a": COST_STRUCTURE}
        assert request.timing == TIMING
        assert request.stockout_rule is StockoutRule.LOST_SALES
        return (
            OrderProposal(
                series_key="a",
                model_name="fixture",
                quantity=1.0,
            ),
        )

    engine = _engine(
        panel=panel,
        events=events,
        artifacts=artifacts,
        states=states,
        sink=sink,
        dispatch=dispatch,
        orderer=order,
        panel_source=panel_source,
    )
    phase_events: list[PhaseEvent] = []
    spine = Spine(engine, reporter=phase_events.append)

    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("engine orchestration attempted direct I/O")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(socket, "create_connection", unexpected_io)
    monkeypatch.setattr(sqlite3, "connect", unexpected_io)

    for origin in (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")):
        spine.run_origin(
            OriginRequest(
                session=session,
                origin=origin,
                scope=Scope.LOCAL,
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
            )
        )

    expected_phases = [
        Phase.RESOLVE,
        Phase.PREDICT,
        Phase.RECONCILE,
        Phase.CALIBRATE,
        Phase.ORDER,
        Phase.COMMIT,
    ] * 2
    assert [event.phase for event in phase_events] == expected_phases
    assert all(event.error is None and event.duration_seconds >= 0.0 for event in phase_events)
    assert dispatch.batch_sizes == [1, 1]
    assert panel_source.loads == 1
    assert events == [
        "fit",
        "predict",
        "order",
        "load",
        "update",
        "predict",
        "order",
    ]
    assert states.snapshots == 3
    assert states.states == {}
    assert len(artifacts.artifacts) == 2
    assert len(sink.forecasts) == 4
    assert sink.forecasts[0].actual_value == 5.0
    assert all(row.actual_value is None for row in sink.forecasts[1:])
    assert len(sink.orders) == 2


@pytest.mark.parametrize(
    ("adapter_type", "fails_during_fit"),
    [
        (FailingLocalBatchFixtureAdapter, True),
        (MalformedLocalBatchFixtureAdapter, False),
    ],
)
def test_forecast_batch_failure_publishes_no_partial_checkpoint_or_cycle_state(
    adapter_type: type[PersistentFixtureAdapter],
    fails_during_fit: bool,
) -> None:
    series_keys = ("a", "b")
    panel = _panel(series_keys=series_keys)
    session = _session(series_keys=series_keys)
    artifacts = InMemoryArtifactStore()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=artifacts,
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        adapter_resolver=lambda _config: adapter_type([]),
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    if fails_during_fit:
        with pytest.raises(RuntimeError, match="second local prediction failed"):
            engine.fit(request)
    else:
        fitted = engine.fit(request)
        with pytest.raises(ForecastFrameError):
            engine.predict(fitted)

    assert artifacts.artifacts == {}
    assert artifacts.artifact_indexes == {}
    assert engine._forecast_results == {}
    assert engine._active_cycles == {}


def test_predict_collapses_dataframe_subclasses_before_phase_validation() -> None:
    panel = _panel()
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        adapter_resolver=lambda _config: DataFrameSubclassFixtureAdapter([]),
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    forecasts = engine.predict(engine.fit(request))

    assert type(forecasts.frame) is pd.DataFrame


def test_forecast_batch_collapses_dataframe_subclasses_before_validation() -> None:
    panel = _panel()
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )
    malformed = engine.predict(engine.fit(request)).frame
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


def test_forecast_batch_materializes_empty_issuance_maps_for_native_quantiles() -> None:
    panel = _panel()
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    origin = pd.Timestamp("2026-01-05")
    predicted = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )
    frame = predicted.frame
    quantile = quantile_column(0.5)
    frame[quantile] = frame[POINT_FORECAST] + 1.0

    forecasts = ForecastBatch(frame, calendar=CALENDAR)

    assert forecasts.frame[quantile].tolist() == [5.0]
    assert {key: dict(row_issuances) for key, row_issuances in forecasts.issuances.items()} == {
        ("a", origin, 1, "fixture"): {}
    }


def test_forecast_batch_refuses_partially_issued_native_quantiles() -> None:
    series_keys = ("a", "b")
    panel = _panel(series_keys=series_keys)
    session = _session(series_keys=series_keys)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    origin = pd.Timestamp("2026-01-05")
    predicted = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
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


def test_forecast_batch_refuses_empty_issuance_maps_for_intervals() -> None:
    panel = _panel()
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    origin = pd.Timestamp("2026-01-05")
    predicted = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )
    frame = predicted.frame
    lower, upper = interval_columns(0.8)
    frame[lower] = frame[POINT_FORECAST] - 1.0
    frame[upper] = frame[POINT_FORECAST] + 1.0

    with pytest.raises(EngineError, match="interval.*issuance"):
        ForecastBatch(
            frame,
            calendar=CALENDAR,
            issuances=predicted.issuances,
        )


def test_settlement_window_requires_explicit_actuals_semantics() -> None:
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


def test_spine_rejects_a_settlement_window_beyond_its_origin_before_phases() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    events: list[str] = []
    artifacts = InMemoryArtifactStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    spine = Spine(
        _engine(
            panel=panel,
            events=events,
            artifacts=artifacts,
            states=InMemoryCalibrationStateStore(),
            sink=sink,
            dispatch=RecordingDispatch(),
        )
    )
    origin = pd.Timestamp("2026-01-05")
    following = CALENDAR.advance(origin, 1)
    snapshot = SettlementSnapshot(
        session=session,
        calendar=CALENDAR,
        periods=(origin, following),
        frontier=None,
        latest_positions={},
        open_order_quantities={"a": 0.0},
        due_arrivals={},
        actuals_semantics=None,
    )

    with pytest.raises(ValueError, match="exactly its origin"):
        spine.run_origin(
            OriginRequest(
                session=session,
                origin=origin,
                scope=Scope.LOCAL,
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
            ),
            settlement=SettlementWindow(
                snapshot=snapshot,
                actuals={("a", origin): 0.0, ("a", following): 0.0},
                actuals_semantics=ActualsSemantics.DEMAND,
            ),
        )

    assert events == []
    assert artifacts.artifacts == {}
    assert sink.forecasts == ()
    assert sink.orders == ()
    assert sink.settlements == ()


def test_unconfigured_stages_are_exact_identities() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
    )
    fitted = engine.fit(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
        )
    )
    forecasts = engine.predict(fitted)
    order_request = OrderRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        forecasts=forecasts,
    )
    reconciled = engine.reconcile(forecasts)
    assert reconciled is not forecasts
    pd.testing.assert_frame_equal(reconciled.frame, forecasts.frame)
    assert reconciled.issuances == forecasts.issuances
    observation = engine.observe(pd.Timestamp("2026-01-05"), session=session)
    assert (
        engine.calibrate(
            reconciled,
            session=session,
            observation=observation,
        ).forecasts
        is reconciled
    )
    assert engine.order(order_request) is None


def test_order_request_rejects_forecasts_from_any_other_origin() -> None:
    panel = _panel()
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    origin = pd.Timestamp("2026-01-05")
    request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    forecasts = engine.predict(engine.fit(request))
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


def test_unknown_reconciliation_strategy_refuses_before_panel_load() -> None:
    panel = _panel()
    panel_source = RecordingPanelSource(panel)
    session = _session()

    with pytest.raises(
        ReconciliationRegistryError,
        match="unknown strategy.*bottom_up, mint_shrink, none, wls_struct, wls_var",
    ):
        _engine(
            panel=panel,
            events=[],
            artifacts=InMemoryArtifactStore(),
            states=InMemoryCalibrationStateStore(),
            sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
            dispatch=RecordingDispatch(),
            panel_source=panel_source,
            reconciliation_strategy="projection",
        )

    assert panel_source.loads == 0


def test_engine_constructor_has_no_provisional_reconciler_callback() -> None:
    parameters = inspect.signature(Engine).parameters

    assert "reconciler" not in parameters
    assert parameters["reconciliation_strategy"].default == "none"


def test_forecast_batch_provenance_can_only_be_bound_by_an_engine() -> None:
    panel = _panel()
    session = _session()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    origin = pd.Timestamp("2026-01-05")
    forecasts = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )
    malformed_frame = forecasts.frame
    malformed_frame.loc[malformed_frame.index[-1], HORIZON_STEP] = 3
    malformed_frame.loc[malformed_frame.index[-1], TARGET_TIMESTAMP] = CALENDAR.advance(origin, 2)
    unbound = ForecastBatch(malformed_frame, calendar=CALENDAR)

    assert unbound.session is None
    with pytest.raises(TypeError, match="unexpected keyword argument 'session'"):
        ForecastBatch(malformed_frame, calendar=CALENDAR, session=session)  # type: ignore[call-arg]
    assert forecasts.token is not None
    with pytest.raises(EngineError, match="forecasts must match its session"):
        CalibrationResult(unbound, token=forecasts.token)


@pytest.mark.parametrize(
    ("proposals", "expected_quantity"),
    [
        ((OrderProposal("a", "fixture", 1.2),), 2.0),
        ((OrderProposal("a", "fixture", -0.2),), 0.0),
        ((), 0.0),
    ],
)
def test_commit_materializes_fractional_negative_and_empty_proposals(
    proposals: tuple[OrderProposal, ...],
    expected_quantity: float,
) -> None:
    panel = _panel()
    session = _session(with_decision=True)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda _request: proposals,
    )

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
            inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )

    assert result.orders == sink.orders
    assert result.orders == (
        OrderRow(
            session=session,
            series_key="a",
            origin=pd.Timestamp("2026-01-05"),
            model_name="fixture",
            quantity=expected_quantity,
            arrival_period=pd.Timestamp("2026-01-06"),
        ),
    )


def test_public_verb_results_reach_commit_without_private_order_math() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (OrderProposal("a", "fixture", 1.2),),
    )
    origin = pd.Timestamp("2026-01-05")
    positions = {"a": InventoryPosition(0.0, 0.0, 0.0)}
    origin_request = OriginRequest(
        session=session,
        origin=origin,
        scope=Scope.LOCAL,
        inventory_positions=positions,
    )

    observation = engine.observe(origin, session=session)
    forecasts = engine.reconcile(engine.predict(engine.fit(origin_request)))
    calibration = engine.calibrate(
        forecasts,
        session=session,
        observation=observation,
    )
    decisions = engine.order(
        OrderRequest(
            session=session,
            origin=origin,
            forecasts=calibration.forecasts,
            inventory_positions=positions,
        )
    )
    foreign_session = _session(tenant="tenant-b", with_decision=True)
    foreign_panel = _panel()
    foreign_engine = _engine(
        panel=foreign_panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=foreign_session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    foreign_forecasts = foreign_engine.predict(
        foreign_engine.fit(OriginRequest(session=foreign_session, origin=origin, scope=Scope.LOCAL))
    )
    foreign_observation = foreign_engine.observe(origin, session=foreign_session)
    foreign_calibration = foreign_engine.calibrate(
        foreign_forecasts,
        session=foreign_session,
        observation=foreign_observation,
    )
    with pytest.raises(EngineError, match="calibration must match"):
        CommitRequest(
            session=session,
            origin=origin,
            token=observation.token,
            observation=observation,
            calibration=foreign_calibration,
            inventory_positions=positions,
            decisions=decisions,
        )
    with pytest.raises(EngineError, match="cycle token"):
        engine.commit(
            CommitRequest(
                session=session,
                origin=origin,
                token=observation.token,
                observation=observation,
                calibration=calibration,
                inventory_positions=positions,
                decisions=DecisionBatch(
                    token=CycleToken(session, origin, observation.token.revision + 1),
                    requested=(),
                    proposals=(),
                ),
            )
        )
    assert sink.forecasts == sink.orders == ()

    committed = engine.commit(
        CommitRequest(
            session=session,
            origin=origin,
            token=observation.token,
            observation=observation,
            calibration=calibration,
            inventory_positions=positions,
            decisions=decisions,
        )
    )

    assert isinstance(committed, CommitResult)
    assert committed.orders == sink.orders
    assert committed.orders[0].quantity == 2.0
    assert sink.receipt(origin) == committed.receipt


def test_commit_request_retry_replays_settlement_without_duplicate_rows() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (OrderProposal("a", "fixture", 1.2),),
    )
    origin = pd.Timestamp("2026-01-05")
    positions = {"a": InventoryPosition(0.0, 0.0, 0.0)}
    origin_request = OriginRequest(
        session=session,
        origin=origin,
        scope=Scope.LOCAL,
        inventory_positions=positions,
    )
    observation = engine.observe(origin, session=session)
    forecasts = engine.reconcile(engine.predict(engine.fit(origin_request)))
    calibration = engine.calibrate(
        forecasts,
        session=session,
        observation=observation,
    )
    decisions = engine.order(
        OrderRequest(
            session=session,
            origin=origin,
            forecasts=calibration.forecasts,
            inventory_positions=positions,
        )
    )
    request = CommitRequest(
        session=session,
        origin=origin,
        token=observation.token,
        observation=observation,
        calibration=calibration,
        inventory_positions=positions,
        decisions=decisions,
        settlement=SettlementWindow(
            snapshot=sink.settlement_snapshot((origin,)),
            actuals={("a", origin): 0.0},
            actuals_semantics=ActualsSemantics.DEMAND,
            token=observation.token,
        ),
    )

    first = engine.commit(request)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.commit(request)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.reconcile(forecasts)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.calibrate(forecasts, session=session, observation=observation)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=calibration.forecasts,
                inventory_positions=positions,
            )
        )

    assert first.receipt == sink.receipt(origin)
    assert len(sink.orders) == 1
    assert len(sink.settlements) == 1


def test_commit_materializes_missing_decision_groups_as_zero_rows() -> None:
    series_keys = ("b", "a")
    panel = _panel(series_keys=series_keys)
    session = _session(with_decision=True, series_keys=series_keys)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (OrderProposal("b", "fixture", 3.0),),
    )

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
            inventory_positions={
                series_key: InventoryPosition(0.0, 0.0, 0.0) for series_key in series_keys
            },
        )
    )

    assert tuple((row.series_key, row.quantity) for row in result.orders) == (
        ("a", 0.0),
        ("b", 3.0),
    )
    assert result.orders == sink.orders


def test_bottom_up_preserves_bottom_issuances_and_excludes_aggregates_from_ordering() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    seen_series: list[tuple[str, ...]] = []

    def order(request: OrderRequest) -> tuple[OrderProposal, ...]:
        seen_series.append(tuple(request.forecasts.frame[SERIES_KEY].unique()))
        return ()

    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        reconciliation_strategy="bottom_up",
        orderer=order,
    )
    origin = pd.Timestamp("2026-01-05")
    predicted = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )
    reconciled = engine.reconcile(predicted)
    bottom_key = ("a", origin, 1, "fixture")
    total_key = ("__total__", origin, 1, "fixture")

    assert reconciled.issuances[bottom_key] == predicted.issuances[bottom_key]
    assert dict(reconciled.issuances[total_key]) == {}
    assert reconciled.frame[SERIES_KEY].tolist() == [
        "a",
        "a",
        "__total__",
        "__total__",
    ]

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=origin,
            scope=Scope.LOCAL,
            inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )

    assert seen_series == [("a",)]
    assert set(result.forecasts.frame[SERIES_KEY]) == {"a", "__total__"}
    assert tuple((row.series_key, row.quantity) for row in result.orders) == (("a", 0.0),)


def test_complete_hierarchy_node_panel_forecasts_from_bottom_actuals() -> None:
    bottom_panel = _panel(series_keys=("a", "b"))
    hierarchy = HierarchyIndex.from_facts(
        pd.DataFrame(
            {
                SERIES_KEY: ["a", "b"],
                "department": ["all", "all"],
            }
        ),
        bottom_series=bottom_panel.series_keys,
    )
    forecast_panel = _all_node_panel(bottom_panel, hierarchy)
    session = _session(series_keys=forecast_panel.series_keys)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = Engine(
        panel_source=InMemoryPanelSource(forecast_panel),
        actuals_source=InMemoryActualsSource(
            bottom_panel,
            actuals_semantics=ActualsSemantics.DEMAND,
        ),
        artifact_store=InMemoryArtifactStore(),
        calibration_state_store=InMemoryCalibrationStateStore(),
        ledger_sink=sink,
        dispatch_backend=RecordingDispatch(),
        hierarchy=hierarchy,
        adapter_resolver=lambda _config: PersistentFixtureAdapter([]),
        reconciliation_strategy="wls_struct",
    )
    origin = pd.Timestamp("2026-01-05")

    Spine(engine).run_origin(OriginRequest(session=session, origin=origin, scope=Scope.GLOBAL))
    observation = engine.observe(pd.Timestamp("2026-01-06"), session=session)
    engine.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-06"),
            observe_cycle=observation.cycle,
            state_updates=observation.cycle.state_updates,
        )
    )

    resolved = {row.series_key: row.actual_value for row in sink.forecasts}
    assert resolved == {
        "a": 5.0,
        "b": 5.0,
        "__aggregate__:department:s:all": 10.0,
        "__total__": 10.0,
    }
    assert {value.series_key for value in sink.observed_history} == {"a", "b"}
    assert sink.pending_observations == ()


@pytest.mark.parametrize("foreign", (False, True))
def test_hierarchical_panel_refuses_partial_or_extra_node_sets(foreign: bool) -> None:
    bottom_panel = _panel(series_keys=("a", "b"))
    hierarchy = HierarchyIndex.from_facts(
        pd.DataFrame(
            {
                SERIES_KEY: ["a", "b"],
                "department": ["all", "all"],
            }
        ),
        bottom_series=bottom_panel.series_keys,
    )
    node_series = tuple(sorted(hierarchy.node_labels, key=str.encode))
    series_keys = (*node_series, "foreign") if foreign else node_series[:-1]
    panel = _panel(series_keys=series_keys)
    session = _session(series_keys=panel.series_keys)

    with pytest.raises(EngineError, match="hierarchy bottoms or all nodes"):
        _engine(
            panel=panel,
            events=[],
            artifacts=InMemoryArtifactStore(),
            states=InMemoryCalibrationStateStore(),
            sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
            dispatch=RecordingDispatch(),
            hierarchy=hierarchy,
        )


def test_real_conformal_apply_consumes_bottom_up_reconciled_points() -> None:
    series_keys = ("a", "b")
    panel = _panel(series_keys=series_keys)
    conformal_config = {
        "method": SPLIT_PER_STEP,
        "coverage": 0.5,
        "calibration_window": 10,
    }
    session = _session(
        series_keys=series_keys,
        conformal_config=conformal_config,
    )
    hierarchy = HierarchyIndex.from_facts(
        pd.DataFrame(
            {
                SERIES_KEY: ["b", "a"],
                "department": ["all", "all"],
            }
        ),
        bottom_series=series_keys,
    )
    state_store = InMemoryCalibrationStateStore()
    partition = derive_partition_label("fixture", "global", EmissionScope.PER_STEP)
    seeded = resolve_method(conformal_config).calibrate(
        CalibrationSeedBatch({partition: [2.0, 2.0, 2.0]})
    )
    for label, value in seeded.items():
        state_store.save(
            session,
            label,
            value,
            sequence=0,
        )
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=state_store,
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        reconciliation_strategy="bottom_up",
        hierarchy=hierarchy,
    )

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
        )
    )

    calibrated = result.forecasts.frame
    lower, upper = interval_columns(0.5)
    aggregate = calibrated.loc[~calibrated[SERIES_KEY].isin(series_keys)]
    assert aggregate[POINT_FORECAST].tolist() == [8.0, 8.0]
    assert aggregate[lower].tolist() == [0.0, 0.0]
    assert aggregate[upper].tolist() == [10.0, 10.0]


def test_session_owned_decision_scope_excludes_a_panel_aggregate_only_from_ordering() -> None:
    series_keys = ("aggregate", "bottom")
    panel = _panel(series_keys=series_keys)
    session = _session(
        with_decision=True,
        series_keys=series_keys,
        decision_series_keys=("bottom",),
        cost_structure={"bottom": COST_STRUCTURE},
    )
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    received: list[OrderRequest] = []

    def order(request: OrderRequest) -> tuple[OrderProposal, ...]:
        received.append(request)
        return (OrderProposal("bottom", "fixture", 1.2),)

    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=order,
    )

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
            inventory_positions={"bottom": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )

    assert len(received) == 1
    assert tuple(received[0].forecasts.frame[SERIES_KEY].unique()) == ("bottom",)
    assert tuple(received[0].inventory_positions) == ("bottom",)
    assert tuple(received[0].costs_by_series or {}) == ("bottom",)
    assert set(result.forecasts.frame[SERIES_KEY]) == {"aggregate", "bottom"}
    assert {row.series_key for row in sink.forecasts} == {"aggregate", "bottom"}
    assert tuple((row.series_key, row.quantity) for row in result.orders) == (("bottom", 2.0),)


def test_ordering_refuses_missing_inventory_before_policy_or_commit() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    calls: list[OrderRequest] = []
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda request: calls.append(request) or (),
    )
    origin = pd.Timestamp("2026-01-05")
    origin_request = OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)
    forecasts = engine.predict(engine.fit(origin_request))

    with pytest.raises(EngineError, match=r"exactly cover.*missing=\['a'\]"):
        engine.order(OrderRequest(session=session, origin=origin, forecasts=forecasts))
    with pytest.raises(PhaseError, match=r"Order.*inventory positions.*missing=\['a'\]"):
        Spine(engine).run_origin(origin_request)

    assert calls == []
    assert sink.orders == ()
    assert sink.forecasts == ()


def test_ordering_refuses_a_missing_configured_decision_series_before_policy() -> None:
    panel = _panel(series_keys=("a", "b"))
    session = _session(with_decision=True, series_keys=("a", "b"))
    calls: list[OrderRequest] = []
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda request: calls.append(request) or (),
        adapter_resolver=lambda _config: OmitsSeriesFixtureAdapter([]),
    )
    origin = pd.Timestamp("2026-01-05")
    forecasts = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )
    reduced = forecasts

    with pytest.raises(EngineError, match=r"decision series.*missing=\['b'\]"):
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=reduced,
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
            )
        )

    observation = engine.observe(origin, session=session)
    calibration = engine.calibrate(reduced, session=session, observation=observation)
    assert calibration.token is not None
    forged_reduced_decisions = DecisionBatch(
        token=CycleToken(session, origin, calibration.token.revision + 1),
        requested=(("a", "fixture"),),
        proposals=(),
    )
    with pytest.raises(EngineError, match="cycle token"):
        engine.commit(
            CommitRequest(
                session=session,
                origin=origin,
                token=observation.token,
                observation=observation,
                calibration=calibration,
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
                decisions=forged_reduced_decisions,
            )
        )

    assert calls == []
    assert sink.forecasts == sink.orders == ()


def test_ordering_refuses_an_incomplete_engine_issued_horizon_before_policy() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    calls: list[OrderRequest] = []
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        orderer=lambda request: calls.append(request) or (),
        adapter_resolver=lambda _config: MalformedHorizonFixtureAdapter([]),
    )
    origin = pd.Timestamp("2026-01-05")
    forecasts = engine.predict(
        engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))
    )

    with pytest.raises(EngineError, match="forecast horizons must exactly cover"):
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=forecasts,
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
            )
        )

    assert calls == []


def test_orderer_receives_one_immutable_cost_entry_per_series() -> None:
    costs = {
        "a": CostStructure(1.0, 1.0, 0.25, 2.0),
        "b": CostStructure(3.0, 1.0, 0.5, 4.0),
    }
    session = _session(
        with_decision=True,
        series_keys=("a", "b"),
        ordering_policy={"name": "newsvendor", "explicit_decision_fractile": 0.6},
        cost_structure=costs,
    )
    panel = _panel(series_keys=("a", "b"))
    captured: list[OrderRequest] = []
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        orderer=lambda request: captured.append(request) or (),
    )
    origin = pd.Timestamp("2026-01-05")
    fitted = engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))

    engine.order(
        OrderRequest(
            session=session,
            origin=origin,
            forecasts=engine.predict(fitted),
            inventory_positions={
                "a": InventoryPosition(0.0, 0.0, 0.0),
                "b": InventoryPosition(0.0, 0.0, 0.0),
            },
        )
    )

    received = captured[0].costs_by_series
    assert received is not None
    assert received == costs
    assert tuple(received) == ("a", "b")
    with pytest.raises(TypeError):
        cast(Any, received)["a"] = COST_STRUCTURE


def test_ordering_refuses_inventory_for_a_nondecision_series() -> None:
    panel = _panel(series_keys=("aggregate", "bottom"))
    session = _session(
        with_decision=True,
        series_keys=("aggregate", "bottom"),
        decision_series_keys=("bottom",),
    )
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (),
    )
    origin = pd.Timestamp("2026-01-05")
    fitted = engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))

    with pytest.raises(EngineError, match="non-decision series"):
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=engine.predict(fitted),
                inventory_positions={"aggregate": InventoryPosition(0.0, 0.0, 0.0)},
            )
        )


def test_whitespace_bearing_series_key_materializes_without_rewriting() -> None:
    series_key = " sku "
    panel = _panel(series_keys=(series_key,))
    session = _session(with_decision=True, series_keys=(series_key,))
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (OrderProposal(series_key, "fixture", 1.0),),
    )

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
            inventory_positions={series_key: InventoryPosition(0.0, 0.0, 0.0)},
        )
    )

    assert tuple(row.series_key for row in result.orders) == (series_key,)


@pytest.mark.parametrize(
    ("proposals", "match"),
    [
        (
            (
                OrderProposal("a", "fixture", 1.0),
                OrderProposal("a", "fixture", 2.0),
            ),
            "duplicate",
        ),
        ((OrderProposal("foreign", "fixture", 1.0),), "not requested"),
    ],
)
def test_order_rejects_duplicate_and_foreign_proposals_atomically(
    proposals: tuple[OrderProposal, ...],
    match: str,
) -> None:
    series_keys = ("a", "b")
    panel = _panel(series_keys=series_keys)
    session = _session(with_decision=True, series_keys=series_keys)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda _request: proposals,
    )
    fitted = engine.fit(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
        )
    )
    request = OrderRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        forecasts=engine.predict(fitted),
        inventory_positions={
            series_key: InventoryPosition(0.0, 0.0, 0.0) for series_key in series_keys
        },
    )

    with pytest.raises(EngineError, match=match):
        engine.order(request)

    assert sink.orders == ()


def test_order_returns_a_canonical_immutable_decision_batch() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    proposal = OrderProposal("a", "fixture", 1.25)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (proposal,),
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
        inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
    )
    forecasts = engine.predict(engine.fit(request))

    result = engine.order(
        OrderRequest(
            session=session,
            origin=request.origin,
            forecasts=forecasts,
            inventory_positions=request.inventory_positions,
        )
    )

    assert isinstance(result, DecisionBatch)
    assert result.session == session
    assert result.origin == request.origin
    assert result.requested == (("a", "fixture"),)
    assert result.proposals == (proposal,)
    with pytest.raises(FrozenInstanceError):
        cast(Any, result).proposals = ()
    with pytest.raises(FrozenInstanceError):
        cast(Any, proposal).quantity = 2.0


def test_commit_rejects_a_publicly_forged_equal_decision_batch() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
        orderer=lambda _request: (OrderProposal("a", "fixture", 1.25),),
    )
    origin = pd.Timestamp("2026-01-05")
    positions = {"a": InventoryPosition(0.0, 0.0, 0.0)}
    forecasts = engine.reconcile(
        engine.predict(engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL)))
    )
    observation = engine.observe(origin, session=session)
    calibration = engine.calibrate(forecasts, session=session, observation=observation)
    issued = engine.order(
        OrderRequest(
            session=session,
            origin=origin,
            forecasts=calibration.forecasts,
            inventory_positions=positions,
        )
    )
    assert issued is not None
    forged = DecisionBatch(
        token=issued.token,
        requested=issued.requested,
        proposals=issued.proposals,
    )

    with pytest.raises(EngineError, match="not issued by this engine"):
        engine.commit(
            CommitRequest(
                session=session,
                origin=origin,
                token=issued.token,
                observation=observation,
                calibration=calibration,
                inventory_positions=positions,
                decisions=forged,
            )
        )

    assert sink.forecasts == sink.orders == ()


def test_decision_batch_snapshots_proposal_subclasses_before_validation() -> None:
    class FlippingProposal(OrderProposal):
        reads = 0

        @property
        def key(self) -> tuple[str, str]:
            type(self).reads += 1
            return ("a", "fixture") if type(self).reads == 1 else ("foreign", "fixture")

    session = _session(with_decision=True)
    result = DecisionBatch(
        token=CycleToken(session, pd.Timestamp("2026-01-05"), 1),
        requested=(("a", "fixture"),),
        proposals=(FlippingProposal("a", "fixture", 1.0),),
    )

    assert type(result.proposals[0]) is OrderProposal
    assert result.proposals[0].key == result.proposals[0].key == ("a", "fixture")


def test_order_bounds_infinite_policy_output_before_atomic_rejection() -> None:
    panel = _panel()
    session = _session(with_decision=True)

    def proposals():
        while True:
            yield OrderProposal("a", "fixture", 1.0)

    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
        orderer=lambda _request: proposals(),
    )
    origin = pd.Timestamp("2026-01-05")
    fitted = engine.fit(OriginRequest(session=session, origin=origin, scope=Scope.LOCAL))

    with pytest.raises(EngineError, match="more proposals than requested"):
        engine.order(
            OrderRequest(
                session=session,
                origin=origin,
                forecasts=engine.predict(fitted),
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
            )
        )


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), True])
def test_order_proposal_requires_a_finite_real_quantity(quantity: object) -> None:
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
    with pytest.raises(EngineError, match=match):
        OrderProposal(series_key, model_name, 1.0)


def test_phase_failure_is_observable_and_commits_no_origin() -> None:
    panel = _panel()
    session = _session()
    artifacts = InMemoryArtifactStore()
    states = InMemoryCalibrationStateStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    phase_events: list[PhaseEvent] = []

    engine = _engine(
        panel=panel,
        events=[],
        artifacts=artifacts,
        states=states,
        sink=sink,
        dispatch=RecordingDispatch(),
        adapter_resolver=lambda _config: ForeignSeriesFixtureAdapter([]),
    )
    spine = Spine(engine, reporter=phase_events.append)
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    with pytest.raises(PhaseError, match="Reconcile.*2026-01-05.*not covered") as failure:
        spine.run_origin(request)

    assert failure.value.phase is Phase.RECONCILE
    assert [event.phase for event in phase_events] == [
        Phase.RESOLVE,
        Phase.PREDICT,
        Phase.RECONCILE,
    ]
    assert phase_events[-1].error is not None
    assert "not covered" in phase_events[-1].error
    assert len(artifacts.artifacts) == 1
    assert states.states == {}
    assert sink.forecasts == ()
    assert sink.orders == ()


def test_fit_artifact_failure_is_retryable_before_any_origin_commit() -> None:
    panel = _panel()
    session = _session()
    artifacts = FailOnceArtifactStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    spine = Spine(
        _engine(
            panel=panel,
            events=[],
            artifacts=artifacts,
            states=InMemoryCalibrationStateStore(),
            sink=sink,
            dispatch=RecordingDispatch(),
        )
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    with pytest.raises(PhaseError, match="Predict.*artifact store unavailable"):
        spine.run_origin(request)
    assert sink.receipt(request.origin) is None
    assert sink.forecasts == ()

    spine.run_origin(request)
    assert len(artifacts.artifacts) == 1
    assert len(sink.forecasts) == 1
    assert sink.orders == ()


@pytest.mark.parametrize("failing_port", ["state", "ledger"])
def test_commit_failure_retries_without_a_split_origin(failing_port: str) -> None:
    panel = _panel()
    session = _session(
        conformal_config={
            "method": "split-per-step",
            "coverage": 0.5,
            "calibration_window": 10,
        }
    )
    artifacts = InMemoryArtifactStore()
    states = FailOnceStateStore() if failing_port == "state" else InMemoryCalibrationStateStore()
    sink = (
        FailAfterCommitLedgerSink(session=session, calendar=CALENDAR)
        if failing_port == "ledger"
        else InMemoryLedgerSink(session=session, calendar=CALENDAR)
    )
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=artifacts,
        states=states,
        sink=sink,
        dispatch=RecordingDispatch(),
    )
    spine = Spine(engine)
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    with pytest.raises(PhaseError, match="Commit.*(store unavailable|response lost)"):
        spine.run_origin(request)
    assert len(sink.forecasts) == 1

    receipt = sink.receipt(request.origin)
    assert receipt is not None
    assert not hasattr(receipt, "forecasts")
    engine.commit(receipt)
    engine.commit(receipt)
    assert len(sink.forecasts) == 1
    assert len(artifacts.artifacts) == 1
    assert len(states.snapshot(session)) == 1


def test_commit_repair_requires_the_ledger_journal_and_never_rolls_state_back() -> None:
    panel = _panel()
    session = _session()
    states = InMemoryCalibrationStateStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=states,
        sink=sink,
        dispatch=RecordingDispatch(),
    )
    first_origin = pd.Timestamp("2026-01-05")
    second_origin = pd.Timestamp("2026-01-06")
    unjournaled = CommitReceipt(
        session=session,
        origin=first_origin,
        digest="0" * 64,
        state_updates={"global": b"forged"},
        sequence=0,
    )

    with pytest.raises(EngineError, match="not journaled"):
        engine.commit(unjournaled)
    assert states.states == {}

    first = engine.commit(
        OriginCommit(
            session=session,
            origin=first_origin,
            state_updates={"global": b"v1"},
        )
    )
    second = engine.commit(
        OriginCommit(
            session=session,
            origin=second_origin,
            state_updates={"global": b"v2"},
        )
    )
    mismatched = CommitReceipt(
        session=session,
        origin=second.origin,
        digest="f" * 64,
        state_updates={"global": b"wrong"},
        sequence=second.sequence,
    )
    with pytest.raises(EngineError, match="does not match"):
        engine.commit(mismatched)

    engine.commit(first)
    assert states.snapshot(session) == {"global": b"v2"}
    assert engine.commit(second) == second


def test_commit_snapshots_ledger_receipts_before_comparison_and_state_repair() -> None:
    class MisleadingReceipt(CommitReceipt):
        def __eq__(self, other: object) -> bool:
            return True

        def __getattribute__(self, name: str) -> Any:
            if name == "state_updates":
                return {"global": b"forged"}
            return super().__getattribute__(name)

    class MisleadingReceiptLedgerSink(InMemoryLedgerSink):
        def commit(self, write: OriginCommit) -> CommitReceipt:
            receipt = super().commit(write)
            return MisleadingReceipt(
                session=receipt.session,
                origin=receipt.origin,
                digest=receipt.digest,
                state_updates=receipt.state_updates,
                sequence=receipt.sequence,
                settlement_periods=receipt.settlement_periods,
            )

    panel = _panel()
    session = _session()
    states = InMemoryCalibrationStateStore()
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=states,
        sink=MisleadingReceiptLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    origin = pd.Timestamp("2026-01-05")

    with pytest.raises(EngineError, match="mismatched commit receipt"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=origin,
                state_updates={"global": b"expected"},
            )
        )

    assert states.states == {}


def test_fit_result_is_process_safe_and_artifacts_are_session_scoped() -> None:
    panel = _panel()
    session = _session()
    shared_artifacts = InMemoryArtifactStore()
    fit_events: list[str] = []
    first_engine = _engine(
        panel=panel,
        events=fit_events,
        artifacts=shared_artifacts,
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )
    fitted = first_engine.fit(request)
    assert all(not hasattr(value, "adapter") for value in fitted)
    assert all(not hasattr(value, "artifact_key") for value in fitted)
    transported = pickle.loads(pickle.dumps(fitted))

    predict_events: list[str] = []
    fresh_engine = _engine(
        panel=panel,
        events=predict_events,
        artifacts=shared_artifacts,
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    with pytest.raises(EngineError, match="stale or foreign"):
        fresh_engine.predict(transported)

    first_engine.predict(fitted)
    fresh_fitted = fresh_engine.fit(request)
    predicted = fresh_engine.predict(fresh_fitted)
    assert len(predicted.frame) == 1
    assert fit_events == ["fit", "predict"]
    assert predict_events == ["load", "predict"]

    other_session = _session(tenant="tenant-b")
    other_events: list[str] = []
    other_engine = _engine(
        panel=panel,
        events=other_events,
        artifacts=shared_artifacts,
        states=InMemoryCalibrationStateStore(),
        sink=InMemoryLedgerSink(session=other_session, calendar=CALENDAR),
        dispatch=RecordingDispatch(),
    )
    other_fitted = other_engine.fit(
        OriginRequest(
            session=other_session,
            origin=request.origin,
            scope=Scope.LOCAL,
        )
    )
    other_engine.predict(other_fitted)
    assert other_events == ["fit", "predict"]
    assert len(shared_artifacts.artifacts) == 2


def test_reporter_failures_never_change_or_mask_phase_outcomes() -> None:
    panel = _panel()
    session = _session()

    def broken_reporter(_event: PhaseEvent) -> None:
        raise RuntimeError("metrics unavailable")

    committed_sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    committed_spine = Spine(
        _engine(
            panel=panel,
            events=[],
            artifacts=InMemoryArtifactStore(),
            states=InMemoryCalibrationStateStore(),
            sink=committed_sink,
            dispatch=RecordingDispatch(),
        ),
        reporter=broken_reporter,
    )
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )
    with pytest.warns(RuntimeWarning, match="phase reporter failed"):
        committed_spine.run_origin(request)
    assert len(committed_sink.forecasts) == 1

    failing_spine = Spine(
        _engine(
            panel=panel,
            events=[],
            artifacts=InMemoryArtifactStore(),
            states=InMemoryCalibrationStateStore(),
            sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
            dispatch=RecordingDispatch(),
            adapter_resolver=lambda _config: ForeignSeriesFixtureAdapter([]),
        ),
        reporter=broken_reporter,
    )
    with (
        pytest.warns(RuntimeWarning, match="phase reporter failed"),
        pytest.raises(PhaseError, match="Reconcile.*not covered"),
    ):
        failing_spine.run_origin(request)


def test_mismatched_session_refuses_before_any_engine_work() -> None:
    panel = _panel()
    session = _session()
    other_session = SessionIdentity.derive(
        tenant="tenant-b",
        series_keys=("a",),
        calendar=CALENDAR,
        horizon=1,
        model_config=MODEL_CONFIG,
    )
    artifacts = InMemoryArtifactStore()
    states = InMemoryCalibrationStateStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    events: list[str] = []
    spine = Spine(
        _engine(
            panel=panel,
            events=events,
            artifacts=artifacts,
            states=states,
            sink=sink,
            dispatch=RecordingDispatch(),
        )
    )
    request = OriginRequest(
        session=other_session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    with pytest.raises(PhaseError, match="Resolve.*session does not match"):
        spine.run_origin(request)

    assert events == []
    assert artifacts.artifacts == {}
    assert states.states == {}
    assert sink.forecasts == ()


def test_engine_refuses_a_panel_outside_its_session_definition() -> None:
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
            panel=panel,
            events=[],
            artifacts=InMemoryArtifactStore(),
            states=InMemoryCalibrationStateStore(),
            sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
            dispatch=RecordingDispatch(),
        )


def test_origin_request_owns_nested_inputs() -> None:
    config: dict[str, object] = {
        "backend": "fixture",
        "name": "fixture",
        "nested": {"values": [1]},
    }
    future = pd.DataFrame({"feature": [1.0]})
    positions = {"a": InventoryPosition(1.0, 2.0, 0.0)}
    session = _session(model_config=config)
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
        future_exogenous=future,
        inventory_positions=positions,
    )

    cast(dict[str, object], config["nested"])["values"] = [99]
    future.loc[0, "feature"] = 99.0
    positions["a"] = InventoryPosition(99.0, 0.0, 0.0)
    materialized = request.model_config
    cast(dict[str, object], materialized["nested"])["values"] = [42]
    returned_future = request.future_exogenous
    assert returned_future is not None
    returned_future.loc[0, "feature"] = 42.0

    assert request.model_config["nested"] == {"values": [1]}
    assert request.future_exogenous is not None
    assert request.future_exogenous.loc[0, "feature"] == 1.0
    assert request.inventory_positions["a"].value == 3.0


def test_engine_exposes_exactly_the_closed_eight_verbs() -> None:
    expected = (
        "fit",
        "predict",
        "reconcile",
        "calibrate",
        "order",
        "observe",
        "settle",
        "commit",
    )
    public_methods = {
        name
        for name, value in Engine.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert expected == ENGINE_VERBS
    assert public_methods == set(expected)
    assert "resolve" not in public_methods
