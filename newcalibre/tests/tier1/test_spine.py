"""Exercise the fixed engine spine through its exact public verb surface."""

from __future__ import annotations

import builtins
import pickle
import socket
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError
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
    DecisionTiming,
    ForecastTask,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    StockoutRule,
    target_timestamp,
)
from newcalibre.engine import (
    ENGINE_VERBS,
    CalibrationResult,
    CommitReceipt,
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
from newcalibre.ledger import OrderRow
from newcalibre.ordering import OrderingConfigError

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
MODEL_CONFIG = {"backend": "fixture", "name": "fixture"}
COST_STRUCTURE = CostStructure(1.0, 1.0, 1.0, 1.0)
ORDERING_POLICY = {"name": "newsvendor"}
TIMING = DecisionTiming(lead_time=1, review_period=1)


def _panel(*, series_keys: tuple[str, ...] = ("a",)) -> Panel:
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
    )


def _session(
    *,
    tenant: str = "tenant-a",
    model_config: Mapping[str, object] = MODEL_CONFIG,
    horizon: int | None = None,
    with_decision: bool = False,
    series_keys: tuple[str, ...] = ("a",),
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=(TIMING.protection_period if with_decision else 1) if horizon is None else horizon,
        model_config=model_config,
        ordering_policy=ORDERING_POLICY if with_decision else None,
        cost_structure=COST_STRUCTURE if with_decision else None,
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
        return frozenset({AdapterCapability.ARTIFACT_PERSISTENCE})

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        assert not collect_fitted_values
        self._events.append("fit")
        self._point = float(task.history[OBSERVED_VALUE].iloc[-1])

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

    def fitted_values(self, task: ForecastTask):
        raise AdapterCapabilityError("fixture has no fitted-values capability")

    def update(self, task: ForecastTask) -> None:
        raise AdapterCapabilityError("fixture has no incremental-update capability")


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
        actuals_source=InMemoryActualsSource(panel),
        artifact_store=InMemoryArtifactStore(),
        calibration_state_store=InMemoryCalibrationStateStore(),
        ledger_sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
        dispatch_backend=RecordingDispatch(),
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
        origin: pd.Timestamp,
    ) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("calibration state store unavailable")
        super().save(session, partition, value, origin=origin)


class RecordingStateStore(InMemoryCalibrationStateStore):
    """Count state reads across Resolve and Calibrate."""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load(self, session: SessionIdentity, partition: str) -> bytes | None:
        self.loads += 1
        return super().load(session, partition)


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
    observer=None,
    reconciler=None,
    calibrator=None,
    orderer=None,
    panel_source=None,
) -> Engine:
    return Engine(
        panel_source=panel_source or InMemoryPanelSource(panel),
        actuals_source=InMemoryActualsSource(panel),
        artifact_store=artifacts,
        calibration_state_store=states,
        ledger_sink=sink,
        dispatch_backend=dispatch,
        adapter_resolver=lambda _config: PersistentFixtureAdapter(events),
        observer=observer,
        reconciler=reconciler,
        calibrator=calibrator,
        orderer=orderer,
    )


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
    calibration_inputs: list[bytes | None] = []
    panel_source = RecordingPanelSource(panel)

    def observe(
        _due: pd.DataFrame,
        resolutions: Mapping[tuple[str, pd.Timestamp, int, str], float],
        _prior: Mapping[str, bytes | None],
    ) -> Mapping[str, bytes]:
        events.append("observe")
        if not resolutions:
            return {}
        actual = next(iter(resolutions.values()))
        return {"global": f"observed:{actual}".encode()}

    def reconcile(forecasts: ForecastBatch) -> ForecastBatch:
        events.append("reconcile")
        return forecasts

    def calibrate(
        forecasts: ForecastBatch,
        prior: Mapping[str, bytes | None],
    ) -> CalibrationResult:
        events.append("calibrate")
        calibration_inputs.append(prior["global"])
        state = b"next-state" if prior["global"] is None else prior["global"]
        assert state is not None
        return CalibrationResult(forecasts, {"global": state})

    def order(request: OrderRequest) -> tuple[OrderProposal, ...]:
        events.append("order")
        assert request.session == session
        assert request.inventory_positions["a"].value == 0.0
        assert request.cost_structure is not None
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
        observer=observe,
        reconciler=reconcile,
        calibrator=calibrate,
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
                calibration_partitions=("global",),
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
    assert dispatch.batch_sizes == [1, 1, 1, 1]
    assert panel_source.loads == 1
    assert events == [
        "observe",
        "fit",
        "load",
        "predict",
        "reconcile",
        "calibrate",
        "order",
        "observe",
        "fit",
        "load",
        "predict",
        "reconcile",
        "calibrate",
        "order",
    ]
    assert calibration_inputs == [None, b"observed:5.0"]
    assert states.loads == 2
    assert states.states[(session, "global")] == b"observed:5.0"
    assert len(artifacts.artifacts) == 2
    assert len(sink.forecasts) == 4
    assert sink.forecasts[0].actual_value == 5.0
    assert all(row.actual_value is None for row in sink.forecasts[1:])
    assert len(sink.orders) == 2


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
    assert engine.reconcile(forecasts) is forecasts
    assert engine.calibrate(forecasts, session=session).forecasts is forecasts
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


def test_ordering_excludes_reconciled_aggregate_nodes_before_policy_and_commit() -> None:
    panel = _panel()
    session = _session(with_decision=True)
    seen_series: list[tuple[str, ...]] = []

    def add_aggregate(forecasts: ForecastBatch) -> ForecastBatch:
        frame = forecasts.frame
        aggregate = frame.copy(deep=True)
        aggregate[SERIES_KEY] = "aggregate"
        combined = pd.concat([frame, aggregate], ignore_index=True)
        combined[SERIES_KEY] = combined[SERIES_KEY].astype("string")
        return ForecastBatch(combined, calendar=CALENDAR)

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
        reconciler=add_aggregate,
        orderer=order,
    )

    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            scope=Scope.LOCAL,
        )
    )

    assert seen_series == [("a",)]
    assert tuple((row.series_key, row.quantity) for row in result.orders) == (("a", 0.0),)


def test_ordering_refuses_inventory_for_a_nondecision_series() -> None:
    panel = _panel()
    session = _session(with_decision=True)
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
    )
    forecasts = engine.predict(engine.fit(request))

    result = engine.order(OrderRequest(session=session, origin=request.origin, forecasts=forecasts))

    assert isinstance(result, DecisionBatch)
    assert result.session == session
    assert result.origin == request.origin
    assert result.requested == (("a", "fixture"),)
    assert result.proposals == (proposal,)
    with pytest.raises(FrozenInstanceError):
        cast(Any, result).proposals = ()
    with pytest.raises(FrozenInstanceError):
        cast(Any, proposal).quantity = 2.0


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

    def explode(_forecasts: ForecastBatch) -> ForecastBatch:
        raise RuntimeError("fixture failure")

    engine = _engine(
        panel=panel,
        events=[],
        artifacts=artifacts,
        states=states,
        sink=sink,
        dispatch=RecordingDispatch(),
        reconciler=explode,
    )
    spine = Spine(engine, reporter=phase_events.append)
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
    )

    with pytest.raises(PhaseError, match="Reconcile.*2026-01-05.*fixture failure") as failure:
        spine.run_origin(request)

    assert failure.value.phase is Phase.RECONCILE
    assert [event.phase for event in phase_events] == [
        Phase.RESOLVE,
        Phase.PREDICT,
        Phase.RECONCILE,
    ]
    assert phase_events[-1].error == "fixture failure"
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
    session = _session()
    artifacts = InMemoryArtifactStore()
    states = FailOnceStateStore() if failing_port == "state" else InMemoryCalibrationStateStore()
    sink = (
        FailAfterCommitLedgerSink(session=session, calendar=CALENDAR)
        if failing_port == "ledger"
        else InMemoryLedgerSink(session=session, calendar=CALENDAR)
    )

    def calibrate(
        forecasts: ForecastBatch,
        _prior: Mapping[str, bytes | None],
    ) -> CalibrationResult:
        return CalibrationResult(forecasts, {"global": b"state"})

    engine = _engine(
        panel=panel,
        events=[],
        artifacts=artifacts,
        states=states,
        sink=sink,
        dispatch=RecordingDispatch(),
        calibrator=calibrate,
    )
    spine = Spine(engine)
    request = OriginRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        scope=Scope.LOCAL,
        calibration_partitions=("global",),
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
    assert states.load(session, "global") == b"state"


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
    )
    with pytest.raises(EngineError, match="does not match"):
        engine.commit(mismatched)

    engine.commit(first)
    assert states.load(session, "global") == b"v2"
    assert engine.commit(second) == second


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
    predicted = fresh_engine.predict(transported)
    assert len(predicted.frame) == 1
    assert fit_events == ["fit"]
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
    other_engine.fit(
        OriginRequest(
            session=other_session,
            origin=request.origin,
            scope=Scope.LOCAL,
        )
    )
    assert other_events == ["fit"]
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

    def explode(_forecasts: ForecastBatch) -> ForecastBatch:
        raise RuntimeError("original failure")

    failing_spine = Spine(
        _engine(
            panel=panel,
            events=[],
            artifacts=InMemoryArtifactStore(),
            states=InMemoryCalibrationStateStore(),
            sink=InMemoryLedgerSink(session=session, calendar=CALENDAR),
            dispatch=RecordingDispatch(),
            reconciler=explode,
        ),
        reporter=broken_reporter,
    )
    with (
        pytest.warns(RuntimeWarning, match="phase reporter failed"),
        pytest.raises(PhaseError, match="Reconcile.*original failure"),
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
