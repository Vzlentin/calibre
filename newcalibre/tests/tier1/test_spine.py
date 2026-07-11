"""Exercise the fixed engine spine through its exact public verb surface."""

from __future__ import annotations

import builtins
import socket
import sqlite3
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
    Calendar,
    CostStructure,
    DecisionTiming,
    ForecastTask,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    target_timestamp,
)
from newcalibre.engine import (
    ENGINE_VERBS,
    CalibrationResult,
    CommitReceipt,
    Engine,
    ForecastBatch,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    OrderRequest,
    OriginCommit,
    OriginRequest,
    Phase,
    PhaseError,
    PhaseEvent,
    SettlementRequest,
    Spine,
)
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError
from newcalibre.ledger import Ledger, OrderRow

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
MODEL_CONFIG = {"backend": "fixture", "name": "fixture"}


def _panel() -> Panel:
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["a"] * 7, dtype="string"),
                TIMESTAMP: pd.date_range("2026-01-01", periods=7, freq="D"),
                OBSERVED_VALUE: pd.Series(range(1, 8), dtype="float64"),
            }
        ),
        calendar=CALENDAR,
    )


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("a",),
        calendar=CALENDAR,
        horizon=1,
        model_config=MODEL_CONFIG,
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

    def save(self, session: SessionIdentity, partition: str, value: bytes) -> None:
        if self._fail:
            self._fail = False
            raise RuntimeError("calibration state store unavailable")
        super().save(session, partition, value)


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

    def __init__(self, ledger: Ledger) -> None:
        super().__init__(ledger)
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
    session = _session()
    artifacts = InMemoryArtifactStore()
    states = RecordingStateStore()
    sink = InMemoryLedgerSink(Ledger(session=session, calendar=CALENDAR))
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

    def order(request: OrderRequest) -> tuple[OrderRow, ...]:
        events.append("order")
        assert request.session == session
        assert request.inventory_positions["a"].value == 0.0
        assert request.cost_structure is not None
        return (
            OrderRow(
                session=request.session,
                series_key="a",
                origin=request.origin,
                model_name="fixture",
                quantity=1.0,
                arrival_period=CALENDAR.advance(request.origin, 1),
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
                horizon=1,
                scope=Scope.LOCAL,
                model_config=MODEL_CONFIG,
                calibration_partitions=("global",),
                inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
                cost_structure=CostStructure(1.0, 1.0, 1.0, 1.0),
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
        "predict",
        "reconcile",
        "calibrate",
        "order",
        "observe",
        "fit",
        "predict",
        "reconcile",
        "calibrate",
        "order",
    ]
    assert calibration_inputs == [None, b"observed:5.0"]
    assert states.loads == 2
    assert states.states[(session, "global")] == b"observed:5.0"
    assert len(artifacts.artifacts) == 2
    assert len(sink.forecasts) == 2
    assert sink.forecasts[0].actual_value == 5.0
    assert sink.forecasts[1].actual_value is None
    assert len(sink.orders) == 2


def test_unconfigured_stages_are_exact_identities() -> None:
    panel = _panel()
    session = _session()
    sink = InMemoryLedgerSink(Ledger(session=session, calendar=CALENDAR))
    engine = _engine(
        panel=panel,
        events=[],
        artifacts=InMemoryArtifactStore(),
        states=InMemoryCalibrationStateStore(),
        sink=sink,
        dispatch=RecordingDispatch(),
    )
    fitted = engine.fit(
        origin=pd.Timestamp("2026-01-05"),
        horizon=1,
        scope=Scope.LOCAL,
        model_config=MODEL_CONFIG,
    )
    forecasts = engine.predict(fitted)
    snapshot = sink.snapshot()
    order_request = OrderRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        forecasts=forecasts,
    )
    settlement_request = SettlementRequest(
        session=session,
        origin=pd.Timestamp("2026-01-05"),
        ledger=snapshot,
        actuals={},
        inventory_positions={"a": InventoryPosition(0.0, 0.0, 0.0)},
        timing=DecisionTiming(lead_time=1, review_period=1),
        cost_structure=CostStructure(1.0, 1.0, 1.0, 1.0),
        stockout_rule="lost-sales",
        actuals_semantics="demand",
    )

    assert engine.reconcile(forecasts) is forecasts
    assert engine.calibrate(forecasts, session=session).forecasts is forecasts
    assert engine.order(order_request) == ()
    assert engine.settle(settlement_request) == ()


def test_phase_failure_is_observable_and_persists_nothing() -> None:
    panel = _panel()
    session = _session()
    artifacts = InMemoryArtifactStore()
    states = InMemoryCalibrationStateStore()
    sink = InMemoryLedgerSink(Ledger(session=session, calendar=CALENDAR))
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
        horizon=1,
        scope=Scope.LOCAL,
        model_config=MODEL_CONFIG,
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
    assert artifacts.artifacts == {}
    assert states.states == {}
    assert sink.forecasts == ()
    assert sink.orders == ()


@pytest.mark.parametrize("failing_port", ["artifact", "state", "ledger"])
def test_commit_failure_retries_without_a_split_origin(failing_port: str) -> None:
    panel = _panel()
    session = _session()
    artifacts = FailOnceArtifactStore() if failing_port == "artifact" else InMemoryArtifactStore()
    states = FailOnceStateStore() if failing_port == "state" else InMemoryCalibrationStateStore()
    ledger = Ledger(session=session, calendar=CALENDAR)
    sink = (
        FailAfterCommitLedgerSink(ledger)
        if failing_port == "ledger"
        else InMemoryLedgerSink(ledger)
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
        horizon=1,
        scope=Scope.LOCAL,
        model_config=MODEL_CONFIG,
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
    sink = InMemoryLedgerSink(Ledger(session=session, calendar=CALENDAR))
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
        horizon=1,
        scope=Scope.LOCAL,
        model_config=MODEL_CONFIG,
    )

    with pytest.raises(PhaseError, match="Resolve.*session does not match"):
        spine.run_origin(request)

    assert events == []
    assert artifacts.artifacts == {}
    assert states.states == {}
    assert sink.forecasts == ()


def test_origin_request_owns_nested_inputs() -> None:
    config: dict[str, object] = {
        "backend": "fixture",
        "name": "fixture",
        "nested": {"values": [1]},
    }
    future = pd.DataFrame({"feature": [1.0]})
    positions = {"a": InventoryPosition(1.0, 2.0, 0.0)}
    request = OriginRequest(
        session=_session(),
        origin=pd.Timestamp("2026-01-05"),
        horizon=1,
        scope=Scope.LOCAL,
        model_config=config,
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
