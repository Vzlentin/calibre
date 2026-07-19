"""Prove event transactions survive both journal failure boundaries."""

from __future__ import annotations

from collections.abc import Callable

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
    HierarchyIndex,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    StockoutRule,
    target_timestamp,
)
from newcalibre.engine import (
    ActualsEvent,
    CommitReceipt,
    Engine,
    EventDriver,
    EventDriverError,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
    OrderProposal,
    OriginCommit,
    OriginEvent,
)
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError
from newcalibre.observe import ActualRecord, ActualsSubmission

_CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
_CONFORMAL = {
    "method": "split-window-sum",
    "coverage": 0.5,
    "calibration_window": 20,
    "protection_period": 2,
}


class _Adapter:
    """Emit deterministic two-step forecasts and expose every execution."""

    def __init__(self, effects: list[str]) -> None:
        self._effects = effects
        self._point: float | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        assert not collect_fitted_values
        self._effects.append(f"fit:{task.origin.isoformat()}")
        self._point = float(task.history[OBSERVED_VALUE].iloc[-1])

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        assert self._point is not None
        self._effects.append(f"predict:{task.origin.isoformat()}")
        frame = pd.DataFrame.from_records(
            [
                {
                    SERIES_KEY: "sku",
                    TARGET_TIMESTAMP: target_timestamp(
                        task.origin,
                        step,
                        calendar=task.calendar,
                    ),
                    ACTUAL_VALUE: float("nan"),
                    POINT_FORECAST: self._point,
                    HORIZON_STEP: step,
                    ORIGIN: task.origin,
                    MODEL_NAME: "restart-event",
                }
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

    def fitted_values(self, task: ForecastTask):
        raise AdapterCapabilityError("restart event fixture has no fitted values")

    def dump_state(self) -> bytes:
        raise AdapterCapabilityError("restart event fixture has no persistence")

    def load_state(self, state: bytes) -> None:
        raise AdapterCapabilityError("restart event fixture has no persistence")

    def update(self, task: ForecastTask) -> None:
        raise AdapterCapabilityError("restart event fixture has no incremental update")


class _InterruptingSink(InMemoryLedgerSink):
    """Fail once immediately before or after a selected journal publication."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        fail_after_journal: bool,
        selected: Callable[[OriginCommit], bool],
    ) -> None:
        super().__init__(session=session, calendar=_CALENDAR)
        self._fail_after_journal = fail_after_journal
        self._selected = selected
        self._failed = False

    def commit(self, write: OriginCommit) -> CommitReceipt:
        if self._failed or not self._selected(write):
            return super().commit(write)
        self._failed = True
        if not self._fail_after_journal:
            raise RuntimeError("failure before event journal")
        super().commit(write)
        raise RuntimeError("failure after event journal")


def _panel() -> Panel:
    timestamps = pd.date_range("2026-01-01", periods=7, freq="D")
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["sku"] * len(timestamps), dtype="string"),
                TIMESTAMP: timestamps,
                OBSERVED_VALUE: pd.Series(range(1, 8), dtype="float64"),
            }
        ),
        calendar=_CALENDAR,
    )


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="event-restart",
        series_keys=("sku",),
        calendar=_CALENDAR,
        horizon=2,
        model_config={"backend": "restart-event"},
        conformal_config=_CONFORMAL,
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=("sku",),
        cost_structure=CostStructure(1.0, 1.0, 0.5, 2.0),
        decision_timing=DecisionTiming(lead_time=1, review_period=1),
        stockout_rule=StockoutRule.LOST_SALES,
    )


def _driver(
    *,
    panel: Panel,
    session: SessionIdentity,
    sink: InMemoryLedgerSink,
    states: InMemoryCalibrationStateStore,
    artifacts: InMemoryArtifactStore,
    effects: list[str],
) -> EventDriver:
    actuals = InMemoryActualsSource(panel, actuals_semantics=ActualsSemantics.DEMAND)
    engine = Engine(
        panel_source=InMemoryPanelSource(panel),
        actuals_source=actuals,
        artifact_store=artifacts,
        calibration_state_store=states,
        ledger_sink=sink,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(panel.series_keys),
        adapter_resolver=lambda _configuration: _Adapter(effects),
        orderer=lambda _request: (OrderProposal("sku", "restart-event", 0.0),),
    )
    return EventDriver(
        engine=engine,
        ledger_sink=sink,
        actuals_semantics=ActualsSemantics.DEMAND,
    )


def _origin(session: SessionIdentity, day: int, *, seed: bool = False) -> OriginEvent:
    return OriginEvent(
        session=session,
        origin=pd.Timestamp(f"2026-01-{day:02d}"),
        scope=Scope.GLOBAL,
        initial_inventory_positions=({"sku": InventoryPosition(10.0, 0.0, 0.0)} if seed else None),
    )


def _actual(session: SessionIdentity, day: int, value: float) -> ActualsEvent:
    return ActualsEvent(
        session,
        ActualsSubmission((ActualRecord("sku", pd.Timestamp(f"2026-01-{day:02d}"), value),)),
    )


def _prefix(driver: EventDriver, session: SessionIdentity) -> None:
    driver.handle(_origin(session, 3, seed=True))
    driver.handle(_actual(session, 3, 3.0))
    driver.handle(_origin(session, 4))


def _durable_state(
    sink: InMemoryLedgerSink,
    states: InMemoryCalibrationStateStore,
    artifacts: InMemoryArtifactStore,
) -> tuple:
    return (
        sink.forecasts,
        sink.orders,
        sink.settlements,
        sink.observed_history,
        sink.pending_observations,
        sink.observation_resolutions,
        sink.observe_annotations,
        tuple(sorted(states.snapshot(sink.session).items())),
        tuple(sorted(artifacts.artifacts.items())),
    )


def _uninterrupted_actual_state() -> tuple:
    panel = _panel()
    session = _session()
    sink = InMemoryLedgerSink(session=session, calendar=_CALENDAR)
    states = InMemoryCalibrationStateStore()
    artifacts = InMemoryArtifactStore()
    driver = _driver(
        panel=panel,
        session=session,
        sink=sink,
        states=states,
        artifacts=artifacts,
        effects=[],
    )
    _prefix(driver, session)
    outcome = driver.handle(_actual(session, 4, 4.0))
    driver.handle(_origin(session, 5))
    return _durable_state(sink, states, artifacts), outcome


@pytest.mark.parametrize("fail_after_journal", [False, True])
def test_actual_restart_repairs_one_complete_window_transaction(
    fail_after_journal: bool,
) -> None:
    expected_state, expected_outcome = _uninterrupted_actual_state()
    panel = _panel()
    session = _session()
    target = ("sku", pd.Timestamp("2026-01-04"))
    sink = _InterruptingSink(
        session=session,
        fail_after_journal=fail_after_journal,
        selected=lambda write: target in write.actual_keys,
    )
    states = InMemoryCalibrationStateStore()
    artifacts = InMemoryArtifactStore()
    effects: list[str] = []
    driver = _driver(
        panel=panel,
        session=session,
        sink=sink,
        states=states,
        artifacts=artifacts,
        effects=effects,
    )
    _prefix(driver, session)
    before = _durable_state(sink, states, artifacts)
    retained = sink.pending_observations[0]
    assert retained.resolution is not None

    event = _actual(session, 4, 4.0)
    with pytest.raises(RuntimeError, match="failure (before|after) event journal"):
        driver.handle(event)

    if not fail_after_journal:
        assert _durable_state(sink, states, artifacts) == before
    else:
        assert len(sink.settlements) == 2
        assert sink.receipt(event_key := expected_outcome.receipt.commit_key) is not None
        assert tuple(sorted(states.snapshot(session).items())) == before[7]
        assert event_key == expected_outcome.receipt.commit_key

    resumed_effects: list[str] = []
    resumed = _driver(
        panel=panel,
        session=session,
        sink=sink,
        states=states,
        artifacts=artifacts,
        effects=resumed_effects,
    )
    outcome = resumed.handle(event)

    assert outcome.settlement_periods == expected_outcome.settlement_periods
    assert resumed_effects == []
    resumed.handle(_origin(session, 5))
    assert _durable_state(sink, states, artifacts) == expected_state
    assert sink.pending_observations
    assert len(sink.observe_annotations) == 2
    with pytest.raises(EventDriverError, match="different input facts"):
        resumed.handle(_actual(session, 4, 5.0))
    assert _durable_state(sink, states, artifacts) == expected_state


def test_origin_lost_response_retries_without_model_or_order_callbacks() -> None:
    panel = _panel()
    session = _session()
    fail_origin = pd.Timestamp("2026-01-04")
    sink = _InterruptingSink(
        session=session,
        fail_after_journal=True,
        selected=lambda write: write.origin == fail_origin and bool(write.forecasts),
    )
    states = InMemoryCalibrationStateStore()
    artifacts = InMemoryArtifactStore()
    interrupted_effects: list[str] = []
    driver = _driver(
        panel=panel,
        session=session,
        sink=sink,
        states=states,
        artifacts=artifacts,
        effects=interrupted_effects,
    )
    driver.handle(_origin(session, 3, seed=True))
    event = _origin(session, 4)

    with pytest.raises(RuntimeError, match="failure after event journal"):
        driver.handle(event)
    durable = _durable_state(sink, states, artifacts)

    resumed_effects: list[str] = []
    resumed = _driver(
        panel=panel,
        session=session,
        sink=sink,
        states=states,
        artifacts=artifacts,
        effects=resumed_effects,
    )
    outcome = resumed.handle(event)

    assert outcome.receipt == sink.receipt(fail_origin)
    assert resumed_effects == []
    repaired = _durable_state(sink, states, artifacts)
    assert repaired[:7] == durable[:7]
    assert repaired[8] == durable[8]
    assert tuple(sorted(states.snapshot(session).items())) == tuple(
        sorted(outcome.receipt.state_updates.items())
    )
    with pytest.raises(EventDriverError, match="different input facts"):
        resumed.handle(
            OriginEvent(
                session=session,
                origin=fail_origin,
                scope=Scope.LOCAL,
            )
        )
