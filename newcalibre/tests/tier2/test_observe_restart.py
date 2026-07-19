"""Prove real observe-loop restart equality across journal failure boundaries."""

from __future__ import annotations

import pandas as pd
import pytest

from newcalibre.conformal import derive_partition_label, resolve_method
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
    EmissionScope,
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
    CommitReceipt,
    Engine,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
    OrderProposal,
    OrderRequest,
    OriginCommit,
    PhaseError,
    TimeLoop,
    TimeLoopRequest,
)
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError

_CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
_ORIGINS = tuple(pd.date_range("2026-01-03", periods=3, freq="D"))
_FAIL_ORIGIN = _ORIGINS[1]
_TIMING = DecisionTiming(lead_time=1, review_period=1)
_MODEL = "restart-window"
_CONFIGURATION = {
    "method": "split-window-sum",
    "coverage": 0.5,
    "calibration_window": 20,
    "protection_period": 2,
}


class _Adapter:
    """Emit deterministic two-step forecasts and record fitted origins."""

    def __init__(self, fitted_origins: list[pd.Timestamp]) -> None:
        self._fitted_origins = fitted_origins
        self._point: float | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        if collect_fitted_values:
            raise AdapterCapabilityError("restart fixture has no fitted values")
        self._fitted_origins.append(task.origin)
        self._point = float(task.history[OBSERVED_VALUE].iloc[-1])

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        assert self._point is not None
        frame = pd.DataFrame.from_records(
            [
                {
                    SERIES_KEY: "sku",
                    TARGET_TIMESTAMP: target_timestamp(task.origin, step, calendar=task.calendar),
                    ACTUAL_VALUE: float("nan"),
                    POINT_FORECAST: self._point,
                    HORIZON_STEP: step,
                    ORIGIN: task.origin,
                    MODEL_NAME: _MODEL,
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
        raise AdapterCapabilityError("restart fixture has no fitted values")

    def dump_state(self) -> bytes:
        raise AdapterCapabilityError("restart fixture has no persistence")

    def load_state(self, state: bytes) -> None:
        raise AdapterCapabilityError("restart fixture has no persistence")

    def update(self, task: ForecastTask) -> None:
        raise AdapterCapabilityError("restart fixture has no incremental update")


class _InterruptingSink(InMemoryLedgerSink):
    """Fail once immediately before or after the selected origin journal."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        fail_after_journal: bool,
    ) -> None:
        super().__init__(session=session, calendar=_CALENDAR)
        self._fail_after_journal = fail_after_journal
        self._failed = False

    def commit(self, write: OriginCommit) -> CommitReceipt:
        if self._failed or write.origin != _FAIL_ORIGIN:
            return super().commit(write)
        self._failed = True
        if not self._fail_after_journal:
            raise RuntimeError("failure before journal")
        super().commit(write)
        raise RuntimeError("failure after journal")


def _panel() -> Panel:
    timestamps = pd.date_range("2026-01-01", periods=7, freq="D")
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["sku"] * len(timestamps), dtype="string"),
                TIMESTAMP: timestamps,
                OBSERVED_VALUE: pd.Series(range(1, len(timestamps) + 1), dtype="float64"),
            }
        ),
        calendar=_CALENDAR,
    )


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="observe-restart",
        series_keys=("sku",),
        calendar=_CALENDAR,
        horizon=2,
        model_config={"backend": _MODEL},
        conformal_config=_CONFIGURATION,
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=("sku",),
        cost_structure=CostStructure(1.0, 1.0, 0.5, 2.0),
        decision_timing=_TIMING,
        stockout_rule=StockoutRule.LOST_SALES,
    )


def _order(request: OrderRequest) -> tuple[OrderProposal, ...]:
    assert request.timing == _TIMING
    return (OrderProposal("sku", _MODEL, 0.0),)


def _engine(
    *,
    panel: Panel,
    session: SessionIdentity,
    actuals: InMemoryActualsSource,
    sink: InMemoryLedgerSink,
    states: InMemoryCalibrationStateStore,
    fitted_origins: list[pd.Timestamp],
) -> Engine:
    return Engine(
        panel_source=InMemoryPanelSource(panel),
        actuals_source=actuals,
        artifact_store=InMemoryArtifactStore(),
        calibration_state_store=states,
        ledger_sink=sink,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(panel.series_keys),
        adapter_resolver=lambda _configuration: _Adapter(fitted_origins),
        orderer=_order,
    )


def _request(session: SessionIdentity) -> TimeLoopRequest:
    return TimeLoopRequest(
        session=session,
        origins=_ORIGINS,
        settlement_end=pd.Timestamp("2026-01-06"),
        scope=Scope.GLOBAL,
        initial_inventory_positions={"sku": InventoryPosition(10.0, 0.0, 0.0)},
        actuals_semantics=ActualsSemantics.DEMAND,
    )


def _seed_foreign_state(
    states: InMemoryCalibrationStateStore,
    *,
    session: SessionIdentity,
) -> tuple[str, bytes]:
    label = derive_partition_label(_MODEL, "foreign", EmissionScope.WINDOW_SUM)
    value = resolve_method(_CONFIGURATION).calibrate({label: [1.0, 2.0]})[label]
    states.save(
        session,
        label,
        value,
        origin=pd.Timestamp("2026-01-01"),
    )
    return label, value


def _run_uninterrupted():
    panel = _panel()
    session = _session()
    actuals = InMemoryActualsSource(panel, actuals_semantics=ActualsSemantics.DEMAND)
    sink = InMemoryLedgerSink(session=session, calendar=_CALENDAR)
    states = InMemoryCalibrationStateStore()
    foreign = _seed_foreign_state(states, session=session)
    fitted: list[pd.Timestamp] = []
    result = TimeLoop(
        engine=_engine(
            panel=panel,
            session=session,
            actuals=actuals,
            sink=sink,
            states=states,
            fitted_origins=fitted,
        ),
        actuals_source=actuals,
        ledger_sink=sink,
        request=_request(session),
    ).run()
    return sink, states, result, fitted, foreign


def _durable_state(
    sink: InMemoryLedgerSink,
    states: InMemoryCalibrationStateStore,
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
    )


@pytest.mark.parametrize("fail_after_journal", [False, True])
def test_window_restart_matches_uninterrupted_domain_state(
    fail_after_journal: bool,
) -> None:
    expected_sink, expected_states, expected_result, _expected_fitted, expected_foreign = (
        _run_uninterrupted()
    )
    panel = _panel()
    session = _session()
    actuals = InMemoryActualsSource(panel, actuals_semantics=ActualsSemantics.DEMAND)
    sink = _InterruptingSink(
        session=session,
        fail_after_journal=fail_after_journal,
    )
    states = InMemoryCalibrationStateStore()
    foreign_label, foreign_value = _seed_foreign_state(states, session=session)
    interrupted_fitted: list[pd.Timestamp] = []

    with pytest.raises(PhaseError, match="failure (before|after) journal"):
        TimeLoop(
            engine=_engine(
                panel=panel,
                session=session,
                actuals=actuals,
                sink=sink,
                states=states,
                fitted_origins=interrupted_fitted,
            ),
            actuals_source=actuals,
            ledger_sink=sink,
            request=_request(session),
        ).run()

    assert interrupted_fitted == [origin for origin in _ORIGINS[:2] for _ in range(2)]
    if fail_after_journal:
        assert len(sink.forecasts) == 4
        assert len(sink.orders) == 2
        assert len(sink.settlements) == 2
        retained = sink.pending_observations[0]
        assert retained.resolution is not None
        assert retained.forecast_key.origin == _ORIGINS[0]
    else:
        assert len(sink.forecasts) == 2
        assert len(sink.orders) == 1
        assert len(sink.settlements) == 1
        assert all(value.resolution is None for value in sink.pending_observations)
        assert len(sink.observed_history) == 2
    assert states.snapshot(session)[foreign_label] == foreign_value

    resumed_fitted: list[pd.Timestamp] = []
    result = TimeLoop(
        engine=_engine(
            panel=panel,
            session=session,
            actuals=actuals,
            sink=sink,
            states=states,
            fitted_origins=resumed_fitted,
        ),
        actuals_source=actuals,
        ledger_sink=sink,
        request=_request(session),
    ).run()

    expected_resume_origins = [_ORIGINS[2]] if fail_after_journal else list(_ORIGINS[1:])
    assert resumed_fitted == [origin for origin in expected_resume_origins for _ in range(2)]
    assert result.inventory_positions == expected_result.inventory_positions
    assert _durable_state(sink, states) == _durable_state(expected_sink, expected_states)
    assert sink.pending_observations == ()
    assert sum(value.advanced_delivered_score for value in sink.observe_annotations) == 3
    assert states.snapshot(session)[foreign_label] == foreign_value
    assert expected_foreign == (foreign_label, foreign_value)
