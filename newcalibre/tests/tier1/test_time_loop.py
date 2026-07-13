"""Exercise the historical time loop through in-memory engine ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    CENSOR_STATUS,
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
    CalibrationResult,
    CommitReceipt,
    Engine,
    ForecastBatch,
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
    PhaseEvent,
)
from newcalibre.engine.ports import ActualKey
from newcalibre.engine.time_loop import TimeLoop, TimeLoopError, TimeLoopRequest
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError
from newcalibre.ledger import ForecastKey

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
MODEL_CONFIG = {"backend": "fixture", "name": "fixture"}
COST_STRUCTURE = CostStructure(1.0, 1.0, 1.0, 1.0)
ORDERING_POLICY = {"name": "newsvendor"}
TIMING = DecisionTiming(lead_time=2, review_period=2)
ORIGINS = (
    pd.Timestamp("2026-01-04"),
    pd.Timestamp("2026-01-06"),
    pd.Timestamp("2026-01-09"),
)
INITIAL_POSITION = InventoryPosition(on_hand=100.0, on_order=0.0, backorders=0.0)

type FitHistory = tuple[pd.Timestamp, tuple[pd.Timestamp, ...], tuple[float, ...]]
type ResolutionSnapshot = Mapping[ForecastKey, float]


def _panel(
    values: Sequence[float | None],
    *,
    series_keys: Sequence[str] = ("a",),
) -> Panel:
    frozen_values = tuple(values)
    timestamps = tuple(pd.date_range("2026-01-01", periods=len(frozen_values), freq="D"))
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(
                    [series_key for series_key in series_keys for _value in frozen_values],
                    dtype="string",
                ),
                TIMESTAMP: pd.to_datetime(
                    [timestamp for _series_key in series_keys for timestamp in timestamps]
                ),
                OBSERVED_VALUE: pd.Series(
                    [value for _series_key in series_keys for value in frozen_values],
                    dtype="float64",
                ),
            }
        ),
        calendar=CALENDAR,
    )


def _session(
    *,
    timing: DecisionTiming = TIMING,
    with_decision: bool = True,
    series_keys: tuple[str, ...] = ("a",),
    decision_series_keys: tuple[str, ...] | None = None,
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=timing.protection_period if with_decision else 1,
        model_config=MODEL_CONFIG,
        ordering_policy=ORDERING_POLICY if with_decision else None,
        decision_series_keys=(
            series_keys if with_decision and decision_series_keys is None else decision_series_keys
        ),
        cost_structure=COST_STRUCTURE if with_decision else None,
        decision_timing=timing if with_decision else None,
        stockout_rule=StockoutRule.LOST_SALES if with_decision else None,
    )


class DeterministicFixtureAdapter:
    """Forecast the last strict-history value and expose every fitted window."""

    def __init__(
        self,
        fit_histories: list[FitHistory],
        predicted_origins: list[pd.Timestamp],
    ) -> None:
        self._fit_histories = fit_histories
        self._predicted_origins = predicted_origins
        self._point: float | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset({AdapterCapability.ARTIFACT_PERSISTENCE})

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        assert not collect_fitted_values
        timestamps = tuple(pd.Timestamp(value) for value in task.history[TIMESTAMP])
        values = tuple(float(value) for value in task.history[OBSERVED_VALUE])
        self._fit_histories.append((task.origin, timestamps, values))
        self._point = values[-1]

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        assert self._point is not None
        self._predicted_origins.append(task.origin)
        frame = pd.DataFrame.from_records(
            [
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
        )
        frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
        frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
        frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
        frame[POINT_FORECAST] = frame[POINT_FORECAST].astype("float64")
        frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int64")
        return frame

    def dump_state(self) -> bytes:
        assert self._point is not None
        return self._point.hex().encode()

    def load_state(self, state: bytes) -> None:
        self._point = float.fromhex(state.decode())

    def fitted_values(self, task: ForecastTask):
        raise AdapterCapabilityError("fixture has no fitted-values capability")

    def update(self, task: ForecastTask) -> None:
        raise AdapterCapabilityError("fixture has no incremental-update capability")


class RecordingActualsSource(InMemoryActualsSource):
    """Record exact authoritative lookups and optionally hide declared keys."""

    def __init__(
        self,
        panel: Panel,
        *,
        actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
        omitted: Sequence[ActualKey] = (),
        overrides: Mapping[ActualKey, float] | None = None,
    ) -> None:
        super().__init__(panel, actuals_semantics=actuals_semantics)
        self.calls: list[tuple[tuple[ActualKey, ...], pd.Timestamp]] = []
        self._omitted = frozenset(omitted)
        self._overrides = {} if overrides is None else dict(overrides)

    def for_keys(
        self,
        keys: Sequence[ActualKey],
        *,
        before: pd.Timestamp,
    ) -> Mapping[ActualKey, float]:
        frozen_keys = tuple(keys)
        self.calls.append((frozen_keys, before))
        supplied = {
            key: value
            for key, value in super().for_keys(frozen_keys, before=before).items()
            if key not in self._omitted
        }
        supplied.update(
            (key, value)
            for key, value in self._overrides.items()
            if key in frozen_keys and key[1] < before
        )
        return supplied


class LoseFirstCommitResponseSink(InMemoryLedgerSink):
    """Lose one response after its origin payload and receipt are durable."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        fail_origin: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(session=session, calendar=calendar)
        self._fail_origin = fail_origin
        self._lose_response = True

    def commit(self, write: OriginCommit) -> CommitReceipt:
        receipt = super().commit(write)
        if self._lose_response and (self._fail_origin is None or write.origin == self._fail_origin):
            self._lose_response = False
            raise RuntimeError("ledger commit response lost")
        return receipt


@dataclass(slots=True)
class Runtime:
    """Retain an in-memory engine graph and its observable fixture traces."""

    engine: Engine
    actuals: RecordingActualsSource
    artifacts: InMemoryArtifactStore
    states: InMemoryCalibrationStateStore
    sink: InMemoryLedgerSink
    fit_histories: list[FitHistory]
    predicted_origins: list[pd.Timestamp]
    resolutions: list[ResolutionSnapshot]
    calibration_inputs: list[bytes | None]
    order_origins: list[pd.Timestamp]
    order_positions: dict[pd.Timestamp, InventoryPosition]


def _runtime(
    *,
    session: SessionIdentity,
    forecast_panel: Panel,
    actuals: RecordingActualsSource | None = None,
    artifacts: InMemoryArtifactStore | None = None,
    states: InMemoryCalibrationStateStore | None = None,
    sink: InMemoryLedgerSink | None = None,
    decision_series_key: str = "a",
) -> Runtime:
    actuals = actuals or RecordingActualsSource(forecast_panel)
    artifacts = artifacts or InMemoryArtifactStore()
    states = states or InMemoryCalibrationStateStore()
    sink = sink or InMemoryLedgerSink(session=session, calendar=CALENDAR)
    fit_histories: list[FitHistory] = []
    predicted_origins: list[pd.Timestamp] = []
    resolutions: list[ResolutionSnapshot] = []
    calibration_inputs: list[bytes | None] = []
    order_origins: list[pd.Timestamp] = []
    order_positions: dict[pd.Timestamp, InventoryPosition] = {}

    def observe(
        _due: pd.DataFrame,
        resolved: Mapping[ForecastKey, float],
        _prior: Mapping[str, bytes | None],
    ) -> Mapping[str, bytes]:
        resolutions.append(dict(resolved))
        if not resolved:
            return {}
        value = next(iter(resolved.values()))
        return {"global": f"observed:{value}".encode()}

    def calibrate(
        forecasts: ForecastBatch,
        prior: Mapping[str, bytes | None],
    ) -> CalibrationResult:
        value = prior["global"]
        calibration_inputs.append(value)
        next_value = b"warmup" if value is None else value
        return CalibrationResult(forecasts, {"global": next_value + b"|calibrated"})

    def order(request: OrderRequest) -> tuple[OrderProposal, ...]:
        order_origins.append(request.origin)
        order_positions[request.origin] = request.inventory_positions[decision_series_key]
        return (
            OrderProposal(
                series_key=decision_series_key,
                model_name="fixture",
                quantity=4.0,
            ),
        )

    engine = Engine(
        panel_source=InMemoryPanelSource(forecast_panel),
        actuals_source=actuals,
        artifact_store=artifacts,
        calibration_state_store=states,
        ledger_sink=sink,
        dispatch_backend=InProcessDispatch(),
        adapter_resolver=lambda _config: DeterministicFixtureAdapter(
            fit_histories,
            predicted_origins,
        ),
        observer=observe,
        calibrator=calibrate,
        orderer=order,
    )
    return Runtime(
        engine=engine,
        actuals=actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
        fit_histories=fit_histories,
        predicted_origins=predicted_origins,
        resolutions=resolutions,
        calibration_inputs=calibration_inputs,
        order_origins=order_origins,
        order_positions=order_positions,
    )


def _request(
    session: SessionIdentity,
    *,
    origins: Sequence[pd.Timestamp] = ORIGINS,
    positions: Mapping[str, InventoryPosition] | None = None,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> TimeLoopRequest:
    return TimeLoopRequest(
        session=session,
        origins=origins,
        scope=Scope.LOCAL,
        calibration_partitions=("global",),
        initial_inventory_positions={"a": INITIAL_POSITION} if positions is None else positions,
        actuals_semantics=actuals_semantics,
    )


def _assert_no_effects(runtime: Runtime) -> None:
    assert runtime.fit_histories == []
    assert runtime.predicted_origins == []
    assert runtime.resolutions == []
    assert runtime.calibration_inputs == []
    assert runtime.order_origins == []
    assert runtime.artifacts.artifacts == {}
    assert runtime.states.states == {}
    assert runtime.sink.forecasts == ()
    assert runtime.sink.orders == ()
    assert runtime.sink.settlements == ()


def test_request_requires_explicit_actuals_semantics_before_effects() -> None:
    session = _session()
    runtime = _runtime(session=session, forecast_panel=_panel(range(101, 112)))

    with pytest.raises(TypeError, match="actuals_semantics"):
        TimeLoopRequest(
            session=session,
            origins=ORIGINS,
            scope=Scope.LOCAL,
            calibration_partitions=("global",),
            initial_inventory_positions={"a": INITIAL_POSITION},
        )  # type: ignore[call-arg]

    assert runtime.actuals.calls == []
    _assert_no_effects(runtime)


@pytest.mark.parametrize(
    ("origins", "match"),
    [
        ((), "must not be empty"),
        ((ORIGINS[0], ORIGINS[0]), "strictly increasing and unique"),
        ((ORIGINS[1], ORIGINS[0]), "strictly increasing and unique"),
    ],
)
def test_request_rejects_empty_duplicate_and_decreasing_origins_before_effects(
    origins: tuple[pd.Timestamp, ...],
    match: str,
) -> None:
    session = _session()
    runtime = _runtime(session=session, forecast_panel=_panel(range(101, 112)))

    with pytest.raises(TimeLoopError, match=match):
        _request(session, origins=origins)

    assert runtime.actuals.calls == []
    _assert_no_effects(runtime)


def test_constructor_rejects_off_grid_and_partial_configuration_before_effects() -> None:
    forecast_panel = _panel(range(101, 112))
    decision_session = _session()
    runtime = _runtime(session=decision_session, forecast_panel=forecast_panel)
    off_grid = _request(
        decision_session,
        origins=(pd.Timestamp("2026-01-04 12:00:00"),),
    )

    with pytest.raises(TimeLoopError, match="calendar|grid|member"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=off_grid,
        )
    assert runtime.actuals.calls == []
    _assert_no_effects(runtime)

    no_decision_session = _session(with_decision=False)
    incomplete = _runtime(session=no_decision_session, forecast_panel=forecast_panel)
    with pytest.raises(TimeLoopError, match="complete decision configuration"):
        TimeLoop(
            engine=incomplete.engine,
            actuals_source=incomplete.actuals,
            ledger_sink=incomplete.sink,
            request=_request(no_decision_session),
        )
    assert incomplete.actuals.calls == []
    _assert_no_effects(incomplete)

    with pytest.raises(TimeLoopError, match="exactly match the decision series"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=_request(
                decision_session,
                positions={"other": INITIAL_POSITION},
            ),
        )
    assert runtime.actuals.calls == []
    _assert_no_effects(runtime)

    with pytest.raises(TimeLoopError, match="must not be empty"):
        _request(decision_session, positions={})
    _assert_no_effects(runtime)


@pytest.mark.parametrize(
    ("position", "match"),
    [
        (InventoryPosition(1.0, 0.0, 1.0), "zero opening backorders"),
        (InventoryPosition(1.0, 2.0, 0.0), "does not match the compact ledger index"),
    ],
)
def test_constructor_rejects_invalid_initial_inventory_before_effects(
    position: InventoryPosition,
    match: str,
) -> None:
    session = _session()
    runtime = _runtime(session=session, forecast_panel=_panel(range(101, 112)))

    with pytest.raises(TimeLoopError, match=match):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=_request(session, positions={"a": position}),
        )

    assert runtime.actuals.calls == []
    _assert_no_effects(runtime)


def test_gapped_origins_use_sequence_cadence_and_settle_the_calendar_through_drain() -> None:
    # Deliberately disagree so forecast history and settlement actuals prove their port owners.
    forecast_panel = _panel(range(101, 112))
    authoritative_panel = _panel(range(1, 12))
    session = _session()
    actuals = RecordingActualsSource(authoritative_panel)
    runtime = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=actuals,
    )

    result = TimeLoop(
        engine=runtime.engine,
        actuals_source=runtime.actuals,
        ledger_sink=runtime.sink,
        request=_request(session),
    ).run()

    expected_periods = tuple(pd.date_range("2026-01-04", "2026-01-11", freq="D"))
    assert result.settlement_periods == expected_periods
    assert tuple(record.period for record in runtime.sink.settlements) == expected_periods
    assert result.decision_origins == (ORIGINS[0], ORIGINS[2])
    assert runtime.order_origins == [ORIGINS[0], ORIGINS[2]]
    assert tuple(order.origin for order in runtime.sink.orders) == result.decision_origins
    assert tuple(order.arrival_period for order in runtime.sink.orders) == (
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-11"),
    )

    demands = {record.period: record.transition.demand for record in runtime.sink.settlements}
    assert demands == {period: float(period.day) for period in expected_periods}
    assert runtime.sink.settlements[0].transition.demand == 4.0
    assert runtime.order_positions[ORIGINS[0]].on_hand == 100.0
    assert runtime.order_positions[ORIGINS[2]].on_hand == 74.0

    drain = runtime.sink.settlements[-TIMING.lead_time :]
    assert tuple(record.period for record in drain) == (
        pd.Timestamp("2026-01-10"),
        pd.Timestamp("2026-01-11"),
    )
    assert all(
        order.origin not in {record.period for record in drain} for order in runtime.sink.orders
    )
    assert drain[-1].arrivals == 4.0
    assert result.inventory_positions["a"] == InventoryPosition(48.0, 0.0, 0.0)

    assert [
        (origin, history[-1], values[-1]) for origin, history, values in runtime.fit_histories
    ] == [
        (ORIGINS[0], pd.Timestamp("2026-01-03"), 103.0),
        (ORIGINS[1], pd.Timestamp("2026-01-05"), 105.0),
        (ORIGINS[2], pd.Timestamp("2026-01-08"), 108.0),
    ]
    assert runtime.predicted_origins == list(ORIGINS)

    resolved_by_origin_and_horizon = {
        (row.origin, row.horizon_step): row.actual_value for row in runtime.sink.forecasts
    }
    assert resolved_by_origin_and_horizon == {
        (ORIGINS[0], 1): 4.0,
        (ORIGINS[0], 2): 5.0,
        (ORIGINS[0], 3): 6.0,
        (ORIGINS[0], 4): 7.0,
        (ORIGINS[1], 1): 6.0,
        (ORIGINS[1], 2): 7.0,
        (ORIGINS[1], 3): 8.0,
        (ORIGINS[1], 4): None,
        (ORIGINS[2], 1): None,
        (ORIGINS[2], 2): None,
        (ORIGINS[2], 3): None,
        (ORIGINS[2], 4): None,
    }
    assert runtime.calibration_inputs == [None, b"observed:4.0", b"observed:6.0"]
    assert runtime.states.states[(session, "global")] == b"observed:6.0|calibrated"
    assert len(result.receipts) == len(expected_periods)
    assert all(runtime.sink.receipt(period) is not None for period in expected_periods)


def test_explicit_surrogate_semantics_labels_settlement_without_changing_arithmetic() -> None:
    session = _session()
    forecast_panel = _panel(range(101, 112))
    authoritative_panel = _panel(range(1, 12))
    censored_frame = authoritative_panel.frame
    censored_frame[CENSOR_STATUS] = pd.Series(
        ["censored"] * len(censored_frame),
        dtype="string",
    )
    censored_panel = Panel.from_frame(censored_frame, calendar=CALENDAR)
    demand = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=RecordingActualsSource(authoritative_panel),
    )
    surrogate = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=RecordingActualsSource(
            censored_panel,
            actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
        ),
    )

    demand_result = TimeLoop(
        engine=demand.engine,
        actuals_source=demand.actuals,
        ledger_sink=demand.sink,
        request=_request(session, actuals_semantics=ActualsSemantics.DEMAND),
    ).run()
    surrogate_result = TimeLoop(
        engine=surrogate.engine,
        actuals_source=surrogate.actuals,
        ledger_sink=surrogate.sink,
        request=_request(
            session,
            actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
        ),
    ).run()

    assert {row.actuals_semantics for row in demand.sink.settlements} == {ActualsSemantics.DEMAND}
    assert {row.actuals_semantics for row in surrogate.sink.settlements} == {
        ActualsSemantics.CENSORED_SALES_SURROGATE
    }
    assert surrogate_result.settlement_periods == demand_result.settlement_periods
    assert surrogate_result.decision_origins == demand_result.decision_origins
    assert surrogate_result.inventory_positions == demand_result.inventory_positions
    assert (
        tuple(
            replace(row, actuals_semantics=ActualsSemantics.DEMAND)
            for row in surrogate.sink.settlements
        )
        == demand.sink.settlements
    )


def test_time_loop_refuses_semantics_that_do_not_match_the_actuals_source() -> None:
    session = _session()
    actuals = RecordingActualsSource(
        _panel(range(1, 12)),
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=actuals,
    )

    with pytest.raises(TimeLoopError, match="do not match the actuals source"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=actuals,
            ledger_sink=runtime.sink,
            request=_request(session, actuals_semantics=ActualsSemantics.DEMAND),
        )

    assert actuals.calls == []
    _assert_no_effects(runtime)


@pytest.mark.parametrize("committed_origins", (ORIGINS[:1], ORIGINS))
def test_resume_refuses_changed_actuals_semantics_before_reading_actuals(
    committed_origins: Sequence[pd.Timestamp],
) -> None:
    session = _session()
    forecast_panel = _panel(range(101, 112))
    authoritative_panel = _panel(range(1, 12))
    artifacts = InMemoryArtifactStore()
    states = InMemoryCalibrationStateStore()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    demand_actuals = RecordingActualsSource(authoritative_panel)
    initial = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=demand_actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
    )
    TimeLoop(
        engine=initial.engine,
        actuals_source=demand_actuals,
        ledger_sink=sink,
        request=_request(session, origins=committed_origins),
    ).run()
    durable_counts = (len(sink.forecasts), len(sink.orders), len(sink.settlements))
    durable_states = dict(states.states)

    surrogate_actuals = RecordingActualsSource(
        authoritative_panel,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    resumed = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=surrogate_actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
    )

    with pytest.raises(TimeLoopError, match="durable settlement state"):
        TimeLoop(
            engine=resumed.engine,
            actuals_source=surrogate_actuals,
            ledger_sink=sink,
            request=_request(
                session,
                actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
            ),
        )

    assert surrogate_actuals.calls == []
    assert (len(sink.forecasts), len(sink.orders), len(sink.settlements)) == durable_counts
    assert states.states == durable_states
    assert resumed.fit_histories == []
    assert resumed.predicted_origins == []
    assert resumed.order_origins == []


def test_time_loop_requests_actuals_and_settles_only_the_session_decision_series() -> None:
    series_keys = ("aggregate", "bottom")
    session = _session(
        series_keys=series_keys,
        decision_series_keys=("bottom",),
    )
    forecast_panel = _panel(range(101, 112), series_keys=series_keys)
    actuals = RecordingActualsSource(_panel(range(1, 12), series_keys=series_keys))
    runtime = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=actuals,
        decision_series_key="bottom",
    )
    origins = (ORIGINS[0],)

    result = TimeLoop(
        engine=runtime.engine,
        actuals_source=actuals,
        ledger_sink=runtime.sink,
        request=_request(
            session,
            origins=origins,
            positions={"bottom": INITIAL_POSITION},
        ),
    ).run()

    expected_periods = tuple(pd.date_range("2026-01-04", "2026-01-06", freq="D"))
    assert actuals.calls[0] == (
        tuple(("bottom", period) for period in expected_periods),
        pd.Timestamp("2026-01-07"),
    )
    assert result.settlement_periods == expected_periods
    assert {row.series_key for row in runtime.sink.forecasts} == {"aggregate", "bottom"}
    assert {row.series_key for row in runtime.sink.orders} == {"bottom"}
    assert {row.series_key for row in runtime.sink.settlements} == {"bottom"}
    assert tuple(result.inventory_positions) == ("bottom",)


def test_request_ending_before_the_durable_frontier_is_rejected() -> None:
    session = _session()
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=RecordingActualsSource(_panel(range(1, 12))),
    )
    TimeLoop(
        engine=runtime.engine,
        actuals_source=runtime.actuals,
        ledger_sink=runtime.sink,
        request=_request(session),
    ).run()
    calls_before = tuple(runtime.actuals.calls)

    with pytest.raises(TimeLoopError, match="precedes durable settlement frontier"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=_request(session, origins=(ORIGINS[0],)),
        )

    assert tuple(runtime.actuals.calls) == calls_before


def test_missing_non_drain_actual_fails_at_construction_without_engine_effects() -> None:
    session = _session()
    missing = ("a", pd.Timestamp("2026-01-05"))
    actuals = RecordingActualsSource(_panel(range(1, 12)), omitted=(missing,))
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=actuals,
    )

    with pytest.raises(TimeLoopError, match="settlement actuals.*missing.*2026-01-05"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=_request(session),
        )

    expected_periods = tuple(pd.date_range("2026-01-04", "2026-01-11", freq="D"))
    assert runtime.actuals.calls == [
        (
            tuple(("a", period) for period in expected_periods),
            pd.Timestamp("2026-01-12"),
        )
    ]
    _assert_no_effects(runtime)


def test_missing_drain_actual_fails_at_construction_without_engine_effects() -> None:
    session = _session()
    missing = ("a", pd.Timestamp("2026-01-11"))
    actuals = RecordingActualsSource(_panel(range(1, 12)), omitted=(missing,))
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=actuals,
    )

    with pytest.raises(TimeLoopError, match="settlement actuals.*missing.*2026-01-11"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=_request(session),
        )

    expected_periods = tuple(pd.date_range("2026-01-04", "2026-01-11", freq="D"))
    assert runtime.actuals.calls == [
        (
            tuple(("a", period) for period in expected_periods),
            pd.Timestamp("2026-01-12"),
        )
    ]
    _assert_no_effects(runtime)


def test_invalid_drain_actual_fails_at_construction_without_engine_effects() -> None:
    session = _session()
    invalid = ("a", pd.Timestamp("2026-01-11"))
    actuals = RecordingActualsSource(
        _panel(range(1, 12)),
        overrides={invalid: float("nan")},
    )
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=actuals,
    )

    with pytest.raises(TimeLoopError, match="settlement actuals.*2026-01-11.*finite"):
        TimeLoop(
            engine=runtime.engine,
            actuals_source=runtime.actuals,
            ledger_sink=runtime.sink,
            request=_request(session),
        )

    _assert_no_effects(runtime)


def test_settlement_free_receipt_cannot_skip_a_calendar_period() -> None:
    session = _session()
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=RecordingActualsSource(_panel(range(1, 12))),
    )
    runtime.sink.commit(
        OriginCommit(
            session=session,
            origin=ORIGINS[0],
            state_updates={"global": b"partial"},
        )
    )
    loop = TimeLoop(
        engine=runtime.engine,
        actuals_source=runtime.actuals,
        ledger_sink=runtime.sink,
        request=_request(session),
    )

    with pytest.raises(TimeLoopError, match="receipt.*does not contain exactly"):
        loop.run()

    assert runtime.sink.settlements == ()
    assert runtime.fit_histories == []
    assert runtime.predicted_origins == []
    assert runtime.order_origins == []
    assert runtime.states.states == {}


def test_receipt_hole_before_a_later_origin_is_rejected_before_callbacks() -> None:
    session = _session()
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 112)),
        actuals=RecordingActualsSource(_panel(range(1, 12))),
    )
    later = ORIGINS[1]
    for index, period in enumerate((ORIGINS[0], later), start=1):
        runtime.sink._commits[period] = CommitReceipt(  # type: ignore[attr-defined]
            session=session,
            origin=period,
            digest=str(index) * 64,
            state_updates={},
            settlement_periods=(period,),
        )
    loop = TimeLoop(
        engine=runtime.engine,
        actuals_source=runtime.actuals,
        ledger_sink=runtime.sink,
        request=_request(session),
    )

    with pytest.raises(TimeLoopError, match="follows uncommitted period 2026-01-05"):
        loop.run()

    assert runtime.fit_histories == []
    assert runtime.predicted_origins == []
    assert runtime.order_origins == []
    assert runtime.sink.settlements == ()


def test_reconstructed_loop_repairs_lost_commit_without_callbacks_or_rebooking() -> None:
    session = _session()
    forecast_panel = _panel(range(101, 112))
    actuals = RecordingActualsSource(_panel(range(1, 12)))
    artifacts = InMemoryArtifactStore()
    states = InMemoryCalibrationStateStore()
    sink = LoseFirstCommitResponseSink(session=session, calendar=CALENDAR)
    interrupted = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
    )

    with pytest.raises(PhaseError, match="Commit.*response lost"):
        TimeLoop(
            engine=interrupted.engine,
            actuals_source=actuals,
            ledger_sink=sink,
            request=_request(session),
        ).run()

    first_receipt = sink.receipt(ORIGINS[0])
    assert first_receipt is not None
    assert len(sink.forecasts) == TIMING.protection_period
    assert len(sink.orders) == 1
    assert len(sink.settlements) == 1
    assert states.states == {}

    resumed_actuals = RecordingActualsSource(
        _panel(range(1, 12)),
        omitted=(("a", ORIGINS[0]),),
    )
    resumed = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=resumed_actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
    )
    phase_events: list[PhaseEvent] = []
    loop = TimeLoop(
        engine=resumed.engine,
        actuals_source=resumed_actuals,
        ledger_sink=sink,
        request=_request(session),
        reporter=phase_events.append,
    )
    expected_uncommitted = tuple(pd.date_range("2026-01-05", "2026-01-11", freq="D"))
    assert resumed_actuals.calls == [
        (
            tuple(("a", period) for period in expected_uncommitted),
            pd.Timestamp("2026-01-12"),
        )
    ]
    result = loop.run()

    assert all(event.origin != ORIGINS[0] for event in phase_events)
    assert [origin for origin, _history, _values in resumed.fit_histories] == [
        ORIGINS[1],
        ORIGINS[2],
    ]
    assert resumed.predicted_origins == [ORIGINS[1], ORIGINS[2]]
    assert resumed.order_origins == [ORIGINS[2]]
    assert len(sink.forecasts) == len(ORIGINS) * TIMING.protection_period
    assert tuple(order.origin for order in sink.orders) == result.decision_origins
    assert tuple(record.period for record in sink.settlements) == result.settlement_periods
    assert len({record.key for record in sink.settlements}) == len(sink.settlements)
    assert sink.receipt(ORIGINS[0]) == first_receipt
    assert states.states[(session, "global")] == b"observed:6.0|calibrated"


def test_reconstructed_loop_skips_a_durable_filler_after_its_response_is_lost() -> None:
    session = _session()
    forecast_panel = _panel(range(101, 112))
    actuals = RecordingActualsSource(_panel(range(1, 12)))
    artifacts = InMemoryArtifactStore()
    states = InMemoryCalibrationStateStore()
    filler = pd.Timestamp("2026-01-05")
    sink = LoseFirstCommitResponseSink(
        session=session,
        calendar=CALENDAR,
        fail_origin=filler,
    )
    interrupted = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
    )

    with pytest.raises(RuntimeError, match="commit response lost"):
        TimeLoop(
            engine=interrupted.engine,
            actuals_source=actuals,
            ledger_sink=sink,
            request=_request(session),
        ).run()

    filler_receipt = sink.receipt(filler)
    assert filler_receipt is not None
    assert filler_receipt.settlement_periods == (filler,)
    assert tuple(record.period for record in sink.settlements) == (ORIGINS[0], filler)

    resumed = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals=actuals,
        artifacts=artifacts,
        states=states,
        sink=sink,
    )
    phase_events: list[PhaseEvent] = []
    result = TimeLoop(
        engine=resumed.engine,
        actuals_source=actuals,
        ledger_sink=sink,
        request=_request(session),
        reporter=phase_events.append,
    ).run()

    assert all(event.origin != ORIGINS[0] for event in phase_events)
    assert tuple(record.period for record in sink.settlements) == result.settlement_periods
    assert len({record.key for record in sink.settlements}) == len(sink.settlements)
    assert sink.receipt(filler) == filler_receipt
