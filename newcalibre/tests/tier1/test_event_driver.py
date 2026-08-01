"""Exercise the typed event driver over the closed engine surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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
    TargetSupport,
    target_timestamp,
)
from newcalibre.engine import (
    ENGINE_VERBS,
    ActualsCommit,
    ActualsEvent,
    Engine,
    EventDriver,
    EventDriverError,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    OrderProposal,
    OriginCommit,
    OriginEvent,
    PhaseError,
)
from newcalibre.forecasting import (
    AdapterCapability,
    AdapterCapabilityError,
    AdapterExecutionMode,
)
from newcalibre.observe import ActualRecord, ActualsSubmission


def _session() -> SessionIdentity:
    calendar = Calendar("D", phase=pd.Timestamp("2026-01-01"))
    return SessionIdentity.derive(
        tenant="event-driver",
        series_keys=("a",),
        calendar=calendar,
        horizon=1,
        model_config={"backend": "seasonal-naive", "m": 1},
    )


def test_event_values_snapshot_inputs_and_actual_order_is_not_identity() -> None:
    session = _session()
    frame = pd.DataFrame({"feature": [1.0]})
    positions = {"a": InventoryPosition(1.0, 0.0, 0.0)}
    origin = OriginEvent(
        session=session,
        origin=pd.Timestamp("2026-01-03"),
        scope=Scope.LOCAL,
        future_exogenous=frame,
        initial_inventory_positions=positions,
    )
    frame.loc[0, "feature"] = 9.0
    positions["a"] = InventoryPosition(9.0, 0.0, 0.0)

    returned = origin.future_exogenous
    assert returned is not None
    returned.loc[0, "feature"] = 7.0
    assert origin.future_exogenous is not None
    assert origin.future_exogenous.loc[0, "feature"] == 1.0
    assert origin.initial_inventory_positions["a"].on_hand == 1.0

    first = ActualRecord("a", pd.Timestamp("2026-01-01"), 1.0)
    second = ActualRecord("a", pd.Timestamp("2026-01-02"), 2.0)
    forward = ActualsEvent(session, ActualsSubmission((first, second)))
    reverse = ActualsEvent(session, ActualsSubmission((second, first)))

    assert forward.submission.records == reverse.submission.records
    assert forward.fingerprint == reverse.fingerprint


class _Adapter:
    """Emit deterministic forecasts while making model effects observable."""

    def __init__(self, effects: list[str]) -> None:
        self._effects = effects
        self._point: float | None = None

    @property
    def execution_mode(self) -> AdapterExecutionMode:
        return AdapterExecutionMode.MONOLITHIC

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask) -> None:
        self._effects.append("fit")
        self._point = float(task.history.materialize()[OBSERVED_VALUE].iloc[-1])

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        assert self._point is not None
        self._effects.append("predict")
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
        return frame.astype(
            {
                SERIES_KEY: "string",
                ACTUAL_VALUE: "float64",
                POINT_FORECAST: "float64",
                HORIZON_STEP: "int64",
                MODEL_NAME: "string",
            }
        )

    def fitted_values(self):
        raise AdapterCapabilityError("event fixture has no fitted values")

    def dump_state(self) -> bytes:
        raise AdapterCapabilityError("event fixture has no persistence")

    def load_state(self, state: bytes) -> None:
        raise AdapterCapabilityError("event fixture has no persistence")

    def update(self, delta) -> None:
        del delta
        raise AdapterCapabilityError("event fixture has no incremental update")


def _decision_session(
    calendar: Calendar,
    *,
    review_period: int = 1,
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="event-driver",
        series_keys=("a",),
        calendar=calendar,
        horizon=review_period + 1,
        model_config={"backend": "fixture"},
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=("a",),
        cost_structure=CostStructure(1.0, 1.0, 0.5, 2.0),
        decision_timing=DecisionTiming(lead_time=1, review_period=review_period),
        stockout_rule=StockoutRule.LOST_SALES,
    )


def _panel(calendar: Calendar) -> Panel:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="D")
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["a"] * len(timestamps), dtype="string"),
                TIMESTAMP: timestamps,
                OBSERVED_VALUE: pd.Series(range(1, 9), dtype="float64"),
            }
        ),
        calendar=calendar,
        target_support=TargetSupport.REAL,
    )


def _driver(
    *,
    panel: Panel,
    session: SessionIdentity,
    store: InMemoryIndexedRunStore,
    effects: list[str],
    order_origins: list[pd.Timestamp],
) -> EventDriver:
    def order(request) -> tuple[OrderProposal, ...]:
        order_origins.append(request.origin)
        return (OrderProposal("a", "fixture", 1.0),)

    engine = Engine(
        session=session,
        panel_source=InMemoryPanelSource(panel),
        run_store=store,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(panel.series_keys),
        adapter_resolver=lambda _configuration: _Adapter(effects),
        orderer=order,
    )
    return EventDriver(
        engine=engine,
        run_store=store,
        actuals_semantics=ActualsSemantics.DEMAND,
    )


def _runtime(
    *,
    store_factory: Callable[
        [SessionIdentity, Calendar],
        InMemoryIndexedRunStore,
    ] = lambda session, calendar: InMemoryIndexedRunStore(
        session=session,
        calendar=calendar,
        actuals_semantics=ActualsSemantics.DEMAND,
    ),
    review_period: int = 1,
) -> tuple[
    EventDriver,
    SessionIdentity,
    InMemoryIndexedRunStore,
    list[str],
    list[pd.Timestamp],
]:
    calendar = Calendar("D", phase=pd.Timestamp("2026-01-01"))
    panel = _panel(calendar)
    session = _decision_session(panel.calendar, review_period=review_period)
    sink = store_factory(session, panel.calendar)
    effects: list[str] = []
    order_origins: list[pd.Timestamp] = []
    driver = _driver(
        panel=panel,
        session=session,
        store=sink,
        effects=effects,
        order_origins=order_origins,
    )
    return driver, session, sink, effects, order_origins


def _origin(
    session: SessionIdentity,
    value: str,
    *,
    positions: Mapping[str, InventoryPosition] | None = None,
    scope: Scope = Scope.LOCAL,
) -> OriginEvent:
    return OriginEvent(
        session=session,
        origin=pd.Timestamp(value),
        scope=scope,
        initial_inventory_positions=positions,
    )


def _actuals(session: SessionIdentity, value: str, demand: float) -> ActualsEvent:
    return ActualsEvent(
        session,
        ActualsSubmission((ActualRecord("a", pd.Timestamp(value), demand),)),
    )


class _InterleavingRunStore(InMemoryIndexedRunStore):
    """Hold both origin writes at commit and publish one selected origin first."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        first_origin: pd.Timestamp,
    ) -> None:
        super().__init__(
            session=session,
            calendar=calendar,
            actuals_semantics=ActualsSemantics.DEMAND,
        )
        self._first_origin = first_origin
        self._origin_writes_ready = Barrier(2)
        self._first_origin_published = Barrier(2)
        self._race_complete = False

    def commit(self, write):
        if (
            isinstance(write, ActualsCommit)
            or not isinstance(write, OriginCommit)
            or not write.forecasts
            or self._race_complete
        ):
            return super().commit(write)
        self._origin_writes_ready.wait(timeout=5)
        if write.origin == self._first_origin:
            try:
                return super().commit(write)
            finally:
                self._first_origin_published.wait(timeout=5)
        self._first_origin_published.wait(timeout=5)
        try:
            return super().commit(write)
        finally:
            self._race_complete = True


@pytest.mark.parametrize(
    "first_origin",
    [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-04")],
)
def test_concurrent_origin_admission_and_inventory_validation_are_atomic(
    first_origin: pd.Timestamp,
) -> None:
    driver, session, sink, _effects, _order_origins = _runtime(
        store_factory=lambda session, calendar: _InterleavingRunStore(
            session=session,
            calendar=calendar,
            first_origin=first_origin,
        )
    )
    origins = {
        pd.Timestamp("2026-01-04"): _origin(
            session,
            "2026-01-04",
            positions={"a": InventoryPosition(10.0, 0.0, 0.0)},
        ),
        pd.Timestamp("2026-01-05"): _origin(
            session,
            "2026-01-05",
            positions={"a": InventoryPosition(20.0, 0.0, 0.0)},
        ),
    }
    losing_origin = next(origin for origin in origins if origin != first_origin)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            origin: executor.submit(driver.handle, event) for origin, event in origins.items()
        }
        winner = futures[first_origin].result(timeout=10)
        with pytest.raises(PhaseError, match="revision is stale"):
            futures[losing_origin].result(timeout=10)

    assert winner.receipt == sink.receipt(first_origin)
    assert sink.latest_origin == first_origin
    assert {row.origin for row in sink.forecasts} == {first_origin}
    assert {row.origin for row in sink.orders} == {first_origin}
    expected_on_hand = 10.0 if first_origin.day == 4 else 20.0
    snapshot = sink.settlement_snapshot((pd.Timestamp("2026-01-06"),))
    assert snapshot.current_positions["a"].on_hand == expected_on_hand


def test_concurrent_drivers_admit_decision_cadence_at_the_journal_boundary() -> None:
    first_origin = pd.Timestamp("2026-01-04")
    first_driver, session, sink, effects, order_origins = _runtime(
        store_factory=lambda session, calendar: _InterleavingRunStore(
            session=session,
            calendar=calendar,
            first_origin=first_origin,
        ),
        review_period=2,
    )
    second_driver = _driver(
        panel=_panel(sink.calendar),
        session=session,
        store=sink,
        effects=effects,
        order_origins=order_origins,
    )
    initial = {"a": InventoryPosition(10.0, 0.0, 0.0)}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            first_driver.handle,
            _origin(session, "2026-01-04", positions=initial),
        )
        second = executor.submit(
            second_driver.handle,
            _origin(session, "2026-01-05", positions=initial),
        )
        first.result(timeout=10)
        with pytest.raises(PhaseError, match="revision is stale"):
            second.result(timeout=10)

    assert {row.origin for row in sink.orders} == {first_origin}
    retry = second_driver.handle(_origin(session, "2026-01-05"))
    assert retry.orders == ()
    assert sink.latest_origin == pd.Timestamp("2026-01-05")
    assert {row.origin for row in sink.forecasts} == {
        first_origin,
        pd.Timestamp("2026-01-05"),
    }
    assert {row.origin for row in sink.orders} == {first_origin}


def test_driver_retries_conflicts_and_settles_only_contiguous_eligible_periods() -> None:
    driver, session, sink, effects, order_origins = _runtime()
    initial = {"a": InventoryPosition(10.0, 0.0, 0.0)}
    first = _origin(session, "2026-01-04", positions=initial)

    first_outcome = driver.handle(first)
    effects_after_first = tuple(effects)
    orders_after_first = tuple(order_origins)
    durable_after_first = (sink.forecasts, sink.orders, sink.settlements)
    assert driver.handle(first) == first_outcome
    assert tuple(effects) == effects_after_first
    assert tuple(order_origins) == orders_after_first
    assert (sink.forecasts, sink.orders, sink.settlements) == durable_after_first

    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(_origin(session, "2026-01-04", positions=initial, scope=Scope.GLOBAL))
    with pytest.raises(EventDriverError, match="strictly monotonically"):
        driver.handle(_origin(session, "2026-01-03"))

    fourth = _actuals(session, "2026-01-04", 2.0)
    fourth_outcome = driver.handle(fourth)
    assert fourth_outcome.settlement_periods == (pd.Timestamp("2026-01-04"),)
    assert sink.settlement_snapshot((pd.Timestamp("2026-01-05"),)).frontier == pd.Timestamp(
        "2026-01-04"
    )
    durable_after_actual = (
        sink.forecasts,
        sink.observed_history,
        sink.settlements,
    )
    assert driver.handle(fourth) == fourth_outcome
    assert (sink.forecasts, sink.observed_history, sink.settlements) == durable_after_actual
    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(_actuals(session, "2026-01-04", 3.0))

    with pytest.raises(EventDriverError, match="rejected after durable settlement"):
        driver.handle(
            _origin(
                session,
                "2026-01-05",
                positions={"a": InventoryPosition(99.0, 0.0, 0.0)},
            )
        )
    driver.handle(_origin(session, "2026-01-05"))
    old_issuance = tuple(sink.forecasts)

    future = driver.handle(_actuals(session, "2026-01-06", 4.0))
    assert future.settlement_periods == ()
    assert tuple(sink.forecasts) == old_issuance
    fifth = driver.handle(_actuals(session, "2026-01-05", 3.0))
    assert fifth.settlement_periods == (pd.Timestamp("2026-01-05"),)

    driver.handle(_origin(session, "2026-01-06"))
    assert sink.settlement_snapshot((pd.Timestamp("2026-01-07"),)).frontier == pd.Timestamp(
        "2026-01-06"
    )
    assert tuple(record.period for record in sink.settlements) == tuple(
        pd.date_range("2026-01-04", "2026-01-06", freq="D")
    )
    assert len({record.key for record in sink.settlements}) == 3
    assert sum(record.realized_cost for record in sink.settlements) == pytest.approx(8.5)


def test_multi_record_actual_retry_is_order_independent_and_conflicts_atomically() -> None:
    driver, session, sink, _effects, _order_origins = _runtime()
    driver.handle(
        _origin(
            session,
            "2026-01-04",
            positions={"a": InventoryPosition(10.0, 0.0, 0.0)},
        )
    )
    first_record = ActualRecord("a", pd.Timestamp("2026-01-04"), 2.0)
    second_record = ActualRecord("a", pd.Timestamp("2026-01-05"), 3.0)
    event = ActualsEvent(session, ActualsSubmission((first_record, second_record)))

    first = driver.handle(event)
    durable = (sink.observed_history, sink.settlements, sink.pending_observations)
    retry = driver.handle(ActualsEvent(session, ActualsSubmission((second_record, first_record))))

    assert retry.receipt == first.receipt
    assert retry == first
    assert (sink.observed_history, sink.settlements, sink.pending_observations) == durable
    assert tuple(value.key for value in sink.observed_history) == (
        first_record.key,
        second_record.key,
    )
    assert first.settlement_periods == (first_record.timestamp,)

    conflict = ActualsEvent(
        session,
        ActualsSubmission(
            (
                first_record,
                ActualRecord("a", second_record.timestamp, 99.0),
            )
        ),
    )
    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(conflict)
    assert (sink.observed_history, sink.settlements, sink.pending_observations) == durable


def test_driver_rejects_wrong_session_and_off_calendar_actual_before_effects() -> None:
    driver, session, sink, effects, order_origins = _runtime()
    foreign = SessionIdentity.derive(
        tenant="foreign",
        series_keys=("a",),
        calendar=sink.calendar,
        horizon=1,
        model_config={"backend": "fixture"},
    )

    with pytest.raises(EventDriverError, match="session"):
        driver.handle(_origin(foreign, "2026-01-04"))
    with pytest.raises(ValueError, match="calendar"):
        driver.handle(_actuals(session, "2026-01-04 12:00:00", 1.0))

    assert effects == []
    assert order_origins == []
    assert sink.forecasts == sink.orders == sink.settlements == ()
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


def test_every_valid_actual_fact_and_origin_input_changes_its_fingerprint() -> None:
    from newcalibre.domain import CensoringAssertion

    session = _session()
    timestamp = pd.Timestamp("2026-01-01")
    baseline = ActualsEvent(
        session,
        ActualsSubmission((ActualRecord("a", timestamp, 1.0),)),
    )
    variants = (
        ActualRecord("b", timestamp, 1.0),
        ActualRecord("a", pd.Timestamp("2026-01-02"), 1.0),
        ActualRecord("a", timestamp, 2.0),
        ActualRecord("a", timestamp, 1.0, CensoringAssertion.CENSORED),
        ActualRecord("a", timestamp, 1.0, availability_bound=2.0),
    )
    assert (
        len(
            {
                baseline.fingerprint,
                *(
                    ActualsEvent(session, ActualsSubmission((record,))).fingerprint
                    for record in variants
                ),
            }
        )
        == len(variants) + 1
    )

    origin = pd.Timestamp("2026-01-03")
    initial = {"a": InventoryPosition(1.0, 0.0, 0.0)}
    first = OriginEvent(
        session=session,
        origin=origin,
        scope=Scope.LOCAL,
        future_exogenous=pd.DataFrame({"feature": pd.Series([1.0], dtype="float64")}),
        initial_inventory_positions=initial,
    )
    origin_variants = (
        OriginEvent(
            session=session,
            origin=pd.Timestamp("2026-01-04"),
            scope=Scope.LOCAL,
            future_exogenous=pd.DataFrame({"feature": pd.Series([1.0], dtype="float64")}),
            initial_inventory_positions=initial,
        ),
        OriginEvent(
            session=session,
            origin=origin,
            scope=Scope.GLOBAL,
            future_exogenous=pd.DataFrame({"feature": pd.Series([1.0], dtype="float64")}),
            initial_inventory_positions=initial,
        ),
        OriginEvent(
            session=session,
            origin=origin,
            scope=Scope.LOCAL,
            future_exogenous=pd.DataFrame({"feature": pd.Series([2.0], dtype="float64")}),
            initial_inventory_positions=initial,
        ),
        OriginEvent(
            session=session,
            origin=origin,
            scope=Scope.LOCAL,
            future_exogenous=pd.DataFrame({"other": pd.Series([1.0], dtype="float64")}),
            initial_inventory_positions=initial,
        ),
        OriginEvent(
            session=session,
            origin=origin,
            scope=Scope.LOCAL,
            future_exogenous=pd.DataFrame({"feature": pd.Series([1.0], dtype="float64")}),
            initial_inventory_positions={"a": InventoryPosition(2.0, 0.0, 0.0)},
        ),
    )
    assert len({first.fingerprint, *(value.fingerprint for value in origin_variants)}) == 6


def test_actual_event_requires_a_nonempty_submission_and_driver_rejects_other_types() -> None:
    session = _session()
    with pytest.raises(ValueError, match="must not be empty"):
        ActualsEvent(session, ActualsSubmission(()))

    driver, _session_value, _sink, _effects, _origins = _runtime()
    with pytest.raises(TypeError, match="only OriginEvent or ActualsEvent"):
        driver.handle(object())  # type: ignore[arg-type]


def test_event_driver_source_keeps_only_typed_domain_inputs_and_engine_verbs() -> None:
    import ast
    import inspect

    import newcalibre.engine.event_driver as event_module

    source = inspect.getsource(event_module)
    lowered = source.lower()
    forbidden = (
        "event_id",
        "correlation_id",
        "job_id",
        "queue_name",
        "http_request",
        "transport_version",
        "transport_metadata",
    )
    assert all(value not in lowered for value in forbidden)
    assert "newcalibre.forecasting" not in source
    assert "newcalibre.conformal" not in source
    assert "newcalibre.ordering" not in source

    tree = ast.parse(source)
    engine_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "_engine"
    }
    assert engine_calls == {"commit", "observe", "settle"}
    assert {"commit", "observe", "settle"} <= set(ENGINE_VERBS)
