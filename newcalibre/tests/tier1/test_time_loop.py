"""Exercise the historical time loop over one transactional run store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionTiming,
    HierarchyIndex,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    StockoutRule,
    TargetSupport,
)
from newcalibre.engine import (
    ActualsCommit,
    CommitReceipt,
    Engine,
    EngineError,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    OrderProposal,
    OriginCommit,
    PhaseError,
    PhaseEvent,
    TimeLoop,
    TimeLoopError,
    TimeLoopRequest,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
MODEL_CONFIG = {"backend": "seasonal-naive", "m": 1}
COST_STRUCTURE = CostStructure(1.0, 1.0, 1.0, 1.0)
ORDERING_POLICY = {"name": "newsvendor"}
TIMING = DecisionTiming(lead_time=2, review_period=2)
ORIGINS = (
    pd.Timestamp("2026-01-04"),
    pd.Timestamp("2026-01-06"),
    pd.Timestamp("2026-01-09"),
)
SETTLEMENT_END = pd.Timestamp("2026-01-11")
INITIAL_POSITION = InventoryPosition(on_hand=100.0, on_order=0.0, backorders=0.0)


def _panel(
    values: Sequence[float | None],
    *,
    series_keys: Sequence[str] = ("a",),
) -> Panel:
    frozen = tuple(values)
    timestamps = tuple(pd.date_range("2026-01-01", periods=len(frozen), freq="D"))
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(
                    [key for key in series_keys for _value in frozen],
                    dtype="string",
                ),
                TIMESTAMP: pd.to_datetime(
                    [timestamp for _key in series_keys for timestamp in timestamps]
                ),
                OBSERVED_VALUE: pd.Series(
                    [value for _key in series_keys for value in frozen],
                    dtype="float64",
                ),
            }
        ),
        calendar=CALENDAR,
        target_support=TargetSupport.REAL,
    )


def _session(
    *,
    timing: DecisionTiming = TIMING,
    with_decision: bool = True,
    series_keys: tuple[str, ...] = ("a",),
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=timing.protection_period if with_decision else 1,
        model_config=MODEL_CONFIG,
        ordering_policy=ORDERING_POLICY if with_decision else None,
        decision_series_keys=series_keys if with_decision else None,
        cost_structure=COST_STRUCTURE if with_decision else None,
        decision_timing=timing if with_decision else None,
        stockout_rule=StockoutRule.LOST_SALES if with_decision else None,
    )


class _LostResponseRunStore(InMemoryIndexedRunStore):
    """Lose one selected response after its transaction is durable."""

    def __init__(self, *, fail_origin: pd.Timestamp, **keywords) -> None:
        super().__init__(**keywords)
        self._fail_origin = fail_origin
        self._failed = False

    def commit(self, write: OriginCommit | ActualsCommit) -> CommitReceipt:
        """Publish normally, then interrupt one caller response."""
        receipt = super().commit(write)
        if not self._failed and write.origin == self._fail_origin:
            self._failed = True
            raise RuntimeError("transactional commit response lost")
        return receipt


@dataclass(slots=True)
class Runtime:
    """Retain one engine/store graph and observable order callbacks."""

    engine: Engine
    store: InMemoryIndexedRunStore
    order_origins: list[pd.Timestamp]


def _runtime(
    *,
    session: SessionIdentity,
    forecast_panel: Panel,
    actuals_panel: Panel | None = None,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
    store: InMemoryIndexedRunStore | None = None,
    initial_arrivals: Mapping[tuple[str, pd.Timestamp], float] | None = None,
    enable_ordering: bool = True,
) -> Runtime:
    actuals_panel = forecast_panel if actuals_panel is None else actuals_panel
    run_store = store or InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=actuals_panel,
        actuals_semantics=actuals_semantics,
        initial_arrivals=initial_arrivals,
    )
    order_origins: list[pd.Timestamp] = []

    def order(request) -> tuple[OrderProposal, ...]:
        order_origins.append(request.origin)
        return (OrderProposal("a", "seasonal-naive", 4.0),)

    engine = Engine(
        session=session,
        panel_source=InMemoryPanelSource(forecast_panel),
        run_store=run_store,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(forecast_panel.series_keys),
        orderer=order if enable_ordering else None,
    )
    return Runtime(engine, run_store, order_origins)


def _request(
    session: SessionIdentity,
    *,
    origins: Sequence[pd.Timestamp] = ORIGINS,
    settlement_end: pd.Timestamp = SETTLEMENT_END,
    positions: Mapping[str, InventoryPosition] | None = None,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> TimeLoopRequest:
    return TimeLoopRequest(
        session=session,
        origins=origins,
        settlement_end=settlement_end,
        scope=Scope.LOCAL,
        initial_inventory_positions={"a": INITIAL_POSITION} if positions is None else positions,
        actuals_semantics=actuals_semantics,
    )


def _loop(runtime: Runtime, request: TimeLoopRequest, *, reporter=None) -> TimeLoop:
    return TimeLoop(
        engine=runtime.engine,
        run_store=runtime.store,
        request=request,
        reporter=reporter,
    )


def _assert_no_effects(runtime: Runtime) -> None:
    assert runtime.order_origins == []
    assert runtime.store.checkpoints == {}
    assert runtime.store.states == {}
    assert runtime.store.forecasts == ()
    assert runtime.store.orders == ()
    assert runtime.store.settlements == ()
    assert runtime.store.revision == 1


def test_request_requires_explicit_actuals_semantics() -> None:
    """Reject an unlabeled observation contract before constructing the loop."""
    session = _session()
    with pytest.raises(TypeError, match="actuals_semantics"):
        TimeLoopRequest(
            session=session,
            origins=ORIGINS,
            settlement_end=SETTLEMENT_END,
            scope=Scope.LOCAL,
            initial_inventory_positions={"a": INITIAL_POSITION},
        )  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("settlement_end", "match"),
    [
        (pd.Timestamp("2026-01-09 12:00:00"), "calendar|grid|member"),
        (pd.Timestamp("2026-01-08"), "last origin"),
        (pd.Timestamp("2026-01-10"), "final decision.*lead"),
    ],
)
def test_constructor_rejects_invalid_settlement_end_before_effects(
    settlement_end: pd.Timestamp,
    match: str,
) -> None:
    """Validate the full drain window before model or store publication."""
    session = _session()
    runtime = _runtime(session=session, forecast_panel=_panel(range(1, 13)))
    with pytest.raises(TimeLoopError, match=match):
        _loop(runtime, _request(session, settlement_end=settlement_end))
    _assert_no_effects(runtime)


@pytest.mark.parametrize(
    "origins",
    [(), (ORIGINS[0], ORIGINS[0]), (ORIGINS[1], ORIGINS[0])],
)
def test_request_rejects_empty_duplicate_and_decreasing_origins(
    origins: tuple[pd.Timestamp, ...],
) -> None:
    """Require a nonempty increasing decision schedule."""
    with pytest.raises(TimeLoopError):
        _request(_session(), origins=origins)


def test_constructor_rejects_off_grid_and_invalid_inventory_before_effects() -> None:
    """Validate calendar and decision-series identity before opening work."""
    session = _session()
    runtime = _runtime(session=session, forecast_panel=_panel(range(1, 13)))
    with pytest.raises(TimeLoopError, match="calendar|grid|member"):
        _loop(
            runtime,
            _request(session, origins=(pd.Timestamp("2026-01-04 12:00:00"),)),
        )
    with pytest.raises(TimeLoopError, match="exactly match the decision series"):
        _loop(runtime, _request(session, positions={"other": INITIAL_POSITION}))
    _assert_no_effects(runtime)


def test_loop_rejects_a_store_not_owned_by_the_engine() -> None:
    """Require engine and driver to share the exact transactional store instance."""
    session = _session(with_decision=False)
    panel = _panel(range(1, 13))
    runtime = _runtime(
        session=session,
        forecast_panel=panel,
        enable_ordering=False,
    )
    foreign = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    with pytest.raises(EngineError, match="run store does not belong"):
        TimeLoop(
            engine=runtime.engine,
            run_store=foreign,
            request=_request(session, positions={}),
        )
    _assert_no_effects(runtime)


def test_decision_free_loop_runs_origins_and_closes_observations() -> None:
    """Commit forecasts and the close cycle without inventory settlement."""
    session = _session(with_decision=False)
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(1, 13)),
        enable_ordering=False,
    )
    events: list[PhaseEvent] = []
    result = _loop(runtime, _request(session, positions={}), reporter=events.append).run()

    expected_commits = tuple(pd.date_range(ORIGINS[0], SETTLEMENT_END, freq="D")) + (
        pd.Timestamp("2026-01-12"),
    )
    assert result.settlement_periods == ()
    assert result.inventory_positions == {}
    assert tuple(receipt.origin for receipt in result.receipts) == expected_commits
    assert len(runtime.store.forecasts) == len(ORIGINS)
    assert len(runtime.store.observation_resolutions) == len(ORIGINS)
    assert runtime.store.pending_observations == ()
    assert runtime.store.orders == runtime.store.settlements == ()
    assert len(events) == 6 * len(ORIGINS)


def test_gapped_origins_use_sequence_cadence_and_settle_through_drain() -> None:
    """Settle every calendar period while ordering only on review origins."""
    session = _session()
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 113)),
        actuals_panel=_panel(range(1, 13)),
    )
    result = _loop(runtime, _request(session)).run()

    expected_periods = tuple(pd.date_range(ORIGINS[0], SETTLEMENT_END, freq="D"))
    assert result.settlement_periods == expected_periods
    assert result.decision_origins == (ORIGINS[0], ORIGINS[2])
    assert tuple(runtime.order_origins) == result.decision_origins
    assert tuple(record.period for record in runtime.store.settlements) == expected_periods
    assert tuple(order.arrival_period for order in runtime.store.orders) == (
        CALENDAR.advance(ORIGINS[0], TIMING.lead_time),
        CALENDAR.advance(ORIGINS[2], TIMING.lead_time),
    )
    assert [record.transition.demand for record in runtime.store.settlements] == [
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
    ]
    assert all(row.actual_value is None for row in runtime.store.forecasts)
    assert runtime.store.observation_resolutions


def test_surrogate_semantics_labels_rows_without_changing_arithmetic() -> None:
    """Carry explicit observation semantics through every settlement record."""
    session = _session()
    panel = _panel(range(1, 13))
    demand = _runtime(session=session, forecast_panel=panel)
    surrogate = _runtime(
        session=session,
        forecast_panel=panel,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    demand_result = _loop(demand, _request(session)).run()
    surrogate_result = _loop(
        surrogate,
        _request(
            session,
            actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
        ),
    ).run()

    assert demand_result.inventory_positions == surrogate_result.inventory_positions
    assert [row.realized_cost for row in demand.store.settlements] == [
        row.realized_cost for row in surrogate.store.settlements
    ]
    assert {row.actuals_semantics for row in surrogate.store.settlements} == {
        ActualsSemantics.CENSORED_SALES_SURROGATE
    }


@pytest.mark.parametrize("with_decision", [False, True])
def test_loop_refuses_semantics_that_do_not_match_the_store(with_decision: bool) -> None:
    """Prevent a fresh session from relabeling store-owned observations."""
    session = _session(with_decision=with_decision)
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(1, 13)),
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    with pytest.raises(TimeLoopError, match="semantics"):
        _loop(runtime, _request(session, positions={} if not with_decision else None))
    _assert_no_effects(runtime)


def test_missing_settlement_actual_fails_without_publication() -> None:
    """Reject an incomplete decision-series settlement window atomically."""
    session = _session()
    values: list[float | None] = [float(value) for value in range(1, 13)]
    values[ORIGINS[0].day - 1] = None
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(101, 113)),
        actuals_panel=_panel(values),
    )
    with pytest.raises(TimeLoopError, match="missing"):
        _loop(runtime, _request(session))
    _assert_no_effects(runtime)


def test_constructor_requires_initial_on_order_to_match_seeded_arrivals() -> None:
    """Bind opening on-order inventory to the store-owned arrival index."""
    session = _session()
    runtime = _runtime(
        session=session,
        forecast_panel=_panel(range(1, 13)),
        initial_arrivals={
            ("a", ORIGINS[0]): 2.0,
            ("a", CALENDAR.advance(ORIGINS[0], 1)): 3.0,
        },
    )
    with pytest.raises(TimeLoopError, match="does not match the compact ledger index"):
        _loop(
            runtime,
            _request(
                session,
                positions={"a": InventoryPosition(10.0, 4.0, 0.0)},
            ),
        )
    _assert_no_effects(runtime)


@pytest.mark.parametrize("fail_origin", [ORIGINS[0], pd.Timestamp("2026-01-05")])
def test_reconstructed_loop_repairs_a_lost_commit_response(
    fail_origin: pd.Timestamp,
) -> None:
    """Resume from the exact durable receipt without rebooking rows or checkpoints."""
    session = _session()
    forecast_panel = _panel(range(101, 113))
    actuals_panel = _panel(range(1, 13))
    baseline = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals_panel=actuals_panel,
    )
    expected = _loop(baseline, _request(session)).run()

    interrupted_store = _LostResponseRunStore(
        fail_origin=fail_origin,
        session=session,
        calendar=CALENDAR,
        actuals=actuals_panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    interrupted = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals_panel=actuals_panel,
        store=interrupted_store,
    )
    with pytest.raises((PhaseError, RuntimeError), match="response lost"):
        _loop(interrupted, _request(session)).run()

    resumed = _runtime(
        session=session,
        forecast_panel=forecast_panel,
        actuals_panel=actuals_panel,
        store=interrupted_store,
    )
    result = _loop(resumed, _request(session)).run()

    assert result.inventory_positions == expected.inventory_positions
    assert interrupted_store.forecasts == baseline.store.forecasts
    assert interrupted_store.observation_resolutions == baseline.store.observation_resolutions
    assert interrupted_store.orders == baseline.store.orders
    assert interrupted_store.settlements == baseline.store.settlements
    assert interrupted_store.states == baseline.store.states
    assert interrupted_store.checkpoints == baseline.store.checkpoints
    assert interrupted_store.checkpoint_indexes == baseline.store.checkpoint_indexes
