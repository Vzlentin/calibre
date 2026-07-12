"""Lock the pure Gate-A settlement transition and accounting contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

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
    InventoryPosition,
    Panel,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine import (
    Engine,
    EngineError,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginCommit,
    SettlementError,
    SettlementRequest,
    SettlementSnapshot,
    settle,
)
from newcalibre.ledger import (
    BookedCost,
    LedgerError,
    OrderRow,
    SettlementRecord,
    StockoutTransition,
)

CALENDAR = Calendar("W-MON", phase=pd.Timestamp("2026-01-05"))
PERIODS = tuple(pd.date_range("2026-01-05", periods=8, freq="W-MON"))
TIMING = DecisionTiming(lead_time=2, review_period=3)
COST = CostStructure(underage=7.0, overage=3.0, holding=0.5, shortage=2.0)
MODEL_CONFIG = {"backend": "fixture", "name": "fixture"}
ORDERING_POLICY = {"name": "fixture"}


def _session(
    *,
    series_keys: Sequence[str] = ("sku-a",),
    tenant: str = "tenant-a",
    with_decision: bool = True,
    cost: CostStructure = COST,
    timing: DecisionTiming = TIMING,
    stockout_rule: StockoutRule = StockoutRule.LOST_SALES,
) -> SessionIdentity:
    decision = (
        {
            "ordering_policy": ORDERING_POLICY,
            "cost_structure": cost,
            "decision_timing": timing,
            "stockout_rule": stockout_rule,
        }
        if with_decision
        else {}
    )
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=series_keys,
        calendar=CALENDAR,
        horizon=3,
        model_config=MODEL_CONFIG,
        **decision,
    )


def _snapshot(
    session: SessionIdentity,
    *,
    periods: Sequence[pd.Timestamp],
    frontier: pd.Timestamp | None = None,
    latest_positions: Mapping[str, InventoryPosition] | None = None,
    open_order_quantities: Mapping[str, float] | None = None,
    due_arrivals: Mapping[tuple[str, pd.Timestamp], float] | None = None,
    actuals_semantics: ActualsSemantics | None = None,
    calendar: Calendar = CALENDAR,
) -> SettlementSnapshot:
    series_keys = tuple(
        sorted(
            (latest_positions or open_order_quantities or {"sku-a": 0.0}),
            key=lambda value: value.encode(),
        )
    )
    return SettlementSnapshot(
        session=session,
        calendar=calendar,
        periods=tuple(periods),
        frontier=frontier,
        latest_positions=latest_positions or {},
        open_order_quantities=(
            {series_key: 0.0 for series_key in series_keys}
            if open_order_quantities is None
            else open_order_quantities
        ),
        due_arrivals=due_arrivals or {},
        actuals_semantics=actuals_semantics,
    )


def _order(
    session: SessionIdentity,
    *,
    origin: pd.Timestamp,
    quantity: float,
    series_key: str = "sku-a",
    model_name: str = "policy",
    arrival_period: pd.Timestamp | None = None,
) -> OrderRow:
    return OrderRow(
        session=session,
        series_key=series_key,
        origin=origin,
        model_name=model_name,
        quantity=quantity,
        arrival_period=(
            CALENDAR.advance(origin, TIMING.lead_time) if arrival_period is None else arrival_period
        ),
    )


def _request(
    session: SessionIdentity,
    *,
    periods: Sequence[pd.Timestamp],
    actuals: Mapping[tuple[str, pd.Timestamp], float],
    positions: Mapping[str, InventoryPosition],
    snapshot: SettlementSnapshot | None = None,
    orders: Sequence[OrderRow] = (),
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> SettlementRequest:
    return SettlementRequest(
        session=session,
        snapshot=snapshot
        or _snapshot(
            session,
            periods=periods,
            open_order_quantities={
                series_key: position.on_order for series_key, position in positions.items()
            },
        ),
        actuals=actuals,
        inventory_positions=positions,
        orders=orders,
        actuals_semantics=actuals_semantics,
    )


def _zero_actuals(
    series_keys: Sequence[str],
    periods: Sequence[pd.Timestamp],
) -> dict[tuple[str, pd.Timestamp], float]:
    return {(series_key, period): 0.0 for period in periods for series_key in series_keys}


def _prior_settlement(
    session: SessionIdentity,
    *,
    period: pd.Timestamp,
    series_key: str = "sku-a",
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
    rule: StockoutRule = StockoutRule.LOST_SALES,
    holding_rate: float = COST.holding,
    arrivals: float = 0.0,
    closing_on_hand: float = 0.0,
) -> SettlementRecord:
    transition = StockoutTransition(
        rule=rule,
        demand=0.0,
        fulfilled_demand=0.0,
        unmet_demand=0.0,
        closing_on_hand=closing_on_hand,
        closing_backorders=0.0,
    )
    return SettlementRecord(
        session=session,
        series_key=series_key,
        period=period,
        arrivals=arrivals,
        actuals_semantics=actuals_semantics,
        transition=transition,
        inventory_position=InventoryPosition(closing_on_hand, 0.0, 0.0),
        holding=BookedCost(
            rate=holding_rate,
            basis=closing_on_hand,
            amount=holding_rate * closing_on_hand,
        ),
        shortage=BookedCost(rate=COST.shortage, basis=0.0, amount=0.0),
    )


def _panel(series_keys: Sequence[str]) -> Panel:
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(series_keys, dtype="string"),
                TIMESTAMP: pd.Series([PERIODS[0]] * len(series_keys), dtype="datetime64[ns]"),
                OBSERVED_VALUE: pd.Series([0.0] * len(series_keys), dtype="float64"),
            }
        ),
        calendar=CALENDAR,
    )


def _engine(
    session: SessionIdentity,
    *,
    series_keys: Sequence[str] = ("sku-a",),
) -> tuple[Engine, InMemoryLedgerSink]:
    panel = _panel(series_keys)
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    return (
        Engine(
            panel_source=InMemoryPanelSource(panel),
            actuals_source=InMemoryActualsSource(panel),
            artifact_store=InMemoryArtifactStore(),
            calibration_state_store=InMemoryCalibrationStateStore(),
            ledger_sink=sink,
            dispatch_backend=InProcessDispatch(),
        ),
        sink,
    )


def test_settlement_applies_arrivals_before_demand_and_books_traceable_costs() -> None:
    session = _session()
    result = settle(
        _request(
            session,
            periods=(PERIODS[2],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[2],),
                open_order_quantities={"sku-a": 7.0},
                due_arrivals={("sku-a", PERIODS[2]): 4.0},
            ),
            actuals={("sku-a", PERIODS[2]): 12.0},
            positions={"sku-a": InventoryPosition(5.0, 7.0, 0.0)},
        )
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record.arrivals == 4.0
    assert record.actuals_semantics is ActualsSemantics.DEMAND
    assert record.transition == StockoutTransition(
        rule=StockoutRule.LOST_SALES,
        demand=12.0,
        fulfilled_demand=9.0,
        unmet_demand=3.0,
        closing_on_hand=0.0,
        closing_backorders=0.0,
    )
    assert record.inventory_position == InventoryPosition(0.0, 3.0, 0.0)
    assert record.holding == BookedCost(rate=0.5, basis=0.0, amount=0.0)
    assert record.shortage == BookedCost(rate=2.0, basis=3.0, amount=6.0)
    assert record.realized_cost == 6.0
    assert result.inventory_positions == {"sku-a": InventoryPosition(0.0, 3.0, 0.0)}


def test_order_arrives_at_calendar_advance_once_and_remains_open_until_then() -> None:
    session = _session()
    window = PERIODS[1:4]
    result = settle(
        _request(
            session,
            periods=window,
            snapshot=_snapshot(
                session,
                periods=window,
                open_order_quantities={"sku-a": 4.0},
                due_arrivals={("sku-a", PERIODS[2]): 4.0},
            ),
            actuals=_zero_actuals(("sku-a",), window),
            positions={"sku-a": InventoryPosition(0.0, 4.0, 0.0)},
        )
    )

    assert [record.arrivals for record in result.records] == [0.0, 4.0, 0.0]
    assert [record.transition.closing_on_hand for record in result.records] == [0.0, 4.0, 4.0]
    assert result.inventory_positions["sku-a"] == InventoryPosition(4.0, 0.0, 0.0)


def test_order_placed_in_period_cannot_serve_that_period_demand() -> None:
    session = _session()
    order = _order(session, origin=PERIODS[0], quantity=3.0)
    window = PERIODS[0:3]
    actuals = _zero_actuals(("sku-a",), window)
    actuals[("sku-a", PERIODS[0])] = 2.0

    result = settle(
        _request(
            session,
            periods=window,
            orders=(order,),
            actuals=actuals,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )

    first, _, arrival = result.records
    assert first.arrivals == 0.0
    assert first.transition.unmet_demand == 2.0
    assert arrival.arrivals == 3.0
    assert result.inventory_positions["sku-a"] == InventoryPosition(3.0, 0.0, 0.0)


def test_multiple_orders_due_together_are_summed_without_duplication() -> None:
    session = _session()
    result = settle(
        _request(
            session,
            periods=(PERIODS[2], PERIODS[3]),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[2], PERIODS[3]),
                open_order_quantities={"sku-a": 7.0},
                due_arrivals={("sku-a", PERIODS[2]): 7.0},
            ),
            actuals=_zero_actuals(("sku-a",), PERIODS[2:4]),
            positions={"sku-a": InventoryPosition(0.0, 7.0, 0.0)},
        )
    )
    permuted = settle(
        _request(
            session,
            periods=(PERIODS[2], PERIODS[3]),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[2], PERIODS[3]),
                open_order_quantities={"sku-a": 7.0},
                due_arrivals={("sku-a", PERIODS[2]): 7.0},
            ),
            actuals=_zero_actuals(("sku-a",), PERIODS[2:4]),
            positions={"sku-a": InventoryPosition(0.0, 7.0, 0.0)},
        )
    )

    assert [record.arrivals for record in result.records] == [7.0, 0.0]
    assert result.inventory_positions["sku-a"] == InventoryPosition(7.0, 0.0, 0.0)
    assert result == permuted


def test_gapped_order_origins_use_calendar_arrivals_not_origin_positions() -> None:
    session = _session()
    orders = (
        _order(session, origin=PERIODS[0], quantity=5.0, model_name="early"),
        _order(session, origin=PERIODS[3], quantity=7.0, model_name="late"),
    )
    window = PERIODS[1:7]
    result = settle(
        _request(
            session,
            periods=window,
            snapshot=_snapshot(
                session,
                periods=window,
                open_order_quantities={"sku-a": 5.0},
                due_arrivals={("sku-a", PERIODS[2]): 5.0},
            ),
            orders=orders[1:],
            actuals=_zero_actuals(("sku-a",), window),
            positions={"sku-a": InventoryPosition(0.0, 5.0, 0.0)},
        )
    )

    assert [record.arrivals for record in result.records] == [0.0, 5.0, 0.0, 0.0, 7.0, 0.0]
    assert result.inventory_positions["sku-a"] == InventoryPosition(12.0, 0.0, 0.0)


def test_zero_order_window_still_settles_every_period_and_exact_demand_anchor() -> None:
    session = _session()
    window = PERIODS[0:3]
    result = settle(
        _request(
            session,
            periods=window,
            actuals={
                ("sku-a", PERIODS[0]): 1.0,
                ("sku-a", PERIODS[1]): 2.0,
                ("sku-a", PERIODS[2]): 8.0,
            },
            positions={"sku-a": InventoryPosition(6.0, 0.0, 0.0)},
        )
    )

    assert [record.period for record in result.records] == list(window)
    assert [record.transition.demand for record in result.records] == [1.0, 2.0, 8.0]
    assert [record.transition.closing_on_hand for record in result.records] == [5.0, 3.0, 0.0]
    assert [record.transition.unmet_demand for record in result.records] == [0.0, 0.0, 5.0]


def test_records_are_period_then_stable_series_order() -> None:
    series_keys = ("sku-b", "sku-a")
    session = _session(series_keys=series_keys)
    window = PERIODS[0:2]
    result = settle(
        _request(
            session,
            periods=window,
            actuals=_zero_actuals(series_keys, window),
            positions={
                "sku-b": InventoryPosition(0.0, 0.0, 0.0),
                "sku-a": InventoryPosition(0.0, 0.0, 0.0),
            },
        )
    )

    assert [(record.period, record.series_key) for record in result.records] == [
        (PERIODS[0], "sku-a"),
        (PERIODS[0], "sku-b"),
        (PERIODS[1], "sku-a"),
        (PERIODS[1], "sku-b"),
    ]


def test_settlement_preserves_nonempty_series_keys_verbatim() -> None:
    series_key = " sku-a "
    session = _session(series_keys=(series_key,))
    result = settle(
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={(series_key, PERIODS[0]): 0.0},
            positions={series_key: InventoryPosition(0.0, 0.0, 0.0)},
        )
    )

    assert result.records[0].series_key == series_key


def test_settlement_snapshot_cannot_carry_unrelated_history() -> None:
    session = _session()
    snapshot = _snapshot(
        session,
        periods=(PERIODS[1],),
        open_order_quantities={"sku-a": 0.0},
    )

    assert not hasattr(snapshot, "forecasts")
    assert not hasattr(snapshot, "orders")
    assert not hasattr(snapshot, "settlements")


@pytest.mark.parametrize("semantics", list(ActualsSemantics))
def test_actuals_semantics_is_carried_by_every_derived_row(
    semantics: ActualsSemantics,
) -> None:
    session = _session()
    result = settle(
        _request(
            session,
            periods=PERIODS[0:2],
            actuals={
                ("sku-a", PERIODS[0]): 1.0,
                ("sku-a", PERIODS[1]): 1.0,
            },
            positions={"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
            actuals_semantics=semantics,
        )
    )

    assert {record.actuals_semantics for record in result.records} == {semantics}


def test_semantics_label_changes_no_inventory_or_cost_arithmetic() -> None:
    session = _session()
    inputs = {
        "periods": (PERIODS[0],),
        "actuals": {("sku-a", PERIODS[0]): 2.0},
        "positions": {"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
    }
    demand = settle(_request(session, actuals_semantics=ActualsSemantics.DEMAND, **inputs))
    surrogate = settle(
        _request(
            session,
            actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
            **inputs,
        )
    )

    assert demand.inventory_positions == surrogate.inventory_positions
    assert demand.records[0].transition == surrogate.records[0].transition
    assert demand.records[0].holding == surrogate.records[0].holding
    assert demand.records[0].shortage == surrogate.records[0].shortage


def test_request_owns_inputs_and_settlement_is_deterministic() -> None:
    session = _session(series_keys=("sku-a", "sku-b"))
    actuals = {
        ("sku-b", PERIODS[0]): 2.0,
        ("sku-a", PERIODS[0]): 1.0,
    }
    positions = {
        "sku-b": InventoryPosition(3.0, 0.0, 0.0),
        "sku-a": InventoryPosition(3.0, 0.0, 0.0),
    }
    staged: list[OrderRow] = []
    request = _request(
        session,
        periods=(PERIODS[0],),
        actuals=actuals,
        positions=positions,
        orders=staged,
    )
    actuals[("sku-a", PERIODS[0])] = 99.0
    positions["sku-a"] = InventoryPosition(99.0, 0.0, 0.0)
    staged.append(_order(session, origin=PERIODS[0], quantity=99.0))

    first = settle(request)
    second = settle(request)
    permuted = settle(
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={
                ("sku-a", PERIODS[0]): 1.0,
                ("sku-b", PERIODS[0]): 2.0,
            },
            positions={
                "sku-a": InventoryPosition(3.0, 0.0, 0.0),
                "sku-b": InventoryPosition(3.0, 0.0, 0.0),
            },
        )
    )

    assert first == second == permuted
    assert tuple(first.inventory_positions) == ("sku-a", "sku-b")
    with pytest.raises(TypeError):
        request.actuals[("sku-a", PERIODS[0])] = 3.0  # type: ignore[index]
    with pytest.raises(TypeError):
        first.inventory_positions["sku-a"] = InventoryPosition(0.0, 0.0, 0.0)  # type: ignore[index]


@pytest.mark.parametrize("demand", [-1.0, float("nan"), float("inf")])
def test_nonfinite_or_negative_demand_is_refused(demand: float) -> None:
    session = _session()
    with pytest.raises(SettlementError, match="finite and non-negative"):
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={("sku-a", PERIODS[0]): demand},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_fractional_shortage_preserves_float_conservation() -> None:
    session = _session()
    result = settle(
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={("sku-a", PERIODS[0]): 457.8931265964433},
            positions={"sku-a": InventoryPosition(170.79992387036359, 0.0, 0.0)},
        )
    )

    transition = result.records[0].transition
    assert transition.fulfilled_demand == 170.79992387036359
    assert transition.closing_on_hand == 0.0
    assert transition.unmet_demand > 0.0


def test_demand_that_overflows_float_is_refused_as_nonfinite() -> None:
    session = _session()
    with pytest.raises(SettlementError, match="finite and non-negative"):
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={("sku-a", PERIODS[0]): 10**10000},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


@pytest.mark.parametrize("demand", [True, "1"])
def test_non_numeric_demand_is_refused(demand: object) -> None:
    session = _session()
    with pytest.raises(TypeError, match="must be a real number"):
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={("sku-a", PERIODS[0]): demand},  # type: ignore[dict-item]
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


@pytest.mark.parametrize(
    ("actuals", "match"),
    [
        ({}, "exactly match"),
        (
            {
                ("sku-a", PERIODS[0]): 0.0,
                ("sku-a", PERIODS[1]): 0.0,
            },
            "exactly match",
        ),
    ],
)
def test_missing_or_extra_demand_is_refused_before_simulation(
    actuals: Mapping[tuple[str, pd.Timestamp], float],
    match: str,
) -> None:
    session = _session()
    with pytest.raises(SettlementError, match=match):
        _request(
            session,
            periods=(PERIODS[0],),
            actuals=actuals,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_period_window_must_be_nonempty_contiguous_calendar_members() -> None:
    session = _session()
    positions = {"sku-a": InventoryPosition(0.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="must not be empty"):
        _request(session, periods=(), actuals={}, positions=positions)
    with pytest.raises(ValueError, match="does not lie on calendar"):
        _request(
            session,
            periods=(pd.Timestamp("2026-01-06"),),
            actuals={("sku-a", pd.Timestamp("2026-01-06")): 0.0},
            positions=positions,
        )
    with pytest.raises(ValueError, match="contiguous"):
        _request(
            session,
            periods=(PERIODS[0], PERIODS[2]),
            actuals=_zero_actuals(("sku-a",), (PERIODS[0], PERIODS[2])),
            positions=positions,
        )


def test_lost_sales_refuses_zero_lead_time_and_backorder_state_or_rule() -> None:
    session = _session()
    common = {
        "periods": (PERIODS[0],),
        "actuals": {("sku-a", PERIODS[0]): 0.0},
    }
    zero_lead_session = _session(
        timing=DecisionTiming(lead_time=0, review_period=1),
    )
    with pytest.raises(SettlementError, match="lead time must be at least one"):
        _request(
            zero_lead_session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
            **common,
        )
    with pytest.raises(SettlementError, match="zero opening backorders"):
        _request(
            session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 1.0)},
            **common,
        )
    backorder_session = _session(stockout_rule=StockoutRule.BACKORDER)
    with pytest.raises(SettlementError, match="unsupported.*backorder"):
        _request(
            backorder_session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
            **common,
        )
    with pytest.raises(TypeError, match="must be ActualsSemantics"):
        _request(
            session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
            actuals_semantics="demand",  # type: ignore[arg-type]
            **common,
        )


def test_arrival_law_and_open_order_inventory_are_single_authority() -> None:
    session = _session()
    forged = _order(
        session,
        origin=PERIODS[0],
        quantity=2.0,
        arrival_period=PERIODS[3],
    )
    with pytest.raises(SettlementError, match="must equal calendar.advance"):
        _request(
            session,
            periods=(PERIODS[0],),
            orders=(forged,),
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )

    with pytest.raises(SettlementError, match="does not match the compact ledger index"):
        _request(
            session,
            periods=(PERIODS[1],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[1],),
                open_order_quantities={"sku-a": 2.0},
            ),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_foreign_session_snapshot_is_refused_before_it_can_affect_arrivals() -> None:
    session = _session()
    other_session = _session(tenant="tenant-b")
    with pytest.raises(SettlementError, match="session must match"):
        _request(
            session,
            periods=(PERIODS[1],),
            snapshot=_snapshot(other_session, periods=(PERIODS[1],)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_staged_orders_must_be_current_window_facts_with_exact_arrivals() -> None:
    session = _session()
    common = {
        "periods": (PERIODS[0],),
        "actuals": {("sku-a", PERIODS[0]): 0.0},
        "positions": {"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
    }
    outside = _order(session, origin=PERIODS[1], quantity=1.0)
    with pytest.raises(SettlementError, match="origin must lie inside"):
        _request(session, orders=(outside,), **common)

    forged = _order(
        session,
        origin=PERIODS[0],
        quantity=1.0,
        arrival_period=PERIODS[1],
    )
    with pytest.raises(SettlementError, match="must equal calendar.advance"):
        _request(session, orders=(forged,), **common)

    duplicate = _order(session, origin=PERIODS[0], quantity=1.0)
    with pytest.raises(SettlementError, match="duplicate staged order"):
        _request(session, orders=(duplicate, duplicate), **common)

    foreign = _order(_session(tenant="tenant-b"), origin=PERIODS[0], quantity=1.0)
    with pytest.raises(SettlementError, match="session must match"):
        _request(session, orders=(foreign,), **common)


def test_inventory_positions_must_cover_the_session_series_exactly() -> None:
    session = _session(series_keys=("sku-a", "sku-b"))
    with pytest.raises(SettlementError, match="exactly match the session series set"):
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_settlement_window_and_opening_state_continue_the_ledger_frontier() -> None:
    session = _session()
    prior = _prior_settlement(session, period=PERIODS[0])
    with pytest.raises(SettlementError, match="immediately follow"):
        _request(
            session,
            periods=(PERIODS[2],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[2],),
                frontier=PERIODS[0],
                latest_positions={"sku-a": prior.inventory_position},
                actuals_semantics=ActualsSemantics.DEMAND,
            ),
            actuals={("sku-a", PERIODS[2]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )

    with pytest.raises(SettlementError, match="opening inventory"):
        _request(
            session,
            periods=(PERIODS[1],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[1],),
                frontier=PERIODS[0],
                latest_positions={"sku-a": prior.inventory_position},
                actuals_semantics=ActualsSemantics.DEMAND,
            ),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
        )

    continued = settle(
        _request(
            session,
            periods=(PERIODS[1],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[1],),
                frontier=PERIODS[0],
                latest_positions={"sku-a": prior.inventory_position},
                actuals_semantics=ActualsSemantics.DEMAND,
            ),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    assert continued.records[0].period == PERIODS[1]


def test_compact_snapshot_semantics_must_continue_the_durable_session() -> None:
    session = _session()
    prior = _prior_settlement(session, period=PERIODS[0])
    with pytest.raises(SettlementError, match="actuals semantics"):
        _request(
            session,
            periods=(PERIODS[1],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[1],),
                frontier=PERIODS[0],
                latest_positions={"sku-a": prior.inventory_position},
                actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
            ),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_ledger_calendar_frequency_must_match_the_session() -> None:
    session = _session()
    daily = Calendar("D", phase=PERIODS[0])
    ledger = _snapshot(
        session,
        periods=(PERIODS[0],),
        calendar=daily,
    )
    with pytest.raises(SettlementError, match="calendar must match"):
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=ledger,
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_any_already_booked_series_refuses_the_whole_window() -> None:
    session = _session(series_keys=("sku-a", "sku-b"))
    positions = {
        "sku-a": InventoryPosition(0.0, 0.0, 0.0),
        "sku-b": InventoryPosition(0.0, 0.0, 0.0),
    }
    with pytest.raises(SettlementError, match="immediately follow"):
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=_snapshot(
                session,
                periods=(PERIODS[0],),
                frontier=PERIODS[0],
                latest_positions=positions,
                open_order_quantities={"sku-a": 0.0, "sku-b": 0.0},
                actuals_semantics=ActualsSemantics.DEMAND,
            ),
            actuals=_zero_actuals(("sku-a", "sku-b"), (PERIODS[0],)),
            positions=positions,
        )


def test_session_without_decision_configuration_cannot_settle() -> None:
    session = _session(with_decision=False)
    with pytest.raises(SettlementError, match="no decision configuration"):
        _request(
            session,
            periods=(PERIODS[0],),
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_engine_uses_the_same_settlement_core_and_commit_is_exactly_once() -> None:
    session = _session()
    engine, sink = _engine(session)
    order = _order(session, origin=PERIODS[0], quantity=3.0)
    request = _request(
        session,
        periods=(PERIODS[0],),
        snapshot=sink.settlement_snapshot((PERIODS[0],)),
        orders=(order,),
        actuals={("sku-a", PERIODS[0]): 1.0},
        positions={"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
    )

    direct = settle(request)
    through_engine = engine.settle(request)
    assert through_engine == direct

    missing_order = OriginCommit(
        session=session,
        origin=PERIODS[0],
        settlements=through_engine.records,
    )
    with pytest.raises(LedgerError, match="on_order.*durable open orders"):
        engine.commit(missing_order)

    write = OriginCommit(
        session=session,
        origin=PERIODS[0],
        orders=(order,),
        settlements=through_engine.records,
    )
    receipt = engine.commit(write)
    assert engine.commit(write) == receipt
    assert sink.orders == (order,)
    assert sink.settlements == through_engine.records

    with pytest.raises(SettlementError, match="immediately follow"):
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=sink.settlement_snapshot((PERIODS[0],)),
            actuals={("sku-a", PERIODS[0]): 1.0},
            positions=through_engine.inventory_positions,
        )


def test_sink_reconciles_orders_across_same_write_and_later_settlement_windows() -> None:
    session = _session()
    order = _order(session, origin=PERIODS[0], quantity=3.0)

    same_write_engine, same_write_sink = _engine(session)
    same_write = same_write_engine.settle(
        _request(
            session,
            periods=PERIODS[0:3],
            snapshot=same_write_sink.settlement_snapshot(PERIODS[0:3]),
            orders=(order,),
            actuals=_zero_actuals(("sku-a",), PERIODS[0:3]),
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    same_write_engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[0],
            orders=(order,),
            settlements=same_write.records,
        )
    )
    assert [record.arrivals for record in same_write_sink.settlements] == [0.0, 0.0, 3.0]

    later_engine, later_sink = _engine(session)
    later_engine.commit(OriginCommit(session=session, origin=PERIODS[0], orders=(order,)))
    later = later_engine.settle(
        _request(
            session,
            periods=PERIODS[1:3],
            snapshot=later_sink.settlement_snapshot(PERIODS[1:3]),
            actuals=_zero_actuals(("sku-a",), PERIODS[1:3]),
            positions={"sku-a": InventoryPosition(0.0, 3.0, 0.0)},
        )
    )
    later_engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[2],
            settlements=later.records,
        )
    )
    assert [record.arrivals for record in later_sink.settlements] == [0.0, 3.0]


def test_sink_refuses_forged_timing_and_orders_for_already_settled_arrivals() -> None:
    session = _session()
    engine, sink = _engine(session)
    forged = _order(
        session,
        origin=PERIODS[0],
        quantity=2.0,
        arrival_period=PERIODS[1],
    )
    with pytest.raises(LedgerError, match="calendar.advance"):
        engine.commit(OriginCommit(session=session, origin=PERIODS[0], orders=(forged,)))
    unknown_series = _order(
        session,
        origin=PERIODS[0],
        quantity=2.0,
        series_key="sku-unknown",
    )
    with pytest.raises(LedgerError, match="series does not belong"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                orders=(unknown_series,),
            )
        )
    assert sink.orders == ()

    settled = engine.settle(
        _request(
            session,
            periods=(PERIODS[2],),
            snapshot=sink.settlement_snapshot((PERIODS[2],)),
            actuals={("sku-a", PERIODS[2]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[2],
            settlements=settled.records,
        )
    )

    late_repair = _order(session, origin=PERIODS[0], quantity=2.0)
    with pytest.raises(LedgerError, match="already settled"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                orders=(late_repair,),
            )
        )
    assert sink.orders == ()


def test_sink_refuses_orders_without_supported_session_timing() -> None:
    no_decision = _session(with_decision=False)
    no_decision_engine, no_decision_sink = _engine(no_decision)
    order = _order(no_decision, origin=PERIODS[0], quantity=1.0)
    with pytest.raises(LedgerError, match="decision configuration"):
        no_decision_engine.commit(
            OriginCommit(session=no_decision, origin=PERIODS[0], orders=(order,))
        )
    assert no_decision_sink.orders == ()

    zero_lead = _session(timing=DecisionTiming(lead_time=0, review_period=1))
    zero_lead_engine, zero_lead_sink = _engine(zero_lead)
    zero_lead_order = _order(
        zero_lead,
        origin=PERIODS[0],
        quantity=1.0,
        arrival_period=PERIODS[0],
    )
    with pytest.raises(LedgerError, match="positive decision lead time"):
        zero_lead_engine.commit(
            OriginCommit(
                session=zero_lead,
                origin=PERIODS[0],
                orders=(zero_lead_order,),
            )
        )
    assert zero_lead_sink.orders == ()

    backorder = _session(stockout_rule=StockoutRule.BACKORDER)
    backorder_engine, backorder_sink = _engine(backorder)
    backorder_order = _order(backorder, origin=PERIODS[0], quantity=1.0)
    with pytest.raises(LedgerError, match="stock-out rule is not supported"):
        backorder_engine.commit(
            OriginCommit(
                session=backorder,
                origin=PERIODS[0],
                orders=(backorder_order,),
            )
        )
    assert backorder_sink.orders == ()


def test_sink_refuses_an_order_whose_origin_is_already_settled() -> None:
    session = _session()
    engine, sink = _engine(session)
    settled = engine.settle(
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=sink.settlement_snapshot((PERIODS[0],)),
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[1],
            settlements=settled.records,
        )
    )
    late_order = _order(session, origin=PERIODS[0], quantity=5.0)

    with pytest.raises(LedgerError, match="order origin is already settled"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                orders=(late_order,),
            )
        )
    assert sink.orders == ()


def test_sink_refuses_settlement_metadata_and_transition_poisoning() -> None:
    session = _session()
    engine, sink = _engine(session)
    due = _order(session, origin=PERIODS[0], quantity=10.0)
    disappearing_arrival = _prior_settlement(
        session,
        period=PERIODS[2],
        arrivals=10.0,
    )
    with pytest.raises(LedgerError, match="continue durable inventory"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                orders=(due,),
                settlements=(disappearing_arrival,),
            )
        )
    assert sink.orders == ()
    assert sink.settlements == ()

    first = engine.settle(
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=sink.settlement_snapshot((PERIODS[0],)),
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    record = first.records[0]
    wrong_rule = replace(
        record,
        transition=replace(record.transition, rule=StockoutRule.BACKORDER),
    )
    with pytest.raises(LedgerError, match="stock-out rule"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                settlements=(wrong_rule,),
            )
        )

    wrong_rate = replace(
        record,
        holding=BookedCost(
            rate=9.0,
            basis=record.holding.basis,
            amount=9.0 * record.holding.basis,
        ),
    )
    with pytest.raises(LedgerError, match="cost rates"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                settlements=(wrong_rate,),
            )
        )
    assert sink.settlements == ()

    engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[0],
            settlements=first.records,
        )
    )
    second = engine.settle(
        _request(
            session,
            periods=(PERIODS[1],),
            snapshot=sink.settlement_snapshot((PERIODS[1],)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions=first.inventory_positions,
        )
    )
    changed_semantics = replace(
        second.records[0],
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    with pytest.raises(LedgerError, match="actuals semantics changed"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[1],
                settlements=(changed_semantics,),
            )
        )

    forged_transition = _prior_settlement(
        session,
        period=PERIODS[1],
        closing_on_hand=100.0,
    )
    with pytest.raises(LedgerError, match="continue durable inventory"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[1],
                settlements=(forged_transition,),
            )
        )
    assert sink.settlements == first.records


def test_sink_requires_complete_session_series_for_every_settlement_period() -> None:
    series_keys = ("sku-a", "sku-b")
    session = _session(series_keys=series_keys)
    engine, sink = _engine(session, series_keys=series_keys)
    result = engine.settle(
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=sink.settlement_snapshot((PERIODS[0],)),
            actuals=_zero_actuals(series_keys, (PERIODS[0],)),
            positions={series_key: InventoryPosition(0.0, 0.0, 0.0) for series_key in series_keys},
        )
    )

    with pytest.raises(LedgerError, match="complete session series"):
        engine.commit(
            OriginCommit(
                session=session,
                origin=PERIODS[0],
                settlements=(result.records[0],),
            )
        )
    assert sink.settlements == ()


def test_failed_multi_period_accounting_does_not_consume_open_orders() -> None:
    session = _session()
    engine, sink = _engine(session)
    first_order = _order(session, origin=PERIODS[0], quantity=3.0)
    second_order = _order(session, origin=PERIODS[1], quantity=4.0)
    engine.commit(OriginCommit(session=session, origin=PERIODS[0], orders=(first_order,)))
    engine.commit(OriginCommit(session=session, origin=PERIODS[1], orders=(second_order,)))
    result = engine.settle(
        _request(
            session,
            periods=PERIODS[2:4],
            snapshot=sink.settlement_snapshot(PERIODS[2:4]),
            actuals=_zero_actuals(("sku-a",), PERIODS[2:4]),
            positions={"sku-a": InventoryPosition(0.0, 7.0, 0.0)},
        )
    )
    bad_final = replace(
        result.records[-1],
        inventory_position=replace(
            result.records[-1].inventory_position,
            on_order=1.0,
        ),
    )
    bad_write = OriginCommit(
        session=session,
        origin=PERIODS[3],
        settlements=(result.records[0], bad_final),
    )

    with pytest.raises(LedgerError, match="on_order.*durable open orders"):
        engine.commit(bad_write)
    assert sink.settlements == ()

    engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[3],
            settlements=result.records,
        )
    )
    assert [record.arrivals for record in sink.settlements] == [3.0, 4.0]


def test_engine_refuses_a_settlement_snapshot_from_another_calendar_grid() -> None:
    session = _session()
    engine, _sink = _engine(session)
    foreign_calendar = Calendar("W-MON", phase=PERIODS[1])
    foreign_ledger = _snapshot(
        session,
        periods=(PERIODS[1],),
        calendar=foreign_calendar,
    )
    request = _request(
        session,
        periods=(PERIODS[1],),
        snapshot=foreign_ledger,
        actuals={("sku-a", PERIODS[1]): 0.0},
        positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
    )

    with pytest.raises(EngineError, match="calendar"):
        engine.settle(request)


def test_engine_refuses_forged_facts_on_the_owned_calendar_grid() -> None:
    session = _session()
    engine, sink = _engine(session)
    first = engine.settle(
        _request(
            session,
            periods=(PERIODS[0],),
            snapshot=sink.settlement_snapshot((PERIODS[0],)),
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    engine.commit(
        OriginCommit(
            session=session,
            origin=PERIODS[0],
            settlements=first.records,
        )
    )

    forged_position = InventoryPosition(100.0, 0.0, 0.0)
    forged_ledger = _snapshot(
        session,
        periods=(PERIODS[1],),
        frontier=PERIODS[0],
        latest_positions={"sku-a": forged_position},
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    forged_request = _request(
        session,
        periods=(PERIODS[1],),
        snapshot=forged_ledger,
        actuals={("sku-a", PERIODS[1]): 0.0},
        positions={"sku-a": forged_position},
    )

    with pytest.raises(EngineError, match="snapshot"):
        engine.settle(forged_request)


def test_compact_index_work_stays_flat_across_64_growing_origins_and_rebuild() -> None:
    calendar = Calendar("D", phase=pd.Timestamp("2026-01-01"))
    periods = tuple(pd.date_range("2026-01-01", periods=66, freq="D"))
    timing = DecisionTiming(lead_time=2, review_period=1)
    session = SessionIdentity.derive(
        tenant="tenant-prf-10",
        series_keys=("sku-a",),
        calendar=calendar,
        horizon=3,
        model_config=MODEL_CONFIG,
        ordering_policy=ORDERING_POLICY,
        cost_structure=COST,
        decision_timing=timing,
        stockout_rule=StockoutRule.LOST_SALES,
    )

    class NoHistoryReads:
        """Forward mutations while making any durable-family scan fail the witness."""

        def __init__(self, ledger) -> None:
            self._ledger = ledger

        def __getattr__(self, name):
            if name in {"forecasts", "orders", "settlements"}:
                raise AssertionError(f"settlement hot path scanned durable {name}")
            return getattr(self._ledger, name)

    def run(*, rebuild_after_32: bool):
        sink = InMemoryLedgerSink(session=session, calendar=calendar)
        positions = {"sku-a": InventoryPosition(0.0, 0.0, 0.0)}
        steady_work: list[tuple[int, ...]] = []
        durable_ledger = None
        for index, period in enumerate(periods[:64]):
            snapshot = sink.settlement_snapshot((period,))
            order = OrderRow(
                session=session,
                series_key="sku-a",
                origin=period,
                model_name="policy",
                quantity=1.0,
                arrival_period=calendar.advance(period, timing.lead_time),
            )
            result = settle(
                SettlementRequest(
                    session=session,
                    snapshot=snapshot,
                    actuals={("sku-a", period): 0.0},
                    inventory_positions=positions,
                    orders=(order,),
                    actuals_semantics=ActualsSemantics.DEMAND,
                )
            )
            sink.commit(
                OriginCommit(
                    session=session,
                    origin=period,
                    orders=(order,),
                    settlements=result.records,
                )
            )
            positions = dict(result.inventory_positions)
            audit = sink.settlement_index_audit()
            if index >= timing.lead_time:
                steady_work.append(
                    (
                        len(snapshot.due_arrivals),
                        audit.last_work.new_orders,
                        audit.last_work.settlement_records,
                        audit.last_work.due_orders,
                        audit.active_orders,
                        audit.due_buckets,
                    )
                )
            if rebuild_after_32 and index == 31:
                before = sink.settlement_snapshot((periods[32],))
                rebuilt = sink.rebuild_settlement_index()
                assert rebuilt.rebuild_work is not None
                assert (
                    rebuilt.rebuild_work.new_orders,
                    rebuilt.rebuild_work.settlement_records,
                    rebuilt.rebuild_work.due_orders,
                    rebuilt.active_orders,
                    rebuilt.due_buckets,
                ) == (32, 32, 30, 2, 2)
                assert sink.settlement_snapshot((periods[32],)) == before
            if index == 31:
                durable_ledger = sink._ledger
                sink._ledger = NoHistoryReads(durable_ledger)  # type: ignore[assignment]

        assert steady_work == [(1, 1, 1, 1, 2, 2)] * 62
        assert durable_ledger is not None
        sink._ledger = durable_ledger
        assert len(sink.orders) == 64
        assert len(sink.settlements) == 64
        final_snapshot = sink.settlement_snapshot((periods[64],))
        assert not hasattr(final_snapshot, "forecasts")
        assert not hasattr(final_snapshot, "orders")
        assert not hasattr(final_snapshot, "settlements")
        return sink, final_snapshot

    uninterrupted, uninterrupted_snapshot = run(rebuild_after_32=False)
    rebuilt, rebuilt_snapshot = run(rebuild_after_32=True)

    assert rebuilt.orders == uninterrupted.orders
    assert rebuilt.settlements == uninterrupted.settlements
    assert rebuilt_snapshot == uninterrupted_snapshot
