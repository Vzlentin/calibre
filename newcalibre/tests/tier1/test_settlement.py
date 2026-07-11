"""Lock the pure Gate-A settlement transition and accounting contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    InventoryPosition,
    Panel,
    SessionIdentity,
)
from newcalibre.engine import (
    ActualsSemantics,
    Engine,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
    LedgerSnapshot,
    OriginCommit,
    SettlementError,
    SettlementRequest,
    StockoutRule,
    settle,
)
from newcalibre.ledger import (
    BookedCost,
    Ledger,
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
) -> SessionIdentity:
    decision = {"ordering_policy": ORDERING_POLICY, "cost_structure": cost} if with_decision else {}
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
    forecasts=(),
    orders: Sequence[OrderRow] = (),
    settlements: Sequence[SettlementRecord] = (),
) -> LedgerSnapshot:
    return LedgerSnapshot(
        session=session,
        calendar=CALENDAR,
        forecasts=tuple(forecasts),
        orders=tuple(orders),
        settlements=tuple(settlements),
    )


def _order(
    session: SessionIdentity,
    *,
    origin: pd.Timestamp,
    quantity: float,
    series_key: str = "sku-a",
    model_name: str = "policy",
    arrival_period: pd.Timestamp | None = None,
    timing: DecisionTiming = TIMING,
) -> OrderRow:
    return OrderRow(
        session=session,
        series_key=series_key,
        origin=origin,
        model_name=model_name,
        quantity=quantity,
        arrival_period=(
            CALENDAR.advance(origin, timing.lead_time) if arrival_period is None else arrival_period
        ),
    )


def _request(
    session: SessionIdentity,
    *,
    periods: Sequence[pd.Timestamp],
    actuals: Mapping[tuple[str, pd.Timestamp], float],
    positions: Mapping[str, InventoryPosition],
    ledger: LedgerSnapshot | None = None,
    orders: Sequence[OrderRow] = (),
    timing: DecisionTiming = TIMING,
    stockout_rule: StockoutRule = StockoutRule.LOST_SALES,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> SettlementRequest:
    return SettlementRequest(
        session=session,
        periods=periods,
        ledger=ledger or _snapshot(session),
        actuals=actuals,
        inventory_positions=positions,
        orders=orders,
        timing=timing,
        stockout_rule=stockout_rule,
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
    actuals_semantics: str = ActualsSemantics.DEMAND.value,
    rule: str = StockoutRule.LOST_SALES.value,
    holding_rate: float = COST.holding,
) -> SettlementRecord:
    transition = StockoutTransition(
        rule=rule,
        demand=0.0,
        fulfilled_demand=0.0,
        unmet_demand=0.0,
        closing_on_hand=0.0,
        closing_backorders=0.0,
    )
    return SettlementRecord(
        session=session,
        series_key=series_key,
        period=period,
        arrivals=0.0,
        actuals_semantics=actuals_semantics,
        transition=transition,
        holding=BookedCost(rate=holding_rate, basis=0.0, amount=0.0),
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
    due = _order(session, origin=PERIODS[0], quantity=4.0)
    future = _order(session, origin=PERIODS[1], quantity=3.0, model_name="future")
    result = settle(
        _request(
            session,
            periods=(PERIODS[2],),
            ledger=_snapshot(session, orders=(due, future)),
            actuals={("sku-a", PERIODS[2]): 12.0},
            positions={"sku-a": InventoryPosition(5.0, 7.0, 0.0)},
        )
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record.arrivals == 4.0
    assert record.actuals_semantics == "demand"
    assert record.transition == StockoutTransition(
        rule="lost-sales",
        demand=12.0,
        fulfilled_demand=9.0,
        unmet_demand=3.0,
        closing_on_hand=0.0,
        closing_backorders=0.0,
    )
    assert record.holding == BookedCost(rate=0.5, basis=0.0, amount=0.0)
    assert record.shortage == BookedCost(rate=2.0, basis=3.0, amount=6.0)
    assert record.realized_cost == 6.0
    assert result.inventory_positions == {"sku-a": InventoryPosition(0.0, 3.0, 0.0)}


def test_order_arrives_at_calendar_advance_once_and_remains_open_until_then() -> None:
    session = _session()
    order = _order(session, origin=PERIODS[0], quantity=4.0)
    window = PERIODS[1:4]
    result = settle(
        _request(
            session,
            periods=window,
            ledger=_snapshot(session, orders=(order,)),
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
            ledger=_snapshot(session),
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
    orders = (
        _order(session, origin=PERIODS[0], quantity=2.5, model_name="policy-a"),
        _order(session, origin=PERIODS[0], quantity=4.5, model_name="policy-b"),
    )
    result = settle(
        _request(
            session,
            periods=(PERIODS[2], PERIODS[3]),
            ledger=_snapshot(session, orders=orders),
            actuals=_zero_actuals(("sku-a",), PERIODS[2:4]),
            positions={"sku-a": InventoryPosition(0.0, 7.0, 0.0)},
        )
    )
    permuted = settle(
        _request(
            session,
            periods=(PERIODS[2], PERIODS[3]),
            ledger=_snapshot(session, orders=tuple(reversed(orders))),
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
            ledger=_snapshot(session, orders=orders[:1]),
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


def test_forecast_history_cannot_change_settlement_arithmetic() -> None:
    session = _session()
    forecast_ledger = Ledger(session=session, calendar=CALENDAR)
    forecast_ledger.append_forecasts(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["sku-a"], dtype="string"),
                TARGET_TIMESTAMP: pd.Series([PERIODS[1]], dtype="datetime64[ns]"),
                ACTUAL_VALUE: pd.Series([float("nan")], dtype="float64"),
                POINT_FORECAST: pd.Series([999.0], dtype="float64"),
                HORIZON_STEP: pd.Series([2], dtype="int64"),
                ORIGIN: pd.Series([PERIODS[0]], dtype="datetime64[ns]"),
                MODEL_NAME: pd.Series(["irrelevant"], dtype="string"),
            }
        ),
        issuances={("sku-a", PERIODS[0], 2, "irrelevant"): {}},
    )
    inputs = {
        "periods": (PERIODS[1],),
        "actuals": {("sku-a", PERIODS[1]): 2.0},
        "positions": {"sku-a": InventoryPosition(3.0, 0.0, 0.0)},
    }

    empty = settle(_request(session, ledger=_snapshot(session), **inputs))
    with_forecast = settle(
        _request(
            session,
            ledger=_snapshot(session, forecasts=forecast_ledger.forecasts),
            **inputs,
        )
    )

    assert empty == with_forecast


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

    assert {record.actuals_semantics for record in result.records} == {semantics.value}


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
    with pytest.raises(SettlementError, match="must not be empty"):
        _request(session, periods=(), actuals={}, positions=positions)
    with pytest.raises(SettlementError, match="does not lie on calendar"):
        _request(
            session,
            periods=(pd.Timestamp("2026-01-06"),),
            actuals={("sku-a", pd.Timestamp("2026-01-06")): 0.0},
            positions=positions,
        )
    with pytest.raises(SettlementError, match="contiguous"):
        _request(
            session,
            periods=(PERIODS[0], PERIODS[2]),
            actuals=_zero_actuals(("sku-a",), (PERIODS[0], PERIODS[2])),
            positions=positions,
        )


def test_lost_sales_refuses_zero_lead_time_backorders_and_unknown_rules() -> None:
    session = _session()
    common = {
        "periods": (PERIODS[0],),
        "actuals": {("sku-a", PERIODS[0]): 0.0},
    }
    with pytest.raises(SettlementError, match="lead time must be at least one"):
        _request(
            session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
            timing=DecisionTiming(lead_time=0, review_period=1),
            **common,
        )
    with pytest.raises(SettlementError, match="zero opening backorders"):
        _request(
            session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 1.0)},
            **common,
        )
    with pytest.raises(ValueError, match="backorder"):
        StockoutRule("backorder")
    with pytest.raises(TypeError, match="must be a StockoutRule"):
        _request(
            session,
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
            stockout_rule="lost-sales",  # type: ignore[arg-type]
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
            periods=(PERIODS[1],),
            ledger=_snapshot(session, orders=(forged,)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 2.0, 0.0)},
        )

    valid = _order(session, origin=PERIODS[0], quantity=2.0)
    with pytest.raises(SettlementError, match="does not match ledger open orders"):
        _request(
            session,
            periods=(PERIODS[1],),
            ledger=_snapshot(session, orders=(valid,)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_foreign_session_ledger_rows_are_refused_before_they_can_affect_arrivals() -> None:
    session = _session()
    other_session = _session(tenant="tenant-b")
    foreign_order = _order(other_session, origin=PERIODS[0], quantity=2.0)
    with pytest.raises(SettlementError, match="order session must match"):
        _request(
            session,
            periods=(PERIODS[2],),
            ledger=_snapshot(session, orders=(foreign_order,)),
            actuals={("sku-a", PERIODS[2]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 2.0, 0.0)},
        )

    foreign_settlement = _prior_settlement(other_session, period=PERIODS[0])
    with pytest.raises(SettlementError, match="settlement session must match"):
        _request(
            session,
            periods=(PERIODS[1],),
            ledger=_snapshot(session, settlements=(foreign_settlement,)),
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
            ledger=_snapshot(session, settlements=(prior,)),
            actuals={("sku-a", PERIODS[2]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )

    with pytest.raises(SettlementError, match="opening on_hand"):
        _request(
            session,
            periods=(PERIODS[1],),
            ledger=_snapshot(session, settlements=(prior,)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
        )

    continued = settle(
        _request(
            session,
            periods=(PERIODS[1],),
            ledger=_snapshot(session, settlements=(prior,)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )
    )
    assert continued.records[0].period == PERIODS[1]


def test_prior_settlement_history_must_be_complete_and_contiguous() -> None:
    session = _session(series_keys=("sku-a", "sku-b"))
    incomplete = _prior_settlement(session, period=PERIODS[0], series_key="sku-a")
    with pytest.raises(SettlementError, match="every session series"):
        _request(
            session,
            periods=(PERIODS[1],),
            ledger=_snapshot(session, settlements=(incomplete,)),
            actuals=_zero_actuals(("sku-a", "sku-b"), (PERIODS[1],)),
            positions={
                "sku-a": InventoryPosition(0.0, 0.0, 0.0),
                "sku-b": InventoryPosition(0.0, 0.0, 0.0),
            },
        )

    one_series = _session()
    gapped = (
        _prior_settlement(one_series, period=PERIODS[0]),
        _prior_settlement(one_series, period=PERIODS[2]),
    )
    with pytest.raises(SettlementError, match="calendar-contiguous"):
        _request(
            one_series,
            periods=(PERIODS[3],),
            ledger=_snapshot(one_series, settlements=gapped),
            actuals={("sku-a", PERIODS[3]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


@pytest.mark.parametrize(
    ("prior", "match"),
    [
        (
            lambda session: _prior_settlement(
                session,
                period=PERIODS[0],
                actuals_semantics="unlabeled-surrogate",
            ),
            "actuals semantics",
        ),
        (
            lambda session: _prior_settlement(
                session,
                period=PERIODS[0],
                rule="backorder",
            ),
            "stock-out rule",
        ),
        (
            lambda session: _prior_settlement(
                session,
                period=PERIODS[0],
                holding_rate=9.0,
            ),
            "cost rates",
        ),
    ],
)
def test_prior_settlement_configuration_must_match_the_session_run(prior, match: str) -> None:
    session = _session()
    with pytest.raises(SettlementError, match=match):
        _request(
            session,
            periods=(PERIODS[1],),
            ledger=_snapshot(session, settlements=(prior(session),)),
            actuals={("sku-a", PERIODS[1]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_ledger_calendar_frequency_must_match_the_session() -> None:
    session = _session()
    daily = Calendar("D", phase=PERIODS[0])
    ledger = LedgerSnapshot(
        session=session,
        calendar=daily,
        forecasts=(),
        orders=(),
        settlements=(),
    )
    with pytest.raises(SettlementError, match="calendar must match"):
        _request(
            session,
            periods=(PERIODS[0],),
            ledger=ledger,
            actuals={("sku-a", PERIODS[0]): 0.0},
            positions={"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        )


def test_any_already_booked_series_refuses_the_whole_window() -> None:
    session = _session(series_keys=("sku-a", "sku-b"))
    prior = _prior_settlement(session, period=PERIODS[0], series_key="sku-b")
    with pytest.raises(SettlementError, match="already booked"):
        _request(
            session,
            periods=(PERIODS[0],),
            ledger=_snapshot(session, settlements=(prior,)),
            actuals=_zero_actuals(("sku-a", "sku-b"), (PERIODS[0],)),
            positions={
                "sku-a": InventoryPosition(0.0, 0.0, 0.0),
                "sku-b": InventoryPosition(0.0, 0.0, 0.0),
            },
        )


def test_session_without_decision_cost_structure_cannot_settle() -> None:
    session = _session(with_decision=False)
    with pytest.raises(SettlementError, match="no decision cost structure"):
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
        ledger=sink.snapshot(),
        orders=(order,),
        actuals={("sku-a", PERIODS[0]): 1.0},
        positions={"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
    )

    direct = settle(request)
    through_engine = engine.settle(request)
    assert through_engine == direct

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

    with pytest.raises(SettlementError, match="already booked"):
        _request(
            session,
            periods=(PERIODS[0],),
            ledger=sink.snapshot(),
            actuals={("sku-a", PERIODS[0]): 1.0},
            positions=through_engine.inventory_positions,
        )
