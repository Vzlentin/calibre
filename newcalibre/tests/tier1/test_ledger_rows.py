"""Exercise immutable row families and append behavior through the ledger interface."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    ActualsSemantics,
    Calendar,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    InventoryPosition,
    ScoredSeries,
    SessionIdentity,
    StockoutRule,
    interval_columns,
    quantile_column,
)
from newcalibre.ledger import (
    BookedCost,
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    ForecastRow,
    GuaranteedSide,
    Ledger,
    LedgerError,
    OrderRow,
    SettlementRecord,
    StockoutTransition,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
ISSUE_ORIGIN = pd.Timestamp("2026-01-05")
ARRIVAL = pd.Timestamp("2026-01-07")


def _session(*, tenant: str = "tenant-a") -> SessionIdentity:
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=("sku-a", "sku-b"),
        calendar=CALENDAR,
        horizon=2,
        model_config={"name": "seasonal-naive"},
    )


def _descriptor(
    *,
    claim: GuaranteeClaim = GuaranteeClaim.NONE,
    level: float = 0.5,
) -> GuaranteeDescriptor:
    currency = None if claim is GuaranteeClaim.NONE else GuaranteeCurrency.FINITE_SAMPLE_MARGINAL
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=claim,
            currency=currency,
            declared_slack=None,
        ),
        level=level,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )


def _issuance(
    *,
    finite: bool = True,
    reason: str | None = None,
    claim: GuaranteeClaim = GuaranteeClaim.NONE,
    side: GuaranteedSide | None = None,
    level: float = 0.5,
    ready: bool = False,
) -> ForecastIssuance:
    return ForecastIssuance(
        descriptor=_descriptor(claim=claim, level=level),
        guaranteed_side=side,
        calibration_ready=ready,
        bounds_finite=finite,
        bounds_null_reason=reason,
    )


def _frame(*, actual: float | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-a", "sku-b"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(["2026-01-05", "2026-01-06"]),
            ACTUAL_VALUE: pd.Series([actual, actual], dtype="float64"),
            POINT_FORECAST: pd.Series([10.0, 20.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1, 2], dtype="int64"),
            ORIGIN: pd.to_datetime([ISSUE_ORIGIN, ISSUE_ORIGIN]),
            MODEL_NAME: pd.Series(["seasonal", "seasonal"], dtype="string"),
            quantile_column(0.5): pd.Series([10.0, 20.0], dtype="float64"),
            "adapter_note": pd.Series(["first", "second"], dtype="string"),
        }
    )


def _forecast_key(series: str, step: int) -> ForecastKey:
    return (series, ISSUE_ORIGIN, step, "seasonal")


def _issuances() -> dict[ForecastKey, dict[BoundKey, ForecastIssuance]]:
    quantile: BoundKey = (quantile_column(0.5),)
    return {
        _forecast_key("sku-a", 1): {quantile: _issuance()},
        _forecast_key("sku-b", 2): {quantile: _issuance()},
    }


def _order(
    *,
    series: str = "sku-a",
    origin: pd.Timestamp = ISSUE_ORIGIN,
    arrival: pd.Timestamp = ARRIVAL,
    quantity: float = 3.0,
    session: SessionIdentity | None = None,
) -> OrderRow:
    return OrderRow(
        session=session or _session(),
        series_key=series,
        origin=origin,
        model_name="seasonal",
        quantity=quantity,
        arrival_period=arrival,
    )


def _settlement(
    *,
    period: pd.Timestamp = ARRIVAL,
    session: SessionIdentity | None = None,
    holding_basis: float = 2.0,
    shortage_basis: float = 1.0,
) -> SettlementRecord:
    return SettlementRecord(
        session=session or _session(),
        series_key="sku-a",
        period=period,
        arrivals=3.0,
        actuals_semantics=ActualsSemantics.DEMAND,
        transition=StockoutTransition(
            rule=StockoutRule.LOST_SALES,
            demand=5.0,
            fulfilled_demand=4.0,
            unmet_demand=1.0,
            closing_on_hand=2.0,
            closing_backorders=0.0,
        ),
        inventory_position=InventoryPosition(
            on_hand=2.0,
            on_order=0.0,
            backorders=0.0,
        ),
        holding=BookedCost(
            rate=0.5,
            basis=holding_basis,
            amount=0.5 * holding_basis,
        ),
        shortage=BookedCost(
            rate=2.0,
            basis=shortage_basis,
            amount=2.0 * shortage_basis,
        ),
    )


def test_public_ledger_round_trips_all_three_row_families_in_append_order() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    ledger.append_forecasts(_frame(), issuances=_issuances())
    ledger.append_orders([_order(quantity=0.0), _order(series="sku-b", quantity=4.0)])
    ledger.append_settlements([_settlement()])

    assert [row.key for row in ledger.forecasts] == [
        _forecast_key("sku-a", 1),
        _forecast_key("sku-b", 2),
    ]
    assert ledger.forecasts[0].point_forecast == 10.0
    assert ledger.forecasts[0].actual_value is None
    assert ledger.forecasts[0].values["adapter_note"] == "first"
    issuance = ledger.forecasts[0].issuances[(quantile_column(0.5),)]
    assert issuance.descriptor == _descriptor()
    assert issuance.calibration_ready is False
    assert issuance.bounds_finite is True
    assert issuance.bounds_null_reason is None
    assert [row.quantity for row in ledger.orders] == [0.0, 4.0]
    assert ledger.orders[0].key == (_session(), "sku-a", ISSUE_ORIGIN, "seasonal")
    assert ledger.settlements == (_settlement(),)
    assert ledger.settlements[0].key == (_session(), "sku-a", ARRIVAL)


def test_forecast_append_requires_pending_rows_and_exact_issuance_keys() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    with pytest.raises(LedgerError, match="pending"):
        ledger.append_forecasts(_frame(actual=1.0), issuances=_issuances())
    assert ledger.forecasts == ()

    missing = _issuances()
    del missing[_forecast_key("sku-b", 2)]
    with pytest.raises(LedgerError, match="issuance keys"):
        ledger.append_forecasts(_frame(), issuances=missing)

    extra = _issuances()
    extra[_forecast_key("sku-a", 2)] = {cast(BoundKey, (quantile_column(0.5),)): _issuance()}
    with pytest.raises(LedgerError, match="issuance keys"):
        ledger.append_forecasts(_frame(), issuances=extra)
    assert ledger.forecasts == ()


def test_forecast_append_is_atomic_and_rejects_existing_keys() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(_frame(), issuances=_issuances())
    before = ledger.forecasts

    with pytest.raises(LedgerError, match="duplicate forecast key"):
        ledger.append_forecasts(_frame(), issuances=_issuances())

    assert ledger.forecasts == before


def test_forecast_rows_snapshot_the_frame_and_are_deeply_immutable() -> None:
    frame = _frame()
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(frame, issuances=_issuances())
    row = ledger.forecasts[0]

    frame.loc[0, "point_forecast"] = 999.0
    assert row.point_forecast == 10.0
    with pytest.raises(TypeError):
        cast(Any, row.values)["point_forecast"] = 999.0
    with pytest.raises(FrozenInstanceError):
        cast(Any, row).issuances = {}
    with pytest.raises(TypeError):
        cast(Any, row.issuances)[(quantile_column(0.5),)] = _issuance()
    with pytest.raises(TypeError, match="Ledger.append_forecasts"):
        ForecastRow()


def test_forecast_issuance_fact_is_closed_and_matches_the_bound_payload() -> None:
    with pytest.raises(LedgerError, match="null reason"):
        _issuance(finite=False, reason=None)
    with pytest.raises(LedgerError, match="null reason"):
        _issuance(finite=False, reason=" ")
    with pytest.raises(LedgerError, match="null reason"):
        _issuance(finite=True, reason="warm-up")
    with pytest.raises(LedgerError, match="guaranteed side"):
        _issuance(
            finite=False,
            reason="warm-up",
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
        )
    with pytest.raises(LedgerError, match="only one-sided"):
        _issuance(side=GuaranteedSide.UPPER)
    with pytest.raises(LedgerError, match="readiness"):
        _issuance(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            side=GuaranteedSide.UPPER,
        )

    ledger = Ledger(session=_session(), calendar=CALENDAR)
    quantile: BoundKey = (quantile_column(0.5),)
    inconsistent = {
        key: {quantile: _issuance(finite=False, reason="warm-up")} for key in _issuances()
    }
    with pytest.raises(LedgerError, match="finiteness"):
        ledger.append_forecasts(_frame(), issuances=inconsistent)

    unavailable = _frame()
    unavailable[quantile_column(0.5)] = float("nan")
    ledger.append_forecasts(
        unavailable,
        issuances={
            key: {quantile: _issuance(finite=False, reason="calibration warm-up")}
            for key in _issuances()
        },
    )
    assert [row.issuances[quantile].bounds_null_reason for row in ledger.forecasts] == [
        "calibration warm-up",
        "calibration warm-up",
    ]


def test_forecast_bound_issuances_account_for_multiple_groups_and_sides() -> None:
    frame = _frame()
    quantile_09: BoundKey = (quantile_column(0.9),)
    quantile_05: BoundKey = (quantile_column(0.5),)
    interval_08: BoundKey = interval_columns(0.8)
    interval_095: BoundKey = interval_columns(0.95)
    frame[quantile_09[0]] = pd.Series([12.0, 22.0], dtype="float64")
    frame[interval_08[0]] = pd.Series([7.0, 17.0], dtype="float64")
    frame[interval_08[1]] = pd.Series([13.0, 23.0], dtype="float64")
    frame[interval_095[0]] = pd.Series([6.0, 16.0], dtype="float64")
    frame[interval_095[1]] = pd.Series([14.0, 24.0], dtype="float64")
    row_issuances: dict[BoundKey, ForecastIssuance] = {
        quantile_09: _issuance(level=0.9),
        quantile_05: _issuance(),
        interval_08: _issuance(
            claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
            level=0.8,
            ready=True,
        ),
        (interval_095[0],): _issuance(
            level=0.95,
        ),
        (interval_095[1],): _issuance(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            side=GuaranteedSide.UPPER,
            level=0.95,
            ready=True,
        ),
    }
    issuances = {key: row_issuances for key in _issuances()}
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    ledger.append_forecasts(frame, issuances=issuances)

    assert tuple(ledger.forecasts[0].issuances) == tuple(row_issuances)
    assert ledger.forecasts[0].issuances[interval_08].descriptor.level == 0.8
    assert ledger.forecasts[0].issuances[(interval_095[1],)].guaranteed_side is GuaranteedSide.UPPER

    incomplete = {key: dict(row_issuances) for key in _issuances()}
    del incomplete[_forecast_key("sku-b", 2)][quantile_09]
    empty_ledger = Ledger(session=_session(), calendar=CALENDAR)
    with pytest.raises(LedgerError, match="exactly account"):
        empty_ledger.append_forecasts(frame, issuances=incomplete)
    assert empty_ledger.forecasts == ()

    missing_column = {key: dict(row_issuances) for key in _issuances()}
    for row_facts in missing_column.values():
        del row_facts[quantile_09]
        row_facts[(quantile_column(0.7),)] = _issuance(level=0.7)
    with pytest.raises(LedgerError, match="missing or non-bound"):
        empty_ledger.append_forecasts(frame, issuances=missing_column)
    assert empty_ledger.forecasts == ()


def test_risk_control_level_is_independent_of_the_forecast_column_level() -> None:
    quantile: BoundKey = (quantile_column(0.5),)
    risk_issuance = _issuance(
        claim=GuaranteeClaim.RISK_CONTROL,
        level=0.1,
        ready=True,
    )
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    ledger.append_forecasts(
        _frame(),
        issuances={key: {quantile: risk_issuance} for key in _issuances()},
    )

    assert ledger.forecasts[0].issuances[quantile].descriptor.level == 0.1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_forecast_point_must_be_finite(invalid: float) -> None:
    frame = _frame()
    frame.loc[1, POINT_FORECAST] = invalid
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    with pytest.raises(LedgerError, match="point forecast.*finite"):
        ledger.append_forecasts(frame, issuances=_issuances())
    assert ledger.forecasts == ()


def test_forecast_append_late_bound_failure_is_atomic_and_reusable() -> None:
    frame = _frame()
    quantile: BoundKey = (quantile_column(0.5),)
    issuances = _issuances()
    issuances[_forecast_key("sku-b", 2)][quantile] = _issuance(
        finite=False,
        reason="calibration warm-up",
    )
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    with pytest.raises(LedgerError, match="finiteness"):
        ledger.append_forecasts(frame, issuances=issuances)
    assert ledger.forecasts == ()

    ledger.append_forecasts(frame, issuances=_issuances())
    assert [row.key for row in ledger.forecasts] == [
        _forecast_key("sku-a", 1),
        _forecast_key("sku-b", 2),
    ]


def test_ledger_binds_one_calendar_and_preserves_literal_identifiers() -> None:
    with pytest.raises(LedgerError, match="bound Calendar"):
        Ledger(session=_session(), calendar=Calendar("D"))

    frame = _frame().iloc[[0]].copy()
    frame.loc[frame.index[0], SERIES_KEY] = " "
    frame.loc[frame.index[0], MODEL_NAME] = " "
    forecast_key: ForecastKey = (" ", ISSUE_ORIGIN, 1, " ")
    quantile: BoundKey = (quantile_column(0.5),)
    literal_session = SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=(" ",),
        calendar=CALENDAR,
        horizon=1,
        model_config={"name": "literal-identifiers"},
    )
    ledger = Ledger(session=literal_session, calendar=CALENDAR)

    ledger.append_forecasts(frame, issuances={forecast_key: {quantile: _issuance()}})
    order = OrderRow(
        session=literal_session,
        series_key=" ",
        origin=ISSUE_ORIGIN,
        model_name=" ",
        quantity=0.0,
        arrival_period=ARRIVAL,
    )
    ledger.append_orders([order])

    assert ledger.calendar is CALENDAR
    assert ledger.forecasts[0].series_key == ledger.orders[0].series_key == " "
    assert ledger.forecasts[0].model_name == ledger.orders[0].model_name == " "


@pytest.mark.parametrize("quantity", [-0.1, float("nan"), float("inf"), True])
def test_order_quantity_must_be_finite_nonnegative(quantity: object) -> None:
    with pytest.raises(LedgerError, match="quantity"):
        _order(quantity=cast(Any, quantity))


def test_order_keys_are_unique_immutable_and_session_scoped() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    first = _order()
    ledger.append_orders([first])

    with pytest.raises(LedgerError, match="duplicate order key"):
        ledger.append_orders([_order(quantity=9.0)])
    with pytest.raises(LedgerError, match="session"):
        ledger.append_orders([_order(series="sku-b", session=_session(tenant="other"))])
    new = _order(series="sku-b")
    with pytest.raises(LedgerError, match="duplicate order key"):
        ledger.append_orders([new, _order(series="sku-b", quantity=9.0)])
    with pytest.raises(FrozenInstanceError):
        cast(Any, first).quantity = 9.0
    assert ledger.orders == (first,)


def test_settlement_shape_recomputes_costs_and_is_immutable() -> None:
    row = _settlement()

    assert row.holding.recomputed_amount == row.holding.amount == 1.0
    assert row.shortage.recomputed_amount == row.shortage.amount == 2.0
    assert row.realized_cost == 3.0
    assert row.transition.fulfilled_demand + row.transition.unmet_demand == 5.0
    with pytest.raises(FrozenInstanceError):
        cast(Any, row).arrivals = 0.0


def test_settlement_rejects_inconsistent_transition_or_cost_booking() -> None:
    with pytest.raises(LedgerError, match="must be ActualsSemantics"):
        replace(_settlement(), actuals_semantics="demand")  # type: ignore[arg-type]
    with pytest.raises(LedgerError, match="fulfilled.*unmet"):
        StockoutTransition(
            rule=StockoutRule.LOST_SALES,
            demand=5.0,
            fulfilled_demand=3.0,
            unmet_demand=1.0,
            closing_on_hand=0.0,
            closing_backorders=0.0,
        )
    with pytest.raises(LedgerError, match="rate.*basis"):
        BookedCost(rate=0.5, basis=2.0, amount=2.0)
    with pytest.raises(LedgerError, match="holding cost basis"):
        _settlement(holding_basis=3.0)
    with pytest.raises(LedgerError, match="shortage cost basis"):
        _settlement(shortage_basis=2.0)


def test_stockout_transition_accepts_one_ulp_float_conservation_drift() -> None:
    demand = 457.8931265964433
    fulfilled = 170.79992387036359

    transition = StockoutTransition(
        rule=StockoutRule.LOST_SALES,
        demand=demand,
        fulfilled_demand=fulfilled,
        unmet_demand=demand - fulfilled,
        closing_on_hand=0.0,
        closing_backorders=0.0,
    )

    assert transition.demand == demand

    with pytest.raises(LedgerError, match="must be a StockoutRule"):
        StockoutTransition(
            rule="lost-sales",  # type: ignore[arg-type]
            demand=0.0,
            fulfilled_demand=0.0,
            unmet_demand=0.0,
            closing_on_hand=0.0,
            closing_backorders=0.0,
        )


def test_settlement_keys_are_unique_atomic_and_session_scoped() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    first = _settlement()
    ledger.append_settlements([first])

    with pytest.raises(LedgerError, match="duplicate settlement key"):
        ledger.append_settlements([_settlement()])
    with pytest.raises(LedgerError, match="session"):
        ledger.append_settlements(
            [_settlement(period=pd.Timestamp("2026-01-08"), session=_session(tenant="other"))]
        )
    new = _settlement(period=pd.Timestamp("2026-01-08"))
    with pytest.raises(LedgerError, match="duplicate settlement key"):
        ledger.append_settlements([new, _settlement(period=pd.Timestamp("2026-01-08"))])
    assert ledger.settlements == (first,)


class _OnePassOrders(Iterable[OrderRow]):
    def __init__(self, rows: tuple[OrderRow, ...]) -> None:
        self._rows = rows
        self.iterations = 0

    def __iter__(self) -> Iterator[OrderRow]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("append iterated the input chunk more than once")
        return iter(self._rows)


def test_append_consumes_a_new_chunk_once_without_replacing_prior_rows() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    first = _order()
    ledger.append_orders([first])
    new_rows = (
        _order(series="sku-b", quantity=1.0),
        _order(series="sku-a", origin=pd.Timestamp("2026-01-06"), quantity=2.0),
    )
    chunk = _OnePassOrders(new_rows)

    ledger.append_orders(chunk)

    assert chunk.iterations == 1
    assert ledger.orders == (first, *new_rows)


def test_open_orders_and_realized_cost_are_derivable_from_public_rows_only() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    arrived = _order(arrival=pd.Timestamp("2026-01-07"))
    still_open = _order(
        series="sku-b",
        arrival=pd.Timestamp("2026-01-08"),
        quantity=4.0,
    )
    ledger.append_orders([arrived, still_open])
    ledger.append_settlements([_settlement(period=pd.Timestamp("2026-01-07"))])

    settled_periods = {(row.session, row.series_key, row.period) for row in ledger.settlements}
    open_orders = tuple(
        row
        for row in ledger.orders
        if (row.session, row.series_key, row.arrival_period) not in settled_periods
    )
    realized_cost = sum(row.realized_cost for row in ledger.settlements)

    assert open_orders == (still_open,)
    assert realized_cost == 3.0
