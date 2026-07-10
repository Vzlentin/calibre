"""Exercise immutable row families and append behavior through the ledger interface."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pandas as pd
import pytest

from newcalibre.domain import (
    Calendar,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    ScoredSeries,
    SessionIdentity,
    quantile_column,
)
from newcalibre.ledger import (
    BookedCost,
    ForecastIssuance,
    ForecastRow,
    GuaranteedSide,
    Ledger,
    LedgerError,
    OrderRow,
    SettlementRecord,
    StockoutTransition,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
ORIGIN = pd.Timestamp("2026-01-05")
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
) -> ForecastIssuance:
    return ForecastIssuance(
        descriptor=_descriptor(claim=claim),
        guaranteed_side=side,
        calibration_ready=False,
        bounds_finite=finite,
        bounds_null_reason=reason,
    )


def _frame(*, actual: float | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": pd.Series(["sku-a", "sku-b"], dtype="string"),
            "target_timestamp": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "actual_value": pd.Series([actual, actual], dtype="float64"),
            "point_forecast": pd.Series([10.0, 20.0], dtype="float64"),
            "horizon_step": pd.Series([1, 2], dtype="int64"),
            "origin": pd.to_datetime([ORIGIN, ORIGIN]),
            "model_name": pd.Series(["seasonal", "seasonal"], dtype="string"),
            quantile_column(0.5): pd.Series([10.0, 20.0], dtype="float64"),
            "adapter_note": pd.Series(["first", "second"], dtype="string"),
        }
    )


def _forecast_key(series: str, step: int) -> tuple[str, pd.Timestamp, int, str]:
    return (series, ORIGIN, step, "seasonal")


def _issuances() -> dict[tuple[str, pd.Timestamp, int, str], ForecastIssuance]:
    return {
        _forecast_key("sku-a", 1): _issuance(),
        _forecast_key("sku-b", 2): _issuance(),
    }


def _order(
    *,
    series: str = "sku-a",
    origin: pd.Timestamp = ORIGIN,
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
) -> SettlementRecord:
    return SettlementRecord(
        session=session or _session(),
        series_key="sku-a",
        period=period,
        arrivals=3.0,
        transition=StockoutTransition(
            rule="lost-sales",
            demand=5.0,
            fulfilled_demand=4.0,
            unmet_demand=1.0,
            closing_on_hand=2.0,
            closing_backorders=0.0,
        ),
        holding=BookedCost(rate=0.5, basis=2.0, amount=1.0),
        shortage=BookedCost(rate=2.0, basis=1.0, amount=2.0),
    )


def test_public_ledger_round_trips_all_three_row_families_in_append_order() -> None:
    ledger = Ledger(session=_session())

    ledger.append_forecasts(_frame(), calendar=CALENDAR, issuances=_issuances())
    ledger.append_orders([_order(quantity=0.0), _order(series="sku-b", quantity=4.0)])
    ledger.append_settlements([_settlement()])

    assert [row.key for row in ledger.forecasts] == [
        _forecast_key("sku-a", 1),
        _forecast_key("sku-b", 2),
    ]
    assert ledger.forecasts[0].point_forecast == 10.0
    assert ledger.forecasts[0].actual_value is None
    assert ledger.forecasts[0].values["adapter_note"] == "first"
    assert ledger.forecasts[0].descriptor == _descriptor()
    assert ledger.forecasts[0].calibration_ready is False
    assert ledger.forecasts[0].bounds_finite is True
    assert ledger.forecasts[0].bounds_null_reason is None
    assert [row.quantity for row in ledger.orders] == [0.0, 4.0]
    assert ledger.orders[0].key == (_session(), "sku-a", ORIGIN, "seasonal")
    assert ledger.settlements == (_settlement(),)
    assert ledger.settlements[0].key == (_session(), "sku-a", ARRIVAL)


def test_forecast_append_requires_pending_rows_and_exact_issuance_keys() -> None:
    ledger = Ledger(session=_session())

    with pytest.raises(LedgerError, match="pending"):
        ledger.append_forecasts(_frame(actual=1.0), calendar=CALENDAR, issuances=_issuances())
    assert ledger.forecasts == ()

    missing = _issuances()
    del missing[_forecast_key("sku-b", 2)]
    with pytest.raises(LedgerError, match="issuance keys"):
        ledger.append_forecasts(_frame(), calendar=CALENDAR, issuances=missing)

    extra = {**_issuances(), _forecast_key("sku-a", 2): _issuance()}
    with pytest.raises(LedgerError, match="issuance keys"):
        ledger.append_forecasts(_frame(), calendar=CALENDAR, issuances=extra)
    assert ledger.forecasts == ()


def test_forecast_append_is_atomic_and_rejects_existing_keys() -> None:
    ledger = Ledger(session=_session())
    ledger.append_forecasts(_frame(), calendar=CALENDAR, issuances=_issuances())
    before = ledger.forecasts

    with pytest.raises(LedgerError, match="duplicate forecast key"):
        ledger.append_forecasts(_frame(), calendar=CALENDAR, issuances=_issuances())

    assert ledger.forecasts == before


def test_forecast_rows_snapshot_the_frame_and_are_deeply_immutable() -> None:
    frame = _frame()
    ledger = Ledger(session=_session())
    ledger.append_forecasts(frame, calendar=CALENDAR, issuances=_issuances())
    row = ledger.forecasts[0]

    frame.loc[0, "point_forecast"] = 999.0
    assert row.point_forecast == 10.0
    with pytest.raises(TypeError):
        cast(Any, row.values)["point_forecast"] = 999.0
    with pytest.raises(FrozenInstanceError):
        cast(Any, row).issuance = _issuance()
    with pytest.raises(TypeError, match="Ledger.append_forecasts"):
        ForecastRow()


def test_forecast_issuance_fact_is_closed_and_matches_the_bound_payload() -> None:
    with pytest.raises(LedgerError, match="null reason"):
        _issuance(finite=False, reason=None)
    with pytest.raises(LedgerError, match="null reason"):
        _issuance(finite=True, reason="warm-up")
    with pytest.raises(LedgerError, match="guaranteed side"):
        _issuance(claim=GuaranteeClaim.ONE_SIDED_COVERAGE)
    with pytest.raises(LedgerError, match="only one-sided"):
        _issuance(side=GuaranteedSide.UPPER)

    ledger = Ledger(session=_session())
    inconsistent = {key: _issuance(finite=False, reason="warm-up") for key in _issuances()}
    with pytest.raises(LedgerError, match="finiteness"):
        ledger.append_forecasts(_frame(), calendar=CALENDAR, issuances=inconsistent)

    unavailable = _frame()
    unavailable[quantile_column(0.5)] = float("nan")
    ledger.append_forecasts(
        unavailable,
        calendar=CALENDAR,
        issuances={
            key: _issuance(finite=False, reason="calibration warm-up") for key in _issuances()
        },
    )
    assert [row.bounds_null_reason for row in ledger.forecasts] == [
        "calibration warm-up",
        "calibration warm-up",
    ]


@pytest.mark.parametrize("quantity", [-0.1, float("nan"), float("inf"), True])
def test_order_quantity_must_be_finite_nonnegative(quantity: object) -> None:
    with pytest.raises(LedgerError, match="quantity"):
        _order(quantity=cast(Any, quantity))


def test_order_keys_are_unique_immutable_and_session_scoped() -> None:
    ledger = Ledger(session=_session())
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
    with pytest.raises(LedgerError, match="fulfilled.*unmet"):
        StockoutTransition(
            rule="lost-sales",
            demand=5.0,
            fulfilled_demand=3.0,
            unmet_demand=1.0,
            closing_on_hand=0.0,
            closing_backorders=0.0,
        )
    with pytest.raises(LedgerError, match="rate.*basis"):
        BookedCost(rate=0.5, basis=2.0, amount=2.0)


def test_settlement_keys_are_unique_atomic_and_session_scoped() -> None:
    ledger = Ledger(session=_session())
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
    ledger = Ledger(session=_session())
    first = _order()
    ledger.append_orders([first])
    chunk = _OnePassOrders(
        (
            _order(series="sku-b", quantity=1.0),
            _order(series="sku-a", origin=pd.Timestamp("2026-01-06"), quantity=2.0),
        )
    )

    ledger.append_orders(chunk)

    assert chunk.iterations == 1
    assert ledger.orders == (first, *chunk._rows)


def test_open_orders_and_realized_cost_are_derivable_from_public_rows_only() -> None:
    ledger = Ledger(session=_session())
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
