"""Exercise one-shot forecast resolution through the public ledger interface."""

from __future__ import annotations

import math

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
    Calendar,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeDescriptor,
    GuaranteeType,
    ScoredSeries,
    SessionIdentity,
    quantile_column,
)
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    Ledger,
    LedgerError,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
ISSUE_ORIGIN = pd.Timestamp("2026-01-01")
QUANTILE: BoundKey = (quantile_column(0.5),)


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("sku-a",),
        calendar=CALENDAR,
        horizon=4,
        model_config={"name": "seasonal-naive"},
    )


def _issuance() -> ForecastIssuance:
    return ForecastIssuance(
        descriptor=GuaranteeDescriptor(
            type=GuaranteeType(
                claim=GuaranteeClaim.NONE,
                currency=None,
                declared_slack=None,
            ),
            level=0.5,
            scored_series=ScoredSeries.DEMAND_HONEST,
            window=EmissionScope.PER_STEP,
            scope=DecisionScope(
                kind=DecisionScopeKind.PER_DECISION_NODE,
                class_system_name=None,
            ),
        ),
        guaranteed_side=None,
        calibration_ready=False,
        bounds_finite=True,
        bounds_null_reason=None,
    )


def _key(step: int) -> ForecastKey:
    return ("sku-a", ISSUE_ORIGIN, step, "seasonal")


def _frame(*, steps: tuple[int, ...] = (2, 1, 3, 4)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-a"] * len(steps), dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(
                [ISSUE_ORIGIN + pd.Timedelta(days=step - 1) for step in steps]
            ),
            ACTUAL_VALUE: pd.Series([None] * len(steps), dtype="float64"),
            POINT_FORECAST: pd.Series([float(step * 10) for step in steps], dtype="float64"),
            HORIZON_STEP: pd.Series(steps, dtype="int64"),
            ORIGIN: pd.to_datetime([ISSUE_ORIGIN] * len(steps)),
            MODEL_NAME: pd.Series(["seasonal"] * len(steps), dtype="string"),
            QUANTILE[0]: pd.Series([float(step * 10) for step in steps], dtype="float64"),
        }
    )


def _ledger() -> Ledger:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(
        _frame(),
        issuances={_key(step): {QUANTILE: _issuance()} for step in (2, 1, 3, 4)},
    )
    return ledger


def test_due_frame_filters_pending_rows_strictly_before_a_calendar_origin() -> None:
    ledger = _ledger()

    due = ledger.due_frame(pd.Timestamp("2026-01-03"))

    assert isinstance(due.index, pd.RangeIndex)
    assert due.index.equals(pd.RangeIndex(2))
    assert tuple(due.columns) == tuple(_frame().columns)
    assert due[HORIZON_STEP].tolist() == [2, 1]
    assert due[TARGET_TIMESTAMP].tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-01"),
    ]
    assert due[ACTUAL_VALUE].isna().all()
    assert 3 not in due[HORIZON_STEP].tolist()  # Equality with origin is not due.


def test_due_frame_is_a_fresh_snapshot_that_cannot_mutate_the_ledger() -> None:
    ledger = _ledger()
    first = ledger.due_frame(pd.Timestamp("2026-01-03"))

    first.loc[0, POINT_FORECAST] = 999.0
    first.loc[0, ACTUAL_VALUE] = 999.0
    first.index = pd.Index((20, 21))
    second = ledger.due_frame(pd.Timestamp("2026-01-03"))

    assert second is not first
    assert second.index.equals(pd.RangeIndex(2))
    assert second[POINT_FORECAST].tolist() == [20.0, 10.0]
    assert second[ACTUAL_VALUE].isna().all()
    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]


def test_due_frame_does_not_materialize_future_row_extension_schemas() -> None:
    ledger = _ledger()
    future = _frame(steps=(4,))
    future[MODEL_NAME] = pd.Series(["future-model"], dtype="string")
    future["future_only"] = pd.Series([99.0], dtype="float64")
    future_key: ForecastKey = ("sku-a", ISSUE_ORIGIN, 4, "future-model")
    ledger.append_forecasts(
        future,
        issuances={future_key: {QUANTILE: _issuance()}},
    )

    due = ledger.due_frame(pd.Timestamp("2026-01-03"))

    assert "future_only" not in due.columns
    assert due[HORIZON_STEP].tolist() == [2, 1]


def test_due_frame_and_resolution_require_an_origin_on_the_owned_calendar() -> None:
    ledger = _ledger()
    off_grid = pd.Timestamp("2026-01-03 12:00")

    with pytest.raises(LedgerError, match="calendar"):
        ledger.due_frame(off_grid)
    with pytest.raises(LedgerError, match="calendar"):
        ledger.apply_resolutions({_key(1): 11.0}, origin=off_grid)

    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]


def test_keyed_subset_resolution_replaces_rows_without_reordering_or_degrading() -> None:
    ledger = _ledger()
    before = ledger.forecasts
    before_values = [dict(row.values) for row in before]

    # Resolve in the reverse of ledger order and intentionally omit another late row.
    ledger.apply_resolutions(
        {_key(1): 11.0, _key(2): 22.0},
        origin=pd.Timestamp("2026-01-04"),
    )
    after = ledger.forecasts

    assert [row.key for row in after] == [_key(2), _key(1), _key(3), _key(4)]
    assert [row.actual_value for row in after] == [22.0, 11.0, None, None]
    assert [row.actual_value for row in before] == [None, None, None, None]
    assert after[0] is not before[0]
    assert after[1] is not before[1]
    assert after[2] is before[2]
    assert after[3] is before[3]

    for index, expected_actual in ((0, 22.0), (1, 11.0)):
        expected = before_values[index] | {ACTUAL_VALUE: expected_actual}
        assert dict(after[index].values) == expected
        assert after[index].point_forecast == before[index].point_forecast
        assert after[index].issuances == before[index].issuances

    late = ledger.due_frame(pd.Timestamp("2026-01-04"))
    assert late[HORIZON_STEP].tolist() == [3]
    assert late[ACTUAL_VALUE].isna().all()


def test_resolution_rejects_unknown_keys_atomically() -> None:
    ledger = _ledger()
    unknown: ForecastKey = ("missing", ISSUE_ORIGIN, 1, "seasonal")

    with pytest.raises(LedgerError, match="unknown"):
        ledger.apply_resolutions(
            {_key(1): 11.0, unknown: 99.0},
            origin=pd.Timestamp("2026-01-03"),
        )

    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]


def test_resolution_rejects_already_resolved_rows_atomically() -> None:
    ledger = _ledger()
    ledger.apply_resolutions({_key(1): 11.0}, origin=pd.Timestamp("2026-01-03"))
    before = ledger.forecasts

    with pytest.raises(LedgerError, match="resolved"):
        ledger.apply_resolutions(
            {_key(2): 22.0, _key(1): 11.0},
            origin=pd.Timestamp("2026-01-03"),
        )

    assert ledger.forecasts == before
    assert [row.actual_value for row in ledger.forecasts] == [None, 11.0, None, None]


def test_resolution_rejects_not_yet_due_rows_atomically() -> None:
    ledger = _ledger()

    with pytest.raises(LedgerError, match="due"):
        ledger.apply_resolutions(
            {_key(1): 11.0, _key(3): 33.0},
            origin=pd.Timestamp("2026-01-03"),
        )

    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True])
def test_resolution_rejects_nonfinite_values_atomically(invalid: float) -> None:
    ledger = _ledger()

    with pytest.raises(LedgerError, match="finite"):
        ledger.apply_resolutions(
            {_key(1): 11.0, _key(2): invalid},
            origin=pd.Timestamp("2026-01-03"),
        )

    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]
