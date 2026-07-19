"""Exercise one-shot forecast resolution through the public ledger interface."""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from newcalibre.conformal import (
    Delivery,
    EmissionForm,
    IssuedBoundFacts,
    ObserveAnnotation,
    ResolvedObservation,
    derive_partition_label,
)
from newcalibre.conformal import (
    ForecastKey as ConformalForecastKey,
)
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    REQUIRED_FRAME_COLUMNS,
    SERIES_KEY,
    TARGET_TIMESTAMP,
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
    validate_forecast_frame,
)
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    GuaranteedSide,
    Ledger,
    LedgerError,
)
from newcalibre.observe import ObservationResolution, ObserveCycle, PendingObservation

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


def _observation_facts(step: int, *, ready: bool = True) -> IssuedBoundFacts:
    lower = 0.0 if ready else math.nan
    upper = float(step * 10 + 2) if ready else math.nan
    return IssuedBoundFacts(
        method_name="split-per-step",
        emission_form=EmissionForm.ONE_SIDED_UPPER,
        emission_scope=EmissionScope.PER_STEP,
        partition_label=derive_partition_label(
            "seasonal",
            "global",
            EmissionScope.PER_STEP,
        ),
        working_level=0.5,
        state_reference="split-per-step:fixture",
        lower_bound=lower,
        upper_bound=upper,
        calibration_ready=ready,
        bounds_null_reason=None if ready else "warm-up",
        effective_descriptor=GuaranteeDescriptor(
            type=GuaranteeType(
                claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
                currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
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
    )


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


def _conformal_ledger() -> Ledger:
    frame = _frame(steps=(1,))
    lower, upper = "lower_0.5", "upper_0.5"
    frame[lower] = pd.Series([0.0], dtype="float64")
    frame[upper] = pd.Series([12.0], dtype="float64")
    facts = _observation_facts(1)
    upper_claim = ForecastIssuance(
        descriptor=facts.effective_descriptor,
        guaranteed_side=GuaranteedSide.UPPER,
        calibration_ready=True,
        bounds_finite=True,
        bounds_null_reason=None,
    )
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(
        frame,
        issuances={_key(1): {(lower,): _issuance(), (upper,): upper_claim}},
        observation_issuances={_key(1): facts},
    )
    return ledger


def _assert_empty_due_frame(frame: pd.DataFrame) -> None:
    assert frame.index.equals(pd.RangeIndex(0))
    assert tuple(frame.columns) == REQUIRED_FRAME_COLUMNS
    assert isinstance(frame[SERIES_KEY].dtype, pd.StringDtype)
    assert str(frame[TARGET_TIMESTAMP].dtype) == "datetime64[ns]"
    assert str(frame[ACTUAL_VALUE].dtype) == "float64"
    assert str(frame[POINT_FORECAST].dtype) == "float64"
    assert str(frame[HORIZON_STEP].dtype) == "int64"
    assert str(frame[ORIGIN].dtype) == "datetime64[ns]"
    assert isinstance(frame[MODEL_NAME].dtype, pd.StringDtype)
    validate_forecast_frame(frame, calendar=CALENDAR)


def test_pending_observation_projection_is_exact_append_ordered_and_defensive() -> None:
    frame = _frame(steps=(2, 1))
    lower, upper = "lower_0.5", "upper_0.5"
    frame[lower] = pd.Series([0.0, 0.0], dtype="float64")
    frame[upper] = pd.Series([22.0, 12.0], dtype="float64")
    facts = {_key(step): _observation_facts(step) for step in (2, 1)}
    no_claim = _issuance()
    upper_claim = ForecastIssuance(
        descriptor=_observation_facts(1).effective_descriptor,
        guaranteed_side=GuaranteedSide.UPPER,
        calibration_ready=True,
        bounds_finite=True,
        bounds_null_reason=None,
    )
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(
        frame,
        issuances={_key(step): {(lower,): no_claim, (upper,): upper_claim} for step in (2, 1)},
        observation_issuances=facts,
    )

    first = ledger.pending_observations
    second = ledger.pending_observations

    assert first == (
        PendingObservation(
            ConformalForecastKey("sku-a", ISSUE_ORIGIN, 2, "seasonal"),
            pd.Timestamp("2026-01-02"),
            20.0,
            facts[_key(2)],
        ),
        PendingObservation(
            ConformalForecastKey("sku-a", ISSUE_ORIGIN, 1, "seasonal"),
            pd.Timestamp("2026-01-01"),
            10.0,
            facts[_key(1)],
        ),
    )
    assert second == first
    assert second is not first
    assert second[0] is not first[0]
    assert second[0].issued is not first[0].issued


def test_pending_projection_preserves_missing_facts_and_cold_start_nan_bounds() -> None:
    cold = _frame(steps=(1,))
    lower, upper = "lower_0.5", "upper_0.5"
    cold[lower] = pd.Series([math.nan], dtype="float64")
    cold[upper] = pd.Series([math.nan], dtype="float64")
    nonfinite = ForecastIssuance(
        descriptor=_observation_facts(1, ready=False).effective_descriptor,
        guaranteed_side=GuaranteedSide.UPPER,
        calibration_ready=False,
        bounds_finite=False,
        bounds_null_reason="warm-up",
    )
    no_claim_nonfinite = ForecastIssuance(
        descriptor=_issuance().descriptor,
        guaranteed_side=None,
        calibration_ready=False,
        bounds_finite=False,
        bounds_null_reason="warm-up",
    )
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(
        cold,
        issuances={_key(1): {(lower,): no_claim_nonfinite, (upper,): nonfinite}},
        observation_issuances={_key(1): _observation_facts(1, ready=False)},
    )
    plain = Ledger(session=_session(), calendar=CALENDAR)
    plain.append_forecasts(
        _frame(steps=(2,)),
        issuances={_key(2): {QUANTILE: _issuance()}},
    )

    issued = ledger.pending_observations[0].issued
    assert issued is not None
    assert math.isnan(issued.lower_bound)
    assert math.isnan(issued.upper_bound)
    assert plain.pending_observations[0].issued is None


def test_observation_issuance_projection_rejects_mismatched_or_foreign_facts_atomically() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    frame = _frame(steps=(1,))

    with pytest.raises(LedgerError, match="bound columns"):
        ledger.append_forecasts(
            frame,
            issuances={_key(1): {QUANTILE: _issuance()}},
            observation_issuances={_key(1): _observation_facts(1)},
        )
    with pytest.raises(LedgerError, match="unknown forecast key"):
        ledger.append_forecasts(
            frame,
            issuances={_key(1): {QUANTILE: _issuance()}},
            observation_issuances={_key(2): _observation_facts(2)},
        )

    assert ledger.forecasts == ()
    assert ledger.pending_observations == ()


def test_observation_issuance_rejects_payload_bound_mismatch_atomically() -> None:
    frame = _frame(steps=(1,))
    lower, upper = "lower_0.5", "upper_0.5"
    frame[lower] = pd.Series([0.0], dtype="float64")
    frame[upper] = pd.Series([12.0], dtype="float64")
    matching = _observation_facts(1)
    mismatched = replace(matching, upper_bound=13.0)
    upper_claim = ForecastIssuance(
        descriptor=matching.effective_descriptor,
        guaranteed_side=GuaranteedSide.UPPER,
        calibration_ready=True,
        bounds_finite=True,
        bounds_null_reason=None,
    )
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    with pytest.raises(LedgerError, match="bounds must equal the forecast payload"):
        ledger.append_forecasts(
            frame,
            issuances={_key(1): {(lower,): _issuance(), (upper,): upper_claim}},
            observation_issuances={_key(1): mismatched},
        )

    assert ledger.forecasts == ()
    assert ledger.pending_observations == ()


def test_empty_ledger_due_frame_has_a_stable_valid_schema() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)

    due = ledger.due_frame(pd.Timestamp("2026-01-02"))

    _assert_empty_due_frame(due)


def test_future_only_due_frame_has_the_same_empty_schema() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    ledger.append_forecasts(
        _frame(steps=(4,)),
        issuances={_key(4): {QUANTILE: _issuance()}},
    )

    before_target = ledger.due_frame(pd.Timestamp("2026-01-02"))
    at_target = ledger.due_frame(pd.Timestamp("2026-01-04"))

    _assert_empty_due_frame(before_target)
    _assert_empty_due_frame(at_target)


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
    assert isinstance(due[SERIES_KEY].dtype, pd.StringDtype)
    assert isinstance(due[MODEL_NAME].dtype, pd.StringDtype)
    validate_forecast_frame(due, calendar=CALENDAR)


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
    assert ledger.due_frame(pd.Timestamp("2026-01-05"))[HORIZON_STEP].tolist() == [2, 1, 3, 4]


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


def _resolution(row: PendingObservation, actual: float) -> ObservationResolution:
    return ObservationResolution(
        row.forecast_key,
        row.target_timestamp,
        actual,
        None,
        None,
    )


def _delivery_cycle(
    ledger: Ledger,
    values: dict[int, float],
    *,
    retain: bool = False,
) -> ObserveCycle:
    selected = {
        row.forecast_key.horizon_step: row
        for row in ledger.pending_observations
        if row.forecast_key.horizon_step in values
    }
    resolutions = {step: _resolution(selected[step], actual) for step, actual in values.items()}
    retained = []
    for row in ledger.pending_observations:
        resolution = resolutions.get(row.forecast_key.horizon_step)
        if resolution is None:
            retained.append(row)
        elif retain:
            retained.append(replace(row, resolution=resolution))
    return ObserveCycle(
        resolutions=() if retain else resolutions.values(),
        pending_removals=() if retain else (row.forecast_key for row in selected.values()),
        pending_retentions=retained,
    )


def test_due_frame_and_observe_cycle_require_an_owned_calendar_origin() -> None:
    ledger = _ledger()
    off_grid = pd.Timestamp("2026-01-03 12:00")

    with pytest.raises(LedgerError, match="calendar"):
        ledger.due_frame(off_grid)
    with pytest.raises(LedgerError, match="calendar"):
        ledger.apply_observe_cycle(
            _delivery_cycle(ledger, {1: 11.0}),
            origin=off_grid,
        )

    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]


def test_complete_delivery_materializes_censoring_aware_resolutions_once() -> None:
    ledger = _ledger()
    before = ledger.forecasts

    ledger.apply_observe_cycle(
        _delivery_cycle(ledger, {1: 11.0, 2: 22.0}),
        origin=pd.Timestamp("2026-01-04"),
    )

    assert [row.key for row in ledger.forecasts] == [_key(2), _key(1), _key(3), _key(4)]
    assert [row.actual_value for row in ledger.forecasts] == [22.0, 11.0, None, None]
    assert [row.actual_value for row in before] == [None, None, None, None]
    assert [value.actual for value in ledger.observation_resolutions] == [11.0, 22.0]
    assert [value.forecast_key.horizon_step for value in ledger.pending_observations] == [3, 4]


def test_resolved_incomplete_window_remains_pending_until_delivery() -> None:
    ledger = _ledger()
    ledger.apply_observe_cycle(
        _delivery_cycle(ledger, {1: 11.0}, retain=True),
        origin=pd.Timestamp("2026-01-03"),
    )

    retained = next(
        row for row in ledger.pending_observations if row.forecast_key.horizon_step == 1
    )
    assert retained.resolution is not None
    assert retained.resolution.actual == 11.0
    assert all(row.actual_value is None for row in ledger.forecasts)
    assert ledger.observation_resolutions == ()

    ledger.apply_observe_cycle(
        _delivery_cycle(ledger, {1: 11.0, 2: 22.0}),
        origin=pd.Timestamp("2026-01-04"),
    )
    assert [row.actual_value for row in ledger.forecasts] == [22.0, 11.0, None, None]


def test_conformal_issued_removal_requires_a_matching_delivery_without_effect() -> None:
    ledger = _conformal_ledger()
    pending = ledger.pending_observations[0]
    resolution = _resolution(pending, 11.0)
    cycle = ObserveCycle(
        resolutions=(resolution,),
        pending_removals=(pending.forecast_key,),
    )

    with pytest.raises(LedgerError, match="conformal-issued pending removals"):
        ledger.apply_observe_cycle(cycle, origin=pd.Timestamp("2026-01-03"))

    assert ledger.forecasts[0].actual_value is None
    assert ledger.observation_resolutions == ()
    assert ledger.observe_annotations == ()
    assert ledger.pending_observations == (pending,)


def test_delivery_facts_must_match_the_staged_resolution_without_effect() -> None:
    ledger = _conformal_ledger()
    pending = ledger.pending_observations[0]
    issued = pending.issued
    if issued is None:
        raise AssertionError("conformal fixture must carry issued facts")
    resolution = _resolution(pending, 11.0)
    delivery = Delivery(
        issued.partition_label,
        (
            ResolvedObservation(
                pending.forecast_key,
                pending.target_timestamp,
                12.0,
                pending.point_forecast,
                resolution.censoring_assertion,
                resolution.availability_bound,
                issued,
            ),
        ),
    )
    cycle = ObserveCycle(
        resolutions=(resolution,),
        pending_removals=(pending.forecast_key,),
        deliveries=(delivery,),
        annotations=(ObserveAnnotation(pending.forecast_key, 2.0, None, True),),
    )

    with pytest.raises(LedgerError, match="delivery facts do not match staged resolution"):
        ledger.apply_observe_cycle(cycle, origin=pd.Timestamp("2026-01-03"))

    assert ledger.forecasts[0].actual_value is None
    assert ledger.observation_resolutions == ()
    assert ledger.observe_annotations == ()
    assert ledger.pending_observations == (pending,)


def test_observe_cycle_rejects_unknown_pending_keys_without_effect() -> None:
    ledger = _ledger()
    unknown = ConformalForecastKey("missing", ISSUE_ORIGIN, 1, "seasonal")
    cycle = ObserveCycle(
        resolutions=(
            ObservationResolution(
                unknown,
                ISSUE_ORIGIN,
                99.0,
                None,
                None,
            ),
        ),
        pending_removals=(unknown,),
        pending_retentions=ledger.pending_observations,
    )

    with pytest.raises(LedgerError, match="unknown pending"):
        ledger.apply_observe_cycle(cycle, origin=pd.Timestamp("2026-01-03"))

    assert [row.actual_value for row in ledger.forecasts] == [None, None, None, None]
    assert len(ledger.pending_observations) == 4
