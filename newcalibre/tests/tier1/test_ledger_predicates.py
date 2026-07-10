"""Exercise descriptor-keyed ledger scoring and denominator attribution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from itertools import product
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
    interval_columns,
    quantile_column,
)
from newcalibre.ledger import (
    BoundKey,
    CoverageReport,
    CoverageSummary,
    ForecastIssuance,
    ForecastKey,
    GuaranteedSide,
    Ledger,
    LedgerError,
    PredicateKey,
    PredicateRegistration,
    PredicateRegistry,
    PredicateResult,
    ScoreOutcome,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
ISSUE_ORIGIN = pd.Timestamp("2026-01-01")
SCORE_ORIGIN = pd.Timestamp("2026-01-02")


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("sku-a",),
        calendar=CALENDAR,
        horizon=1,
        model_config={"name": "predicate-proof"},
    )


def _descriptor(
    claim: GuaranteeClaim,
    currency: GuaranteeCurrency | None,
    *,
    level: float = 0.9,
) -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=claim,
            currency=currency,
            declared_slack=(
                0.05 if currency is GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK else None
            ),
        ),
        level=level,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=(
                "hierarchy-levels-v1"
                if claim is GuaranteeClaim.CLASS_CONDITIONAL_COVERAGE
                else None
            ),
        ),
    )


def _issuance(
    descriptor: GuaranteeDescriptor,
    *,
    side: GuaranteedSide | None = None,
    finite: bool = True,
    reason: str | None = None,
    ready: bool | None = None,
) -> ForecastIssuance:
    return ForecastIssuance(
        descriptor=descriptor,
        guaranteed_side=side,
        calibration_ready=(
            finite and descriptor.type.claim is not GuaranteeClaim.NONE if ready is None else ready
        ),
        bounds_finite=finite,
        bounds_null_reason=None if finite else reason,
    )


def _append(
    ledger: Ledger,
    *,
    model: str,
    bound_values: Mapping[str, float],
    issuances: Mapping[BoundKey, ForecastIssuance],
) -> ForecastKey:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-a"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([ISSUE_ORIGIN]),
            ACTUAL_VALUE: pd.Series([None], dtype="float64"),
            POINT_FORECAST: pd.Series([10.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1], dtype="int64"),
            ORIGIN: pd.to_datetime([ISSUE_ORIGIN]),
            MODEL_NAME: pd.Series([model], dtype="string"),
            **{
                column: pd.Series([value], dtype="float64")
                for column, value in bound_values.items()
            },
        }
    )
    key: ForecastKey = ("sku-a", ISSUE_ORIGIN, 1, model)
    ledger.append_forecasts(frame, issuances={key: issuances})
    return key


def _append_one_sided(
    ledger: Ledger,
    *,
    model: str,
    side: GuaranteedSide,
    lower: float,
    upper: float,
    finite: bool = True,
    reason: str | None = None,
    ready: bool | None = None,
    currency: GuaranteeCurrency = GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
    level: float = 0.9,
) -> tuple[ForecastKey, BoundKey]:
    interval = interval_columns(level)
    scored_key: BoundKey = (interval[0] if side is GuaranteedSide.LOWER else interval[1],)
    opposite_key: BoundKey = (interval[1] if side is GuaranteedSide.LOWER else interval[0],)
    one_sided = _issuance(
        _descriptor(GuaranteeClaim.ONE_SIDED_COVERAGE, currency, level=level),
        side=side,
        finite=finite,
        reason=reason,
        ready=ready,
    )
    no_claim = _issuance(_descriptor(GuaranteeClaim.NONE, None, level=level))
    key = _append(
        ledger,
        model=model,
        bound_values={interval[0]: lower, interval[1]: upper},
        issuances={scored_key: one_sided, opposite_key: no_claim},
    )
    return key, scored_key


def _outcome(report: CoverageReport, key: ForecastKey, bound_key: BoundKey) -> ScoreOutcome:
    return next(
        outcome
        for outcome in report.outcomes
        if outcome.forecast_key == key and outcome.bound_key == bound_key
    )


def _summary(
    report: CoverageReport,
    key: ForecastKey,
    bound_key: BoundKey,
) -> CoverageSummary:
    outcome = _outcome(report, key, bound_key)
    return report.summaries[outcome.target]


def _indicator_predicate(
    actual_value: float,
    bound_values: tuple[float, ...],
    issuance: ForecastIssuance,
) -> PredicateResult:
    covered = bool(actual_value or bound_values or issuance)
    return PredicateResult(value=float(covered), covered=covered)


@pytest.mark.parametrize(
    "key",
    [
        pair
        for pair in product(
            (
                GuaranteeClaim.ONE_SIDED_COVERAGE,
                GuaranteeClaim.TWO_SIDED_COVERAGE,
                GuaranteeClaim.RISK_CONTROL,
                GuaranteeClaim.CLASS_CONDITIONAL_COVERAGE,
            ),
            tuple(GuaranteeCurrency),
        )
        if pair
        not in {
            (
                GuaranteeClaim.ONE_SIDED_COVERAGE,
                GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            ),
            (
                GuaranteeClaim.TWO_SIDED_COVERAGE,
                GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            ),
        }
    ],
)
def test_registry_rejects_predicates_outside_supported_row_event_pairs(
    key: PredicateKey,
) -> None:
    with pytest.raises(LedgerError, match="row-event predicates support only"):
        PredicateRegistration(key=key, predicate=_indicator_predicate)


def test_supported_custom_predicate_receives_exact_inputs_and_propagates_result() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    key, bound_key = _append_one_sided(
        ledger,
        model="custom-row-event",
        side=GuaranteedSide.UPPER,
        lower=0.0,
        upper=12.0,
    )
    ledger.apply_resolutions({key: 10.0}, origin=SCORE_ORIGIN)
    issuance = ledger.forecasts[0].issuances[bound_key]
    observed: list[tuple[float, tuple[float, ...], ForecastIssuance]] = []

    def custom_predicate(
        actual_value: float,
        bound_values: tuple[float, ...],
        actual_issuance: ForecastIssuance,
    ) -> PredicateResult:
        observed.append((actual_value, bound_values, actual_issuance))
        return PredicateResult(value=0.25, covered=None)

    registry = PredicateRegistry(
        (
            PredicateRegistration(
                key=(
                    GuaranteeClaim.ONE_SIDED_COVERAGE,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                ),
                predicate=custom_predicate,
            ),
        )
    )

    report = ledger.coverage_report(registry)
    outcome = _outcome(report, key, bound_key)

    assert observed == [(10.0, (12.0,), issuance)]
    assert outcome.value == 0.25
    assert outcome.covered is None
    assert outcome.scored is True
    assert report.summaries[outcome.target].coverage_ratio is None


def test_custom_predicate_must_return_a_predicate_result() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    key, _ = _append_one_sided(
        ledger,
        model="bad-custom-row-event",
        side=GuaranteedSide.UPPER,
        lower=0.0,
        upper=12.0,
    )
    ledger.apply_resolutions({key: 10.0}, origin=SCORE_ORIGIN)

    def invalid_predicate(
        actual_value: float,
        bound_values: tuple[float, ...],
        issuance: ForecastIssuance,
    ) -> PredicateResult:
        del actual_value, bound_values, issuance
        return cast(PredicateResult, object())

    registry = PredicateRegistry(
        (
            PredicateRegistration(
                key=(
                    GuaranteeClaim.ONE_SIDED_COVERAGE,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                ),
                predicate=invalid_predicate,
            ),
        )
    )

    with pytest.raises(LedgerError, match="must return a PredicateResult"):
        ledger.coverage_report(registry)


def test_predicate_result_carries_numeric_scores_without_forcing_coverage() -> None:
    result = PredicateResult(value=0.125, covered=None)

    assert result.value == 0.125
    assert result.covered is None
    with pytest.raises(FrozenInstanceError):
        cast(Any, result).value = 1.0
    with pytest.raises(LedgerError, match="boolean indicator"):
        PredicateResult(value=0.125, covered=True)


@pytest.mark.parametrize(
    "key",
    [
        (
            GuaranteeClaim.ONE_SIDED_COVERAGE,
            GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        ),
        (
            GuaranteeClaim.TWO_SIDED_COVERAGE,
            GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        ),
    ],
    ids=["one-sided", "two-sided"],
)
def test_registry_rejects_duplicate_claim_currency_pairs(key: PredicateKey) -> None:
    registration = PredicateRegistration(key=key, predicate=_indicator_predicate)

    with pytest.raises(LedgerError, match="duplicate"):
        PredicateRegistry((registration, registration))


def test_registration_is_immutable_and_none_claim_cannot_be_registered() -> None:
    registration = PredicateRegistration(
        key=(
            GuaranteeClaim.ONE_SIDED_COVERAGE,
            GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        ),
        predicate=_indicator_predicate,
    )

    with pytest.raises(FrozenInstanceError):
        cast(Any, registration).key = (
            GuaranteeClaim.TWO_SIDED_COVERAGE,
            GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        )
    with pytest.raises(LedgerError, match="none"):
        PredicateRegistration(
            key=(GuaranteeClaim.NONE, None),
            predicate=_indicator_predicate,
        )


@pytest.mark.parametrize(
    "key",
    [
        (GuaranteeClaim.NONE, GuaranteeCurrency.FINITE_SAMPLE_MARGINAL),
        (GuaranteeClaim.ONE_SIDED_COVERAGE, None),
        ("one-sided-coverage", GuaranteeCurrency.FINITE_SAMPLE_MARGINAL),
        (GuaranteeClaim.ONE_SIDED_COVERAGE, "finite-sample-marginal"),
        (GuaranteeClaim.ONE_SIDED_COVERAGE,),
    ],
)
def test_registration_rejects_pairs_outside_the_closed_vocabulary(key: object) -> None:
    with pytest.raises(LedgerError, match="predicate key|claim|currency|none"):
        PredicateRegistration(
            key=cast(PredicateKey, key),
            predicate=lambda actual, bounds, issuance: PredicateResult(
                value=1.0,
                covered=True,
            ),
        )


def test_gate_a_registry_recognizes_all_thirteen_closed_descriptor_pairs() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    intended: dict[PredicateKey, tuple[ForecastKey, BoundKey]] = {}
    pairs: tuple[PredicateKey, ...] = (
        *(
            (claim, currency)
            for claim in GuaranteeClaim
            if claim is not GuaranteeClaim.NONE
            for currency in GuaranteeCurrency
        ),
        (GuaranteeClaim.NONE, None),
    )

    for index, (claim, currency) in enumerate(pairs):
        model = f"vocabulary-{index}"
        descriptor = _descriptor(claim, currency)
        if claim is GuaranteeClaim.ONE_SIDED_COVERAGE:
            key, bound_key = _append_one_sided(
                ledger,
                model=model,
                side=GuaranteedSide.UPPER,
                lower=0.0,
                upper=10.0,
                currency=cast(GuaranteeCurrency, currency),
            )
        elif claim is GuaranteeClaim.TWO_SIDED_COVERAGE:
            bound_key = interval_columns(descriptor.level)
            key = _append(
                ledger,
                model=model,
                bound_values={bound_key[0]: 0.0, bound_key[1]: 10.0},
                issuances={bound_key: _issuance(descriptor)},
            )
        else:
            bound_key = (quantile_column(descriptor.level),)
            key = _append(
                ledger,
                model=model,
                bound_values={bound_key[0]: 10.0},
                issuances={bound_key: _issuance(descriptor)},
            )
        intended[(claim, currency)] = (key, bound_key)

    ledger.apply_resolutions(
        {key: 10.0 for key, _ in intended.values()},
        origin=SCORE_ORIGIN,
    )
    report = ledger.coverage_report(PredicateRegistry.gate_a())

    assert len(intended) == 13
    for predicate_key, (key, bound_key) in intended.items():
        outcome = _outcome(report, key, bound_key)
        assert (outcome.target.descriptor.type.claim, outcome.target.descriptor.type.currency) == (
            predicate_key
        )
        if predicate_key in {
            (
                GuaranteeClaim.ONE_SIDED_COVERAGE,
                GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            ),
            (
                GuaranteeClaim.TWO_SIDED_COVERAGE,
                GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            ),
        }:
            assert outcome.scored is True
            assert outcome.value == 1.0
            assert outcome.covered is True
            assert outcome.unscored_reason is None
        elif predicate_key == (GuaranteeClaim.NONE, None):
            assert outcome.scored is False
            assert outcome.value is None
            assert outcome.unscored_reason == "not-engine-calibrated"
        else:
            assert outcome.scored is False
            assert outcome.unscored_reason == "predicate-unregistered"


def test_mixed_target_uses_only_scored_outcomes_as_its_denominator() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    references: list[tuple[ForecastKey, BoundKey]] = []
    references.append(
        _append_one_sided(
            ledger,
            model="pending",
            side=GuaranteedSide.UPPER,
            lower=0.0,
            upper=10.0,
        )
    )
    references.append(
        _append_one_sided(
            ledger,
            model="warm-up",
            side=GuaranteedSide.UPPER,
            lower=0.0,
            upper=math.nan,
            finite=False,
            reason="calibrator-needs-28-resolved-scores",
            ready=False,
        )
    )
    references.append(
        _append_one_sided(
            ledger,
            model="ready-but-nonfinite",
            side=GuaranteedSide.UPPER,
            lower=0.0,
            upper=math.nan,
            finite=False,
            reason="method-returned-nonfinite",
            ready=True,
        )
    )
    references.extend(
        _append_one_sided(
            ledger,
            model=model,
            side=GuaranteedSide.UPPER,
            lower=0.0,
            upper=bound,
        )
        for model, bound in (("covered-a", 10.0), ("uncovered", 9.0), ("covered-b", 11.0))
    )
    ledger.apply_resolutions(
        {key: 10.0 for key, _ in references[1:]},
        origin=SCORE_ORIGIN,
    )

    report = ledger.coverage_report(PredicateRegistry.gate_a())
    summary = _summary(report, *references[0])

    assert summary.total == 6
    assert summary.pending == 1
    assert summary.resolved == 5
    assert summary.scored == 3
    assert summary.covered == 2
    assert summary.unscored == 2
    assert dict(summary.unscored_by_reason) == {
        "warm-up": 1,
        "method-returned-nonfinite": 1,
    }
    assert summary.coverage_ratio == pytest.approx(2 / 3)
    assert not hasattr(report, "coverage_ratio")


def test_distinct_targets_keep_independent_denominators_and_reasons() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    target_a = interval_columns(0.9)
    target_b = interval_columns(0.8)
    descriptor_a = _descriptor(
        GuaranteeClaim.TWO_SIDED_COVERAGE,
        GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        level=0.9,
    )
    descriptor_b = _descriptor(
        GuaranteeClaim.TWO_SIDED_COVERAGE,
        GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        level=0.8,
    )

    target_a_rows = [
        (
            _append(
                ledger,
                model="target-a-covered",
                bound_values={target_a[0]: 0.0, target_a[1]: 10.0},
                issuances={target_a: _issuance(descriptor_a)},
            ),
            target_a,
        ),
        (
            _append(
                ledger,
                model="target-a-uncovered",
                bound_values={target_a[0]: 0.0, target_a[1]: 9.0},
                issuances={target_a: _issuance(descriptor_a)},
            ),
            target_a,
        ),
        (
            _append(
                ledger,
                model="target-a-unscored",
                bound_values={target_a[0]: math.nan, target_a[1]: math.nan},
                issuances={
                    target_a: _issuance(
                        descriptor_a,
                        finite=False,
                        reason="target-a-nonfinite",
                        ready=True,
                    )
                },
            ),
            target_a,
        ),
    ]
    target_b_rows = [
        (
            _append(
                ledger,
                model="target-b-covered-a",
                bound_values={target_b[0]: 0.0, target_b[1]: 10.0},
                issuances={target_b: _issuance(descriptor_b)},
            ),
            target_b,
        ),
        (
            _append(
                ledger,
                model="target-b-covered-b",
                bound_values={target_b[0]: 5.0, target_b[1]: 15.0},
                issuances={target_b: _issuance(descriptor_b)},
            ),
            target_b,
        ),
    ]
    ledger.apply_resolutions(
        {key: 10.0 for key, _ in (*target_a_rows, *target_b_rows)},
        origin=SCORE_ORIGIN,
    )

    report = ledger.coverage_report(PredicateRegistry.gate_a())
    summary_a = _summary(report, *target_a_rows[0])
    summary_b = _summary(report, *target_b_rows[0])

    assert len(report.summaries) == 2
    assert (summary_a.total, summary_a.scored, summary_a.covered, summary_a.unscored) == (
        3,
        2,
        1,
        1,
    )
    assert dict(summary_a.unscored_by_reason) == {"target-a-nonfinite": 1}
    assert summary_a.coverage_ratio == 0.5
    assert (summary_b.total, summary_b.scored, summary_b.covered, summary_b.unscored) == (
        2,
        2,
        2,
        0,
    )
    assert dict(summary_b.unscored_by_reason) == {}
    assert summary_b.coverage_ratio == 1.0


def test_an_all_unscored_target_has_no_coverage_ratio() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    references = [
        _append_one_sided(
            ledger,
            model=f"warm-up-{index}",
            side=GuaranteedSide.UPPER,
            lower=0.0,
            upper=math.nan,
            finite=False,
            reason="calibrator-needs-more-scores",
            ready=False,
        )
        for index in range(2)
    ]
    ledger.apply_resolutions(
        {key: 10.0 for key, _ in references},
        origin=SCORE_ORIGIN,
    )

    report = ledger.coverage_report(PredicateRegistry.gate_a())
    summary = _summary(report, *references[0])

    assert (summary.total, summary.resolved, summary.scored, summary.covered) == (2, 2, 0, 0)
    assert summary.unscored == 2
    assert summary.coverage_ratio is None


def test_unscored_precedence_is_total_and_preserves_the_issuance_reason() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    quantile: BoundKey = (quantile_column(0.5),)
    pending_key = _append(
        ledger,
        model="pending-risk-nonfinite",
        bound_values={quantile[0]: math.nan},
        issuances={
            quantile: _issuance(
                _descriptor(
                    GuaranteeClaim.RISK_CONTROL,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    level=0.1,
                ),
                finite=False,
                reason="not-ready-yet",
                ready=False,
            )
        },
    )
    warm_up_key = _append(
        ledger,
        model="resolved-risk-warm-up",
        bound_values={quantile[0]: math.nan},
        issuances={
            quantile: _issuance(
                _descriptor(
                    GuaranteeClaim.RISK_CONTROL,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    level=0.1,
                ),
                finite=False,
                reason="calibrator-needs-28-resolved-scores",
                ready=False,
            )
        },
    )
    ready_nonfinite_key = _append(
        ledger,
        model="resolved-risk-ready-nonfinite",
        bound_values={quantile[0]: math.nan},
        issuances={
            quantile: _issuance(
                _descriptor(
                    GuaranteeClaim.RISK_CONTROL,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    level=0.1,
                ),
                finite=False,
                reason="method-returned-nonfinite",
                ready=True,
            )
        },
    )
    none_nonfinite_key = _append(
        ledger,
        model="resolved-none-nonfinite",
        bound_values={quantile[0]: math.nan},
        issuances={
            quantile: _issuance(
                _descriptor(GuaranteeClaim.NONE, None, level=0.5),
                finite=False,
                reason="clamp-produced-nonfinite",
            )
        },
    )
    none_key = _append(
        ledger,
        model="resolved-none-finite",
        bound_values={quantile[0]: 10.0},
        issuances={quantile: _issuance(_descriptor(GuaranteeClaim.NONE, None, level=0.5))},
    )
    risk_key = _append(
        ledger,
        model="resolved-risk-unregistered",
        bound_values={quantile[0]: 10.0},
        issuances={
            quantile: _issuance(
                _descriptor(
                    GuaranteeClaim.RISK_CONTROL,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    level=0.1,
                )
            )
        },
    )
    ledger.apply_resolutions(
        {
            warm_up_key: 10.0,
            ready_nonfinite_key: 10.0,
            none_nonfinite_key: 10.0,
            none_key: 10.0,
            risk_key: 10.0,
        },
        origin=SCORE_ORIGIN,
    )

    report = ledger.coverage_report(PredicateRegistry.gate_a())
    pending = _outcome(report, pending_key, quantile)
    warm_up = _outcome(report, warm_up_key, quantile)
    ready_nonfinite = _outcome(report, ready_nonfinite_key, quantile)
    none_nonfinite = _outcome(report, none_nonfinite_key, quantile)
    none = _outcome(report, none_key, quantile)
    risk = _outcome(report, risk_key, quantile)

    assert (pending.resolved, pending.scored, pending.covered, pending.unscored_reason) == (
        False,
        False,
        None,
        None,
    )
    assert warm_up.unscored_reason == "warm-up"
    assert ready_nonfinite.unscored_reason == "method-returned-nonfinite"
    assert none_nonfinite.unscored_reason == "clamp-produced-nonfinite"
    assert none.unscored_reason == "not-engine-calibrated"
    assert risk.unscored_reason == "predicate-unregistered"
    resolved_unscored = [
        outcome for outcome in report.outcomes if outcome.resolved and not outcome.scored
    ]
    assert resolved_unscored
    assert all(
        isinstance(outcome.unscored_reason, str) and outcome.unscored_reason.strip()
        for outcome in resolved_unscored
    )
    assert (report.bound_count, report.pending_bound_count, report.resolved_bound_count) == (
        6,
        1,
        5,
    )
    assert (
        report.scored_bound_count,
        report.covered_bound_count,
        report.unscored_bound_count,
    ) == (0, 0, 5)
    assert dict(report.unscored_by_reason) == {
        "warm-up": 1,
        "method-returned-nonfinite": 1,
        "clamp-produced-nonfinite": 1,
        "not-engine-calibrated": 1,
        "predicate-unregistered": 1,
    }


@pytest.mark.parametrize(
    ("side", "lower", "upper", "actual", "expected"),
    [
        (GuaranteedSide.LOWER, 10.0, 5.0, 10.0, True),
        (GuaranteedSide.LOWER, 11.0, 100.0, 10.0, False),
        (GuaranteedSide.UPPER, 15.0, 10.0, 10.0, True),
        (GuaranteedSide.UPPER, 0.0, 9.0, 10.0, False),
    ],
    ids=["lower-inclusive", "lower-miss", "upper-inclusive", "upper-miss"],
)
def test_one_sided_predicate_uses_only_its_declared_side(
    side: GuaranteedSide,
    lower: float,
    upper: float,
    actual: float,
    expected: bool,
) -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    key, bound_key = _append_one_sided(
        ledger,
        model="one-sided",
        side=side,
        lower=lower,
        upper=upper,
    )
    ledger.apply_resolutions({key: actual}, origin=SCORE_ORIGIN)

    outcome = _outcome(ledger.coverage_report(PredicateRegistry.gate_a()), key, bound_key)

    assert outcome.scored is True
    assert outcome.value == float(expected)
    assert outcome.covered is expected


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(10.0, True), (20.0, True), (9.0, False), (21.0, False)],
    ids=["lower-inclusive", "upper-inclusive", "below", "above"],
)
def test_two_sided_predicate_is_inclusive(actual: float, expected: bool) -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    interval: BoundKey = interval_columns(0.9)
    key = _append(
        ledger,
        model="two-sided",
        bound_values={interval[0]: 10.0, interval[1]: 20.0},
        issuances={
            interval: _issuance(
                _descriptor(
                    GuaranteeClaim.TWO_SIDED_COVERAGE,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                )
            )
        },
    )
    ledger.apply_resolutions({key: actual}, origin=SCORE_ORIGIN)

    outcome = _outcome(ledger.coverage_report(PredicateRegistry.gate_a()), key, interval)

    assert outcome.scored is True
    assert outcome.value == float(expected)
    assert outcome.covered is expected


def test_risk_level_remains_independent_from_the_forecast_column_suffix() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    quantile: BoundKey = (quantile_column(0.5),)
    key = _append(
        ledger,
        model="risk-level",
        bound_values={quantile[0]: 10.0},
        issuances={
            quantile: _issuance(
                _descriptor(
                    GuaranteeClaim.RISK_CONTROL,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    level=0.1,
                )
            )
        },
    )
    ledger.apply_resolutions({key: 10.0}, origin=SCORE_ORIGIN)

    outcome = _outcome(ledger.coverage_report(PredicateRegistry.gate_a()), key, quantile)

    assert outcome.target.descriptor.level == 0.1
    assert outcome.bound_key == ("quantile_0.5",)
    assert outcome.unscored_reason == "predicate-unregistered"


def test_coverage_report_outcomes_summaries_and_reason_maps_are_immutable() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    quantile: BoundKey = (quantile_column(0.5),)
    key = _append(
        ledger,
        model="immutable-report",
        bound_values={quantile[0]: 10.0},
        issuances={quantile: _issuance(_descriptor(GuaranteeClaim.NONE, None, level=0.5))},
    )
    ledger.apply_resolutions({key: 10.0}, origin=SCORE_ORIGIN)
    report = ledger.coverage_report(PredicateRegistry.gate_a())
    outcome = _outcome(report, key, quantile)
    summary = report.summaries[outcome.target]

    assert isinstance(report.outcomes, tuple)
    with pytest.raises(FrozenInstanceError):
        cast(Any, report).bound_count = 0
    with pytest.raises(FrozenInstanceError):
        cast(Any, outcome).covered = True
    with pytest.raises(FrozenInstanceError):
        cast(Any, outcome.target).bound_key = ("quantile_0.9",)
    with pytest.raises(FrozenInstanceError):
        cast(Any, summary).scored = 1
    with pytest.raises(TypeError):
        cast(Any, report.summaries)[outcome.target] = summary
    with pytest.raises(TypeError):
        cast(Any, report.unscored_by_reason)["mutated"] = 1
    with pytest.raises(TypeError):
        cast(Any, summary.unscored_by_reason)["mutated"] = 1


def test_coverage_summary_and_report_reject_internally_inconsistent_counts() -> None:
    ledger = Ledger(session=_session(), calendar=CALENDAR)
    quantile: BoundKey = (quantile_column(0.5),)
    key = _append(
        ledger,
        model="invariant-report",
        bound_values={quantile[0]: 10.0},
        issuances={quantile: _issuance(_descriptor(GuaranteeClaim.NONE, None, level=0.5))},
    )
    ledger.apply_resolutions({key: 10.0}, origin=SCORE_ORIGIN)
    report = ledger.coverage_report(PredicateRegistry.gate_a())
    summary = next(iter(report.summaries.values()))

    with pytest.raises(LedgerError, match="total must equal"):
        replace(summary, total=summary.total + 1)
    with pytest.raises(LedgerError, match="reason counts"):
        replace(summary, unscored_by_reason={"wrong": summary.unscored + 1})
    with pytest.raises(LedgerError, match="counts must match its outcomes"):
        replace(
            report,
            bound_count=report.bound_count + 1,
            pending_bound_count=report.pending_bound_count + 1,
        )
