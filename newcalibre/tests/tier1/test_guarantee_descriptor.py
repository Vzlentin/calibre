"""Exercise the closed guarantee-descriptor domain contract."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from newcalibre.domain import (
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeDescriptorError,
    GuaranteeType,
    ScoredSeries,
)


def _descriptor(
    *,
    guarantee_type: GuaranteeType | None = None,
    level: float = 0.9,
    scored_series: ScoredSeries = ScoredSeries.DEMAND_HONEST,
    window: EmissionScope = EmissionScope.PER_STEP,
    scope: DecisionScope | None = None,
) -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=guarantee_type
        or GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        level=level,
        scored_series=scored_series,
        window=window,
        scope=scope
        or DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )


def test_claim_and_currency_vocabularies_are_closed_and_complete() -> None:
    assert {claim.value for claim in GuaranteeClaim} == {
        "one-sided-coverage",
        "two-sided-coverage",
        "risk-control",
        "class-conditional-coverage",
        "none",
    }
    assert {currency.value for currency in GuaranteeCurrency} == {
        "finite-sample-marginal",
        "long-run-pathwise",
        "approximate-with-declared-slack",
    }

    with pytest.raises(ValueError):
        GuaranteeClaim("simultaneous-coverage")
    with pytest.raises(ValueError):
        GuaranteeCurrency("asymptotic")


def test_descriptor_carries_every_declared_field_and_is_immutable() -> None:
    descriptor = _descriptor(window=EmissionScope.WINDOW_SUM)

    assert descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
    assert descriptor.type.currency is GuaranteeCurrency.FINITE_SAMPLE_MARGINAL
    assert descriptor.type.declared_slack is None
    assert descriptor.level == 0.9
    assert descriptor.scored_series is ScoredSeries.DEMAND_HONEST
    assert descriptor.window is EmissionScope.WINDOW_SUM
    assert descriptor.scope.kind is DecisionScopeKind.PER_DECISION_NODE
    assert descriptor.scope.class_system_name is None

    with pytest.raises(FrozenInstanceError):
        cast(Any, descriptor).level = 0.8


def test_none_claim_requires_not_applicable_currency_and_no_slack() -> None:
    guarantee_type = GuaranteeType(
        claim=GuaranteeClaim.NONE,
        currency=None,
        declared_slack=None,
    )
    assert guarantee_type.currency is None

    with pytest.raises(GuaranteeDescriptorError, match="currency must be not applicable"):
        GuaranteeType(
            claim=GuaranteeClaim.NONE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        )
    with pytest.raises(GuaranteeDescriptorError, match="requires a currency"):
        GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=None,
            declared_slack=None,
        )


@pytest.mark.parametrize("slack", [0.0, 0.125])
def test_approximate_currency_requires_finite_nonnegative_declared_slack(slack: float) -> None:
    guarantee_type = GuaranteeType(
        claim=GuaranteeClaim.RISK_CONTROL,
        currency=GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK,
        declared_slack=slack,
    )
    assert guarantee_type.declared_slack == slack


@pytest.mark.parametrize("slack", [None, -0.1, math.nan, math.inf, -math.inf, True])
def test_approximate_currency_rejects_missing_or_invalid_slack(slack: Any) -> None:
    with pytest.raises(GuaranteeDescriptorError, match="declared slack"):
        GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK,
            declared_slack=slack,
        )


def test_nonapproximate_currencies_and_none_reject_declared_slack() -> None:
    for claim, currency in (
        (GuaranteeClaim.ONE_SIDED_COVERAGE, GuaranteeCurrency.FINITE_SAMPLE_MARGINAL),
        (GuaranteeClaim.TWO_SIDED_COVERAGE, GuaranteeCurrency.LONG_RUN_PATHWISE),
        (GuaranteeClaim.NONE, None),
    ):
        with pytest.raises(GuaranteeDescriptorError, match="only.*approximate"):
            GuaranteeType(claim=claim, currency=currency, declared_slack=0.0)


def test_class_conditional_claim_requires_a_named_class_system() -> None:
    conditional_type = GuaranteeType(
        claim=GuaranteeClaim.CLASS_CONDITIONAL_COVERAGE,
        currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        declared_slack=None,
    )
    scope = DecisionScope(
        kind=DecisionScopeKind.PER_DECISION_NODE,
        class_system_name="hierarchy-levels-v1",
    )

    descriptor = _descriptor(guarantee_type=conditional_type, scope=scope)
    assert descriptor.scope.class_system_name == "hierarchy-levels-v1"

    with pytest.raises(GuaranteeDescriptorError, match="requires a named finite class system"):
        _descriptor(
            guarantee_type=conditional_type,
            scope=DecisionScope(
                kind=DecisionScopeKind.PER_DECISION_NODE,
                class_system_name=None,
            ),
        )


def test_non_class_conditional_claim_rejects_a_class_system() -> None:
    with pytest.raises(GuaranteeDescriptorError, match="only.*class-conditional"):
        _descriptor(
            scope=DecisionScope(
                kind=DecisionScopeKind.PER_DECISION_NODE,
                class_system_name="hierarchy-levels-v1",
            )
        )


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0])
def test_probability_like_level_accepts_finite_boundaries(level: float) -> None:
    assert _descriptor(level=level).level == level


@pytest.mark.parametrize("level", [-0.01, 1.01, math.nan, math.inf, -math.inf, True])
def test_probability_like_level_rejects_invalid_values(level: Any) -> None:
    with pytest.raises(GuaranteeDescriptorError, match="level"):
        _descriptor(level=level)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim", "one-sided-coverage"),
        ("currency", "finite-sample-marginal"),
    ],
)
def test_guarantee_type_rejects_raw_strings(field: str, value: str) -> None:
    arguments: dict[str, object] = {
        "claim": GuaranteeClaim.ONE_SIDED_COVERAGE,
        "currency": GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        "declared_slack": None,
    }
    arguments[field] = value
    with pytest.raises(GuaranteeDescriptorError, match=field):
        GuaranteeType(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "one-sided-coverage"),
        ("scored_series", "demand-honest"),
        ("window", "per-step"),
        ("scope", "per-decision-node"),
    ],
)
def test_descriptor_rejects_raw_strings_and_wrong_value_types(field: str, value: str) -> None:
    arguments: dict[str, object] = {
        "type": GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        "level": 0.9,
        "scored_series": ScoredSeries.DEMAND_HONEST,
        "window": EmissionScope.PER_STEP,
        "scope": DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    }
    arguments[field] = value
    with pytest.raises(GuaranteeDescriptorError, match=field.replace("_", " ")):
        GuaranteeDescriptor(**arguments)  # type: ignore[arg-type]


def test_scope_rejects_raw_kind_and_invalid_class_system_names() -> None:
    with pytest.raises(GuaranteeDescriptorError, match="scope kind"):
        DecisionScope(kind=cast(Any, "per-decision-node"), class_system_name=None)
    for name in ("", "\ud800", 1):
        with pytest.raises(GuaranteeDescriptorError, match="class system name"):
            DecisionScope(
                kind=DecisionScopeKind.PER_DECISION_NODE,
                class_system_name=cast(Any, name),
            )


def test_scored_series_emission_and_decision_scope_vocabularies_are_closed() -> None:
    assert {value.value for value in ScoredSeries} == {"demand-honest", "recorded-sales"}
    assert {value.value for value in EmissionScope} == {"per-step", "window-sum"}
    assert {value.value for value in DecisionScopeKind} == {"per-decision-node"}
