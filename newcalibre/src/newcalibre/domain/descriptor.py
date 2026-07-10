"""Declare immutable guarantee claims carried by issued decision bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real


class GuaranteeDescriptorError(ValueError):
    """Report an invalid guarantee type, scope, or descriptor value."""


class GuaranteeClaim(StrEnum):
    """Name every claim an issued decision bound may declare."""

    ONE_SIDED_COVERAGE = "one-sided-coverage"
    TWO_SIDED_COVERAGE = "two-sided-coverage"
    RISK_CONTROL = "risk-control"
    CLASS_CONDITIONAL_COVERAGE = "class-conditional-coverage"
    NONE = "none"


class GuaranteeCurrency(StrEnum):
    """Name every currency in which a non-empty claim may be stated."""

    FINITE_SAMPLE_MARGINAL = "finite-sample-marginal"
    LONG_RUN_PATHWISE = "long-run-pathwise"
    APPROXIMATE_WITH_DECLARED_SLACK = "approximate-with-declared-slack"


class ScoredSeries(StrEnum):
    """Name the realized series against which the guarantee is scored."""

    DEMAND_HONEST = "demand-honest"
    RECORDED_SALES = "recorded-sales"


class EmissionScope(StrEnum):
    """Name the time scope of each emitted guarantee."""

    PER_STEP = "per-step"
    WINDOW_SUM = "window-sum"


class DecisionScopeKind(StrEnum):
    """Name the admissible decision-node attachment scope."""

    PER_DECISION_NODE = "per-decision-node"


@dataclass(frozen=True, slots=True)
class GuaranteeType:
    """Pair one closed claim with its currency and optional declared slack."""

    claim: GuaranteeClaim
    currency: GuaranteeCurrency | None
    declared_slack: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, GuaranteeClaim):
            raise GuaranteeDescriptorError("claim must be a GuaranteeClaim")
        if self.currency is not None and not isinstance(self.currency, GuaranteeCurrency):
            raise GuaranteeDescriptorError("currency must be a GuaranteeCurrency or not applicable")

        if self.claim is GuaranteeClaim.NONE:
            if self.currency is not None:
                raise GuaranteeDescriptorError(
                    "currency must be not applicable when the claim is none"
                )
        elif self.currency is None:
            raise GuaranteeDescriptorError("a non-none claim requires a currency")

        if self.currency is GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK:
            slack = _finite_real(self.declared_slack, name="declared slack")
            if slack < 0.0:
                raise GuaranteeDescriptorError("declared slack must be nonnegative")
            object.__setattr__(self, "declared_slack", slack)
        elif self.declared_slack is not None:
            raise GuaranteeDescriptorError(
                "declared slack is only valid for approximate-with-declared-slack currency"
            )


@dataclass(frozen=True, slots=True)
class DecisionScope:
    """Attach a guarantee per decision node and optionally name its class system."""

    kind: DecisionScopeKind
    class_system_name: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DecisionScopeKind):
            raise GuaranteeDescriptorError("scope kind must be a DecisionScopeKind")
        if self.class_system_name is None:
            return
        if not isinstance(self.class_system_name, str) or not self.class_system_name.strip():
            raise GuaranteeDescriptorError("class system name must be a non-empty string")
        try:
            self.class_system_name.encode("utf-8")
        except UnicodeError as error:
            raise GuaranteeDescriptorError("class system name must be valid UTF-8") from error


@dataclass(frozen=True, slots=True)
class GuaranteeDescriptor:
    """State the complete guarantee carried by one issued decision bound."""

    type: GuaranteeType
    level: float
    scored_series: ScoredSeries
    window: EmissionScope
    scope: DecisionScope

    def __post_init__(self) -> None:
        if not isinstance(self.type, GuaranteeType):
            raise GuaranteeDescriptorError("type must be a GuaranteeType")
        level = _finite_real(self.level, name="level")
        if not 0.0 <= level <= 1.0:
            raise GuaranteeDescriptorError("level must lie between zero and one")
        if not isinstance(self.scored_series, ScoredSeries):
            raise GuaranteeDescriptorError("scored series must be a ScoredSeries")
        if not isinstance(self.window, EmissionScope):
            raise GuaranteeDescriptorError("window must be an EmissionScope")
        if not isinstance(self.scope, DecisionScope):
            raise GuaranteeDescriptorError("scope must be a DecisionScope")

        class_system_name = self.scope.class_system_name
        if self.type.claim is GuaranteeClaim.CLASS_CONDITIONAL_COVERAGE:
            if class_system_name is None:
                raise GuaranteeDescriptorError(
                    "class-conditional coverage requires a named finite class system"
                )
        elif class_system_name is not None:
            raise GuaranteeDescriptorError(
                "a class system is valid only for class-conditional coverage"
            )
        object.__setattr__(self, "level", level)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GuaranteeDescriptorError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise GuaranteeDescriptorError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise GuaranteeDescriptorError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized
