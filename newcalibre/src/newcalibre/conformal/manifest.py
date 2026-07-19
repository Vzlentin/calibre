"""Declare the immutable capabilities and obligations of conformal methods."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

from newcalibre.domain import EmissionScope, GuaranteeClaim, GuaranteeCurrency, GuaranteeType
from newcalibre.domain.descriptor import GuaranteeDescriptorError


class MethodManifestError(ValueError):
    """Report an invalid conformal-method declaration."""


class EmissionForm(StrEnum):
    """Name the supported shapes of one emitted bound pair."""

    TWO_SIDED = "two-sided"
    ONE_SIDED_LOWER = "one-sided-lower"
    ONE_SIDED_UPPER = "one-sided-upper"


class AssumptionClass(StrEnum):
    """Name the mathematical assumption class carried by a method claim."""

    EXCHANGEABLE = "exchangeable"
    WEIGHTED = "weighted"
    SEQUENTIAL_ADAPTIVE = "sequential-adaptive"


class CensoringPolicy(StrEnum):
    """Name how a method admits censoring facts into score construction."""

    REQUIRES_UNCENSORED = "requires-uncensored"
    CONSUMES_CENSORING_FACTS = "consumes-censoring-facts"
    IMPUTATION_CONSUMER = "imputation-consumer"


class PostWarmupNonFinite(StrEnum):
    """Declare whether non-finite bounds may occur after readiness."""

    FORBIDDEN = "forbidden"
    ALLOWED_WITH_ATTRIBUTION = "allowed-with-attribution"


class ClampGuaranteeImpact(StrEnum):
    """Name the mathematical effect of a configured bound clamp."""

    VOIDS_CLAIM = "voids-claim"
    CONSERVATIVE_WIDENING = "conservative-widening"


class JointClaim(StrEnum):
    """Name the only admissible cross-row claim declarations."""

    NONE = "none"
    CLASS_CONDITIONAL = "class-conditional"


@dataclass(frozen=True, slots=True)
class GuaranteeDeclaration:
    """Declare one claim and currency a conformal method may issue."""

    claim: GuaranteeClaim
    currency: GuaranteeCurrency
    declared_slack: float | None = None
    loss_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, GuaranteeClaim):
            raise MethodManifestError("guarantee claim must be a GuaranteeClaim")
        if self.claim is GuaranteeClaim.NONE:
            raise MethodManifestError("the none claim is produced by attribution, not declared")
        if not isinstance(self.currency, GuaranteeCurrency):
            raise MethodManifestError("guarantee currency must be a GuaranteeCurrency")
        try:
            guarantee_type = GuaranteeType(
                claim=self.claim,
                currency=self.currency,
                declared_slack=self.declared_slack,
            )
        except GuaranteeDescriptorError as error:
            raise MethodManifestError(str(error)) from error
        object.__setattr__(self, "declared_slack", guarantee_type.declared_slack)

        if self.claim is GuaranteeClaim.RISK_CONTROL:
            _require_text(self.loss_name, name="risk-control loss name")
        elif self.loss_name is not None:
            raise MethodManifestError("only risk-control guarantees may name a loss")


@dataclass(frozen=True, slots=True)
class ClampDeclaration:
    """Declare one optional clamp and its mathematical guarantee impact."""

    name: str
    guarantee_impact: ClampGuaranteeImpact

    def __post_init__(self) -> None:
        _require_text(self.name, name="clamp name", trimmed=True)
        if not isinstance(self.guarantee_impact, ClampGuaranteeImpact):
            raise MethodManifestError("clamp guarantee impact must be a ClampGuaranteeImpact")


@dataclass(frozen=True, slots=True)
class MethodManifest:
    """Publish every stable declaration required to host one conformal method."""

    name: str
    emission_form: EmissionForm
    emission_scope: EmissionScope
    guarantees: tuple[GuaranteeDeclaration, ...]
    assumption_class: AssumptionClass
    minimum_calibration_scores: int
    order_sensitive: bool
    censoring_policy: CensoringPolicy
    imputation_policy: str | None
    state_bound: int
    state_schema_version: int
    consumes_calibration_context: bool
    hosted_submodels: tuple[str, ...]
    requires_fitted_values: bool
    post_warmup_non_finite: PostWarmupNonFinite
    clamps: tuple[ClampDeclaration, ...]
    joint_claim: JointClaim

    def __post_init__(self) -> None:
        _require_text(self.name, name="method name", trimmed=True)
        _require_enum(self.emission_form, EmissionForm, name="emission form")
        _require_enum(self.emission_scope, EmissionScope, name="emission scope")
        _validate_guarantees(self.guarantees)
        _require_enum(self.assumption_class, AssumptionClass, name="assumption class")
        _require_nonnegative_integer(
            self.minimum_calibration_scores,
            name="minimum calibration scores",
        )
        _require_bool(self.order_sensitive, name="order sensitivity")
        _require_enum(self.censoring_policy, CensoringPolicy, name="censoring policy")
        _validate_imputation_policy(self.censoring_policy, self.imputation_policy)
        _require_nonnegative_integer(self.state_bound, name="state bound")
        _require_positive_integer(self.state_schema_version, name="state schema version")
        _require_bool(
            self.consumes_calibration_context,
            name="consumes calibration context",
        )
        _validate_names(self.hosted_submodels, name="hosted sub-model")
        _require_bool(self.requires_fitted_values, name="requires fitted values")
        _require_enum(
            self.post_warmup_non_finite,
            PostWarmupNonFinite,
            name="post warmup non finite",
        )
        _validate_clamps(self.clamps)
        _require_enum(self.joint_claim, JointClaim, name="joint claim")
        _validate_joint_claim(self)


def _validate_guarantees(guarantees: object) -> None:
    if not isinstance(guarantees, tuple) or not guarantees:
        raise MethodManifestError("guarantees must be a non-empty tuple")
    if any(not isinstance(value, GuaranteeDeclaration) for value in guarantees):
        raise MethodManifestError("every guarantee declaration must be a GuaranteeDeclaration")
    if len(set(guarantees)) != len(guarantees):
        raise MethodManifestError("guarantee declarations must be unique")


def _validate_clamps(clamps: object) -> None:
    if not isinstance(clamps, tuple):
        raise MethodManifestError("clamps must be a tuple")
    if any(not isinstance(value, ClampDeclaration) for value in clamps):
        raise MethodManifestError("every clamp declaration must be a ClampDeclaration")
    names = tuple(value.name for value in clamps if isinstance(value, ClampDeclaration))
    if len(set(names)) != len(names):
        raise MethodManifestError("clamp names must be unique")


def _validate_names(values: object, *, name: str) -> None:
    if not isinstance(values, tuple):
        raise MethodManifestError(f"{name}s must be a tuple")
    for value in values:
        _require_text(value, name=f"{name} name", trimmed=True)
    if len(set(values)) != len(values):
        raise MethodManifestError(f"{name} names must be unique")


def _validate_imputation_policy(
    censoring_policy: CensoringPolicy,
    imputation_policy: object,
) -> None:
    if censoring_policy is CensoringPolicy.IMPUTATION_CONSUMER:
        try:
            _require_text(
                imputation_policy,
                name="imputation-consumer policy",
                trimmed=True,
            )
        except MethodManifestError as error:
            raise MethodManifestError(
                "imputation-consumer requires a named imputation policy"
            ) from error
    elif imputation_policy is not None:
        raise MethodManifestError(
            "an imputation policy is valid only for imputation-consumer methods"
        )


def _validate_joint_claim(manifest: MethodManifest) -> None:
    has_conditional_guarantee = any(
        declaration.claim is GuaranteeClaim.CLASS_CONDITIONAL_COVERAGE
        for declaration in manifest.guarantees
    )
    if manifest.joint_claim is JointClaim.CLASS_CONDITIONAL:
        if not manifest.consumes_calibration_context:
            raise MethodManifestError(
                "class-conditional joint claim requires calibration-context consumption"
            )
        if not has_conditional_guarantee:
            raise MethodManifestError(
                "class-conditional joint claim requires a class-conditional guarantee"
            )
    elif has_conditional_guarantee:
        raise MethodManifestError(
            "a class-conditional guarantee requires joint_claim class-conditional"
        )


def _require_enum(value: object, enum_type: type[StrEnum], *, name: str) -> None:
    if not isinstance(value, enum_type):
        raise MethodManifestError(f"{name} must be a {enum_type.__name__}")


def _require_bool(value: object, *, name: str) -> None:
    if not isinstance(value, bool):
        raise MethodManifestError(f"{name} must be a boolean")


def _require_nonnegative_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise MethodManifestError(f"{name} must be a nonnegative integer")


def _require_positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise MethodManifestError(f"{name} must be a positive integer")


def _require_text(value: object, *, name: str, trimmed: bool = False) -> str:
    if not isinstance(value, str) or not value or (trimmed and value != value.strip()):
        qualifier = " non-empty trimmed" if trimmed else " non-empty"
        raise MethodManifestError(f"{name} must be a{qualifier} string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise MethodManifestError(f"{name} must be valid UTF-8") from error
    return value
