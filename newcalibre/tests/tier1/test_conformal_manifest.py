"""Exercise the closed conformal-method manifest contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict

from newcalibre.conformal import (
    AssumptionClass,
    CensoringPolicy,
    ClampDeclaration,
    ClampGuaranteeImpact,
    ConservativeRankRequirement,
    EmissionForm,
    FixedCountRequirement,
    GuaranteeDeclaration,
    JointClaim,
    MethodManifest,
    MethodManifestError,
    PostWarmupNonFinite,
)
from newcalibre.domain import EmissionScope, GuaranteeClaim, GuaranteeCurrency

pytestmark = pytest.mark.tier1


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: float = 0.9


def _guarantee(
    *,
    claim: GuaranteeClaim = GuaranteeClaim.ONE_SIDED_COVERAGE,
    currency: GuaranteeCurrency = GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
    declared_slack: float | None = None,
    loss_name: str | None = None,
) -> GuaranteeDeclaration:
    return GuaranteeDeclaration(
        claim=claim,
        currency=currency,
        declared_slack=declared_slack,
        loss_name=loss_name,
    )


def _manifest(**overrides: object) -> MethodManifest:
    values: dict[str, object] = {
        "name": "fixture",
        "emission_form": EmissionForm.ONE_SIDED_UPPER,
        "emission_scope": EmissionScope.PER_STEP,
        "guarantees": (_guarantee(),),
        "assumption_class": AssumptionClass.EXCHANGEABLE,
        "calibration_requirement": FixedCountRequirement(1),
        "order_sensitive": False,
        "censoring_policy": CensoringPolicy.REQUIRES_UNCENSORED,
        "imputation_policy": None,
        "state_bound": 32,
        "state_schema_version": 1,
        "consumes_calibration_context": False,
        "hosted_submodels": (),
        "requires_fitted_values": False,
        "post_warmup_non_finite": PostWarmupNonFinite.FORBIDDEN,
        "clamps": (
            ClampDeclaration(
                name="upper-cap",
                guarantee_impact=ClampGuaranteeImpact.VOIDS_CLAIM,
            ),
            ClampDeclaration(
                name="upper-floor",
                guarantee_impact=ClampGuaranteeImpact.CONSERVATIVE_WIDENING,
            ),
        ),
        "joint_claim": JointClaim.NONE,
    }
    values.update(overrides)
    return MethodManifest(**values)  # type: ignore[arg-type]


def test_complete_manifest_is_immutable_and_carries_every_declaration() -> None:
    manifest = _manifest(
        emission_scope=EmissionScope.WINDOW_SUM,
        guarantees=(
            _guarantee(),
            _guarantee(
                claim=GuaranteeClaim.RISK_CONTROL,
                currency=GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK,
                declared_slack=0.05,
                loss_name="pinball",
            ),
        ),
        assumption_class=AssumptionClass.WEIGHTED,
        calibration_requirement=FixedCountRequirement(10),
        order_sensitive=True,
        censoring_policy=CensoringPolicy.CONSUMES_CENSORING_FACTS,
        state_bound=128,
        state_schema_version=3,
        consumes_calibration_context=True,
        hosted_submodels=("score-forecaster",),
        requires_fitted_values=True,
        post_warmup_non_finite=PostWarmupNonFinite.ALLOWED_WITH_ATTRIBUTION,
    )

    assert manifest.name == "fixture"
    assert manifest.emission_form is EmissionForm.ONE_SIDED_UPPER
    assert manifest.emission_scope is EmissionScope.WINDOW_SUM
    assert manifest.guarantees[1].loss_name == "pinball"
    assert manifest.assumption_class is AssumptionClass.WEIGHTED
    assert manifest.minimum_calibration_scores(_Config()) == 10
    assert manifest.order_sensitive
    assert manifest.censoring_policy is CensoringPolicy.CONSUMES_CENSORING_FACTS
    assert manifest.state_bound == 128
    assert manifest.state_schema_version == 3
    assert manifest.consumes_calibration_context
    assert manifest.hosted_submodels == ("score-forecaster",)
    assert manifest.requires_fitted_values
    assert manifest.post_warmup_non_finite is PostWarmupNonFinite.ALLOWED_WITH_ATTRIBUTION
    assert {clamp.name for clamp in manifest.clamps} == {"upper-cap", "upper-floor"}
    assert manifest.joint_claim is JointClaim.NONE

    with pytest.raises(FrozenInstanceError):
        cast(Any, manifest).name = "changed"


def test_manifest_vocabularies_are_closed() -> None:
    assert {value.value for value in EmissionForm} == {
        "two-sided",
        "one-sided-lower",
        "one-sided-upper",
    }
    assert {value.value for value in AssumptionClass} == {
        "exchangeable",
        "weighted",
        "sequential-adaptive",
    }
    assert {value.value for value in CensoringPolicy} == {
        "requires-uncensored",
        "consumes-censoring-facts",
        "imputation-consumer",
    }
    assert {value.value for value in JointClaim} == {"none", "class-conditional"}
    assert {value.value for value in PostWarmupNonFinite} == {
        "forbidden",
        "allowed-with-attribution",
    }
    assert {value.value for value in ClampGuaranteeImpact} == {
        "voids-claim",
        "conservative-widening",
    }

    for vocabulary, unsupported in (
        (EmissionForm, "quantiles"),
        (AssumptionClass, "asymptotic"),
        (CensoringPolicy, "ignore-censoring"),
        (JointClaim, "simultaneous"),
        (PostWarmupNonFinite, "silent-fallback"),
        (ClampGuaranteeImpact, "unchanged"),
    ):
        with pytest.raises(ValueError):
            vocabulary(unsupported)


def test_manifest_rejects_raw_closed_vocabulary_values() -> None:
    fields = {
        "emission_form": "one-sided-upper",
        "emission_scope": "per-step",
        "assumption_class": "exchangeable",
        "censoring_policy": "requires-uncensored",
        "post_warmup_non_finite": "forbidden",
        "joint_claim": "none",
    }
    for field, value in fields.items():
        with pytest.raises(MethodManifestError, match=field.replace("_", " ")):
            _manifest(**{field: value})


def test_guarantees_reject_unsupported_raw_claims_and_currencies() -> None:
    with pytest.raises(MethodManifestError, match="claim"):
        GuaranteeDeclaration(
            claim=cast(Any, "simultaneous-coverage"),
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        )
    with pytest.raises(MethodManifestError, match="currency"):
        GuaranteeDeclaration(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=cast(Any, "asymptotic"),
        )


def test_risk_control_declarations_are_complete_and_only_risk_names_a_loss() -> None:
    with pytest.raises(MethodManifestError, match="risk-control.*loss"):
        _guarantee(claim=GuaranteeClaim.RISK_CONTROL)
    with pytest.raises(MethodManifestError, match="only risk-control"):
        _guarantee(loss_name="pinball")


def test_approximate_guarantees_validate_declared_slack() -> None:
    declaration = _guarantee(
        currency=GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK,
        declared_slack=0.1,
    )
    assert declaration.declared_slack == 0.1

    with pytest.raises(MethodManifestError, match="declared slack"):
        _guarantee(currency=GuaranteeCurrency.APPROXIMATE_WITH_DECLARED_SLACK)
    with pytest.raises(MethodManifestError, match="declared slack"):
        _guarantee(declared_slack=0.1)


def test_manifest_rejects_duplicate_guarantees_clamps_and_hosted_models() -> None:
    guarantee = _guarantee()
    with pytest.raises(MethodManifestError, match="guarantee declarations must be unique"):
        _manifest(guarantees=(guarantee, guarantee))
    with pytest.raises(MethodManifestError, match="clamp names must be unique"):
        _manifest(
            clamps=(
                ClampDeclaration("cap", ClampGuaranteeImpact.VOIDS_CLAIM),
                ClampDeclaration("cap", ClampGuaranteeImpact.CONSERVATIVE_WIDENING),
            )
        )
    with pytest.raises(MethodManifestError, match="hosted sub-model names must be unique"):
        _manifest(hosted_submodels=("model", "model"))


def test_imputation_consumer_requires_exactly_one_named_policy() -> None:
    with pytest.raises(MethodManifestError, match="requires a named imputation policy"):
        _manifest(censoring_policy=CensoringPolicy.IMPUTATION_CONSUMER)
    with pytest.raises(MethodManifestError, match="only.*imputation-consumer"):
        _manifest(imputation_policy="stockout-repair-v1")
    assert (
        _manifest(
            censoring_policy=CensoringPolicy.IMPUTATION_CONSUMER,
            imputation_policy="stockout-repair-v1",
        ).imputation_policy
        == "stockout-repair-v1"
    )


def test_class_conditional_joint_claim_requires_context_and_matching_guarantee() -> None:
    conditional = _guarantee(claim=GuaranteeClaim.CLASS_CONDITIONAL_COVERAGE)
    with pytest.raises(MethodManifestError, match="requires calibration-context consumption"):
        _manifest(joint_claim=JointClaim.CLASS_CONDITIONAL, guarantees=(conditional,))
    with pytest.raises(MethodManifestError, match="requires.*class-conditional.*guarantee"):
        _manifest(
            joint_claim=JointClaim.CLASS_CONDITIONAL,
            consumes_calibration_context=True,
        )
    with pytest.raises(MethodManifestError, match="requires joint_claim"):
        _manifest(guarantees=(conditional,), consumes_calibration_context=True)

    manifest = _manifest(
        joint_claim=JointClaim.CLASS_CONDITIONAL,
        consumes_calibration_context=True,
        guarantees=(conditional,),
    )
    assert manifest.joint_claim is JointClaim.CLASS_CONDITIONAL


def test_manifest_rejects_malformed_identity_counts_flags_and_members() -> None:
    for name in ("", " fixture", "fixture ", "\ud800"):
        with pytest.raises(MethodManifestError, match="method name"):
            _manifest(name=name)
    for field, value in (
        ("state_bound", -1),
        ("state_bound", 1.5),
        ("state_schema_version", 0),
        ("state_schema_version", True),
    ):
        with pytest.raises(MethodManifestError, match=field.replace("_", " ")):
            _manifest(**{field: value})
    for value in (-1, True):
        with pytest.raises(MethodManifestError, match="fixed calibration count"):
            FixedCountRequirement(value)  # type: ignore[arg-type]
    with pytest.raises(MethodManifestError, match="calibration requirement"):
        _manifest(calibration_requirement=cast(Any, object()))
    for field, message in (
        ("order_sensitive", "order sensitivity"),
        ("consumes_calibration_context", "consumes calibration context"),
        ("requires_fitted_values", "requires fitted values"),
    ):
        with pytest.raises(MethodManifestError, match=message):
            _manifest(**{field: 1})
    with pytest.raises(MethodManifestError, match="guarantees must be a non-empty tuple"):
        _manifest(guarantees=[])
    with pytest.raises(MethodManifestError, match="hosted sub-model"):
        _manifest(hosted_submodels=("",))
    with pytest.raises(MethodManifestError, match="clamp declaration"):
        _manifest(clamps=(cast(Any, "cap"),))


def test_dynamic_conservative_rank_requirement_matches_the_strict_boundary() -> None:
    requirement = ConservativeRankRequirement()

    assert requirement.minimum_scores(_Config(coverage=0.9)) == 10
    assert requirement.minimum_scores(_Config(coverage=0.8)) == 5
    assert (
        _manifest(calibration_requirement=requirement).minimum_calibration_scores(
            _Config(coverage=0.9)
        )
        == 10
    )


def test_manifest_rejects_joint_or_simultaneous_claim_forgery() -> None:
    manifest = _manifest()
    with pytest.raises(MethodManifestError, match="joint claim"):
        replace(manifest, joint_claim=cast(Any, "lattice-wide"))
    with pytest.raises(MethodManifestError, match="claim"):
        replace(
            manifest,
            guarantees=(
                GuaranteeDeclaration(
                    claim=cast(Any, "simultaneous-coverage"),
                    currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                ),
            ),
        )
