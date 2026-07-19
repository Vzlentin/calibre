"""Expose the stable conformal runtime and built-in method contracts."""

from collections.abc import Mapping

from pydantic import BaseModel

from newcalibre.conformal.manifest import (
    AssumptionClass,
    CalibrationRequirement,
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
from newcalibre.conformal.methods import (
    SEQUENTIAL_ADAPTIVE_PER_STEP,
    SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST,
    SPLIT_PER_STEP,
    SPLIT_PER_STEP_MANIFEST,
    SPLIT_WINDOW_SUM,
    SPLIT_WINDOW_SUM_MANIFEST,
    WEIGHTED_PER_STEP,
    WEIGHTED_PER_STEP_MANIFEST,
    SequentialAdaptiveConformalRuntime,
    SequentialAdaptivePerStepConfig,
    SplitConformalRuntime,
    SplitPerStepConfig,
    SplitWindowSumConfig,
    WeightedConformalRuntime,
    WeightedPerStepConfig,
    build_sequential_adaptive_per_step,
    build_split_per_step,
    build_split_window_sum,
    build_weighted_per_step,
)
from newcalibre.conformal.registry import ConformalRegistry, ConformalRegistryError
from newcalibre.conformal.runtime import ConformalRuntime, require_calibration_context
from newcalibre.conformal.types import (
    METHOD_SCOPE_LABEL,
    CalibrationContext,
    CalibrationResult,
    Delivery,
    ForecastKey,
    IssuedBoundFacts,
    ObserveAnnotation,
    ObserveEffect,
    ResolvedObservation,
    RuntimeContractError,
    derive_partition_label,
)

_BUILTIN_METHODS = ConformalRegistry()
_BUILTIN_METHODS.register(
    SEQUENTIAL_ADAPTIVE_PER_STEP,
    SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST,
    SequentialAdaptivePerStepConfig,
    build_sequential_adaptive_per_step,
)
_BUILTIN_METHODS.register(
    SPLIT_PER_STEP,
    SPLIT_PER_STEP_MANIFEST,
    SplitPerStepConfig,
    build_split_per_step,
)
_BUILTIN_METHODS.register(
    SPLIT_WINDOW_SUM,
    SPLIT_WINDOW_SUM_MANIFEST,
    SplitWindowSumConfig,
    build_split_window_sum,
)
_BUILTIN_METHODS.register(
    WEIGHTED_PER_STEP,
    WEIGHTED_PER_STEP_MANIFEST,
    WeightedPerStepConfig,
    build_weighted_per_step,
)


def available_methods() -> tuple[str, ...]:
    """Return the immutable built-in conformal method view."""
    return _BUILTIN_METHODS.available_methods


def method_config_schema(method_name: str) -> type[BaseModel]:
    """Return one built-in method's frozen configuration schema."""
    return _BUILTIN_METHODS.config_schema(method_name)


def resolve_method(
    configuration: Mapping[str, object],
    *,
    states: Mapping[str, bytes] | None = None,
) -> ConformalRuntime:
    """Resolve one explicitly selected built-in conformal runtime."""
    return _BUILTIN_METHODS.resolve(configuration, states=states)


__all__ = [
    "METHOD_SCOPE_LABEL",
    "SEQUENTIAL_ADAPTIVE_PER_STEP",
    "SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST",
    "SPLIT_PER_STEP",
    "SPLIT_PER_STEP_MANIFEST",
    "SPLIT_WINDOW_SUM",
    "SPLIT_WINDOW_SUM_MANIFEST",
    "AssumptionClass",
    "CalibrationContext",
    "CalibrationRequirement",
    "CalibrationResult",
    "CensoringPolicy",
    "ClampDeclaration",
    "ClampGuaranteeImpact",
    "ConformalRegistry",
    "ConformalRegistryError",
    "ConformalRuntime",
    "ConservativeRankRequirement",
    "Delivery",
    "EmissionForm",
    "FixedCountRequirement",
    "ForecastKey",
    "GuaranteeDeclaration",
    "IssuedBoundFacts",
    "JointClaim",
    "MethodManifest",
    "MethodManifestError",
    "ObserveAnnotation",
    "ObserveEffect",
    "PostWarmupNonFinite",
    "ResolvedObservation",
    "RuntimeContractError",
    "SequentialAdaptiveConformalRuntime",
    "SequentialAdaptivePerStepConfig",
    "SplitConformalRuntime",
    "SplitPerStepConfig",
    "SplitWindowSumConfig",
    "WEIGHTED_PER_STEP",
    "WEIGHTED_PER_STEP_MANIFEST",
    "WeightedConformalRuntime",
    "WeightedPerStepConfig",
    "available_methods",
    "derive_partition_label",
    "method_config_schema",
    "require_calibration_context",
    "resolve_method",
]
