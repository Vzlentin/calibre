"""Compile ordering facts and provide the common order-up-to kernel."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from types import MappingProxyType

from newcalibre.domain import (
    CostStructure,
    CostStructureError,
    DecisionError,
    DecisionTiming,
    GuaranteeClaim,
    GuaranteeDescriptor,
    GuaranteeType,
    InventoryPosition,
)

_POLICY_NAMES = frozenset({"newsvendor", "rs", "rss"})
_WINDOW_POLICIES = frozenset({"rs", "rss"})
_EXPLICIT_FRACTILE_BINDING = "explicit_decision_fractile"


class OrderingConfigError(ValueError):
    """Report ordering configuration that cannot execute unambiguously."""


class OrderingInputError(ValueError):
    """Report malformed input to pure ordering arithmetic."""


@dataclass(frozen=True, slots=True)
class AppliedBinding:
    """Record one named configuration value that modified a decision."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise OrderingConfigError("binding name must be a non-empty string")
        try:
            self.name.encode("utf-8")
        except UnicodeError as error:
            raise OrderingConfigError("binding name must be valid UTF-8") from error
        value = _finite_real(self.value, name="binding value")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class OrderingSetup:
    """Carry caller-supplied ordering facts until they are compiled."""

    policy: str
    series_keys: Iterable[str]
    cost_structure: CostStructure | Mapping[str, CostStructure]
    decision_timing: DecisionTiming
    task_horizon: int
    calibration_coverage: float | None = None
    calibration_protection_period: int | None = None
    policy_coverage: float | None = None
    explicit_quantile: float | None = None
    explicit_decision_fractile: float | None = None


@dataclass(frozen=True, slots=True, init=False)
class OrderingConfiguration:
    """Expose one validated, immutable ordering configuration snapshot."""

    policy: str
    series_keys: tuple[str, ...]
    costs_by_series: Mapping[str, CostStructure] = field(repr=False)
    decision_timing: DecisionTiming
    coverage: float | None
    explicit_quantile: float | None
    decision_fractile: float | None
    applied_bindings: tuple[AppliedBinding, ...]

    def __init__(self) -> None:
        raise TypeError("OrderingConfiguration must be created with compile_ordering()")

    @property
    def protection_period(self) -> int:
        """Return the exact lead-time plus review-period protection period."""
        return self.decision_timing.protection_period

    def descriptor_for_decision(
        self,
        descriptor: GuaranteeDescriptor,
    ) -> GuaranteeDescriptor:
        """Return the descriptor after applying claim-voiding configuration."""
        if not isinstance(descriptor, GuaranteeDescriptor):
            raise OrderingInputError("descriptor must be a GuaranteeDescriptor")
        if not self.applied_bindings:
            return descriptor
        return replace(
            descriptor,
            type=GuaranteeType(
                claim=GuaranteeClaim.NONE,
                currency=None,
                declared_slack=None,
            ),
        )


def compile_ordering(setup: OrderingSetup) -> OrderingConfiguration:
    """Validate and snapshot all ordering facts before execution."""
    if not isinstance(setup, OrderingSetup):
        raise OrderingConfigError("setup must be an OrderingSetup")

    policy = _policy_name(setup.policy)
    series_keys = _series_keys(setup.series_keys)
    costs_by_series = _costs_by_series(setup.cost_structure, series_keys=series_keys)
    timing = setup.decision_timing
    if not isinstance(timing, DecisionTiming):
        raise OrderingConfigError("decision_timing must be a DecisionTiming")
    horizon = _positive_integer(setup.task_horizon, name="task_horizon")
    if setup.calibration_protection_period is not None:
        calibration_protection_period = _positive_integer(
            setup.calibration_protection_period,
            name="calibration_protection_period",
        )
        if calibration_protection_period != timing.protection_period:
            raise OrderingConfigError(
                "calibration_protection_period must equal lead_time plus review_period"
            )

    calibration_coverage = _optional_probability(
        setup.calibration_coverage,
        name="calibration_coverage",
    )
    policy_coverage = _optional_probability(
        setup.policy_coverage,
        name="policy_coverage",
    )
    explicit_quantile = _optional_probability(
        setup.explicit_quantile,
        name="explicit_quantile",
    )
    explicit_fractile = _optional_probability(
        setup.explicit_decision_fractile,
        name="explicit_decision_fractile",
    )

    if explicit_quantile is not None and policy != "rs":
        raise OrderingConfigError("explicit_quantile is valid only for the rs policy")
    if explicit_fractile is not None and policy != "newsvendor":
        raise OrderingConfigError(
            "explicit_decision_fractile is valid only for a cost-driven newsvendor policy"
        )

    coverage = _consumed_coverage(
        calibration_coverage=calibration_coverage,
        policy_coverage=policy_coverage,
        explicit_quantile=explicit_quantile,
    )
    if horizon < timing.protection_period:
        raise OrderingConfigError(
            "task_horizon must cover the complete lead-time plus review-period window"
        )
    if policy in _WINDOW_POLICIES and calibration_coverage is None and explicit_quantile is None:
        raise OrderingConfigError(
            "a window policy requires conformal coverage or an rs explicit quantile"
        )

    bindings: tuple[AppliedBinding, ...] = ()
    decision_fractile: float | None = None
    if policy == "newsvendor":
        if explicit_fractile is not None:
            decision_fractile = explicit_fractile
            bindings = (AppliedBinding(name=_EXPLICIT_FRACTILE_BINDING, value=explicit_fractile),)
        else:
            decision_fractile = _shared_critical_ratio(costs_by_series)

    instance = object.__new__(OrderingConfiguration)
    object.__setattr__(instance, "policy", policy)
    object.__setattr__(instance, "series_keys", series_keys)
    object.__setattr__(instance, "costs_by_series", costs_by_series)
    object.__setattr__(instance, "decision_timing", timing)
    object.__setattr__(instance, "coverage", coverage)
    object.__setattr__(instance, "explicit_quantile", explicit_quantile)
    object.__setattr__(instance, "decision_fractile", decision_fractile)
    object.__setattr__(instance, "applied_bindings", bindings)
    return instance


def order_up_to(target: float, inventory_position: InventoryPosition) -> float:
    """Return the finite, real-valued quantity needed to reach a target."""
    target_value = _finite_real(target, name="target", error_type=OrderingInputError)
    if not isinstance(inventory_position, InventoryPosition):
        raise OrderingInputError("inventory_position must be an InventoryPosition")
    try:
        position_value = inventory_position.value
    except DecisionError as error:
        raise OrderingInputError("inventory position exceeds finite float range") from error

    if position_value >= target_value:
        return 0.0
    quantity = target_value - position_value
    if not math.isfinite(quantity):
        raise OrderingInputError("order quantity exceeds finite float range")
    return quantity


def _policy_name(value: object) -> str:
    if not isinstance(value, str) or value not in _POLICY_NAMES:
        names = ", ".join(sorted(_POLICY_NAMES))
        raise OrderingConfigError(f"policy must be one of: {names}")
    return value


def _series_keys(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise OrderingConfigError("series_keys must be an iterable of series keys")
    try:
        snapshot = tuple(values)
    except TypeError as error:
        raise OrderingConfigError("series_keys must be an iterable of series keys") from error
    if not snapshot:
        raise OrderingConfigError("series_keys must not be empty")
    for value in snapshot:
        if not isinstance(value, str) or not value:
            raise OrderingConfigError("series keys must be non-empty strings")
        try:
            value.encode("utf-8")
        except UnicodeError as error:
            raise OrderingConfigError("series keys must be valid UTF-8") from error
    if len(set(snapshot)) != len(snapshot):
        raise OrderingConfigError("series_keys must not contain duplicates")
    return tuple(sorted(snapshot, key=str.encode))


def _costs_by_series(
    value: CostStructure | Mapping[str, CostStructure],
    *,
    series_keys: tuple[str, ...],
) -> Mapping[str, CostStructure]:
    if isinstance(value, CostStructure):
        snapshot = {series_key: value for series_key in series_keys}
    elif isinstance(value, Mapping):
        snapshot = dict(value)
        if set(snapshot) != set(series_keys):
            raise OrderingConfigError(
                "per-series cost_structure keys must exactly match series_keys"
            )
        if any(not isinstance(cost, CostStructure) for cost in snapshot.values()):
            raise OrderingConfigError(
                "every per-series cost_structure value must be a CostStructure"
            )
        snapshot = {series_key: snapshot[series_key] for series_key in series_keys}
    else:
        raise OrderingConfigError(
            "cost_structure must be a CostStructure or an exact per-series mapping"
        )
    return MappingProxyType(snapshot)


def _shared_critical_ratio(costs_by_series: Mapping[str, CostStructure]) -> float:
    ratios: list[float] = []
    try:
        ratios.extend(cost.critical_ratio for cost in costs_by_series.values())
    except CostStructureError as error:
        raise OrderingConfigError("newsvendor requires a defined critical ratio") from error
    if any(not 0.0 < ratio < 1.0 for ratio in ratios):
        raise OrderingConfigError("newsvendor critical ratio must lie strictly inside (0, 1)")
    first = ratios[0]
    if any(ratio != first for ratio in ratios[1:]):
        raise OrderingConfigError(
            "a shared cost-fractile consumer requires homogeneous critical ratios"
        )
    return first


def _consumed_coverage(
    *,
    calibration_coverage: float | None,
    policy_coverage: float | None,
    explicit_quantile: float | None,
) -> float | None:
    if explicit_quantile is not None:
        return None
    if (
        calibration_coverage is not None
        and policy_coverage is not None
        and calibration_coverage != policy_coverage
    ):
        raise OrderingConfigError("policy_coverage must match calibration_coverage")
    return policy_coverage if policy_coverage is not None else calibration_coverage


def _optional_probability(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    normalized = _finite_real(value, name=name)
    if not 0.0 < normalized < 1.0:
        raise OrderingConfigError(f"{name} must lie strictly inside (0, 1)")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OrderingConfigError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise OrderingConfigError(f"{name} must be a positive integer")
    return normalized


def _finite_real(
    value: object,
    *,
    name: str,
    error_type: type[ValueError] = OrderingConfigError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise error_type(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise error_type(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized
