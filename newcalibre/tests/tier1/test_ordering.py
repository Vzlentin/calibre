"""Exercise pre-execution ordering validation and pure arithmetic."""

from __future__ import annotations

import math
import sys
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from newcalibre.domain import (
    CostStructure,
    DecisionScope,
    DecisionScopeKind,
    DecisionTiming,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    InventoryPosition,
    ScoredSeries,
)
from newcalibre.ordering import (
    OrderingConfigError,
    OrderingConfiguration,
    OrderingInputError,
    OrderingSetup,
    compile_ordering,
    order_up_to,
)

pytestmark = pytest.mark.tier1

TIMING = DecisionTiming(lead_time=2, review_period=3)
COST = CostStructure(underage=3.0, overage=1.0, holding=0.5, shortage=2.0)


def _setup(**changes: object) -> OrderingSetup:
    if changes.get("policy") == "rss" and not {
        "reorder_point",
        "reorder_point_scale",
    }.intersection(changes):
        changes["reorder_point"] = 1.0
    setup = OrderingSetup(
        policy="newsvendor",
        series_keys=("sku-b", "sku-a"),
        cost_structure=COST,
        decision_timing=TIMING,
        task_horizon=5,
        calibration_coverage=0.9,
    )
    return replace(setup, **changes)


def _descriptor() -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        level=0.9,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.WINDOW_SUM,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )


def test_compile_snapshots_canonical_series_and_global_costs() -> None:
    series_keys = ["sku-b", "sku-a"]
    setup = _setup(series_keys=series_keys)

    configuration = compile_ordering(setup)
    series_keys.append("sku-c")

    assert configuration.policy == "newsvendor"
    assert configuration.series_keys == ("sku-a", "sku-b")
    assert dict(configuration.costs_by_series) == {"sku-a": COST, "sku-b": COST}
    assert configuration.protection_period == 5
    assert configuration.decision_fractile == 0.75
    assert configuration.coverage == 0.9

    with pytest.raises(TypeError):
        cast(Any, configuration.costs_by_series)["sku-a"] = COST


def test_compile_snapshots_per_series_cost_mapping() -> None:
    costs = {"sku-a": COST, "sku-b": COST}
    configuration = compile_ordering(_setup(cost_structure=costs))
    costs["sku-a"] = CostStructure(1.0, 1.0, 0.0, 0.0)

    assert tuple(configuration.costs_by_series) == configuration.series_keys
    assert configuration.costs_by_series["sku-a"] == COST


@pytest.mark.parametrize(
    "costs",
    [
        {"sku-a": COST},
        {"sku-a": COST, "sku-b": COST, "sku-c": COST},
        {"sku-a": COST, "sku-b": object()},
    ],
)
def test_per_series_cost_mapping_must_match_the_series_set_exactly(costs: object) -> None:
    with pytest.raises(OrderingConfigError, match="cost_structure"):
        compile_ordering(_setup(cost_structure=cast(Any, costs)))


@pytest.mark.parametrize("policy", ["newsvendor", "rs", "rss"])
def test_policy_names_are_closed(policy: str) -> None:
    assert compile_ordering(_setup(policy=policy)).policy == policy


@pytest.mark.parametrize("policy", ["order-up-to", "base-stock"])
def test_unknown_policy_is_rejected(policy: str) -> None:
    with pytest.raises(OrderingConfigError, match="policy must be one of"):
        compile_ordering(_setup(policy=policy))


def test_newsvendor_requires_one_strict_shared_cost_fractile() -> None:
    different_ratio = CostStructure(1.0, 1.0, 0.0, 0.0)
    with pytest.raises(OrderingConfigError, match="homogeneous critical ratios"):
        compile_ordering(_setup(cost_structure={"sku-a": COST, "sku-b": different_ratio}))

    for boundary_cost in (
        CostStructure(0.0, 1.0, 0.0, 0.0),
        CostStructure(1.0, 0.0, 0.0, 0.0),
        CostStructure(0.0, 0.0, 0.0, 0.0),
    ):
        with pytest.raises(OrderingConfigError, match="critical ratio"):
            compile_ordering(_setup(cost_structure=boundary_cost))


def test_explicit_decision_fractile_records_binding_and_voids_only_the_claim() -> None:
    heterogeneous_costs = {
        "sku-a": CostStructure(0.0, 0.0, 0.0, 0.0),
        "sku-b": CostStructure(1.0, 3.0, 0.0, 0.0),
    }
    configuration = compile_ordering(
        _setup(cost_structure=heterogeneous_costs, explicit_decision_fractile=0.6)
    )
    descriptor = _descriptor()

    rewritten = configuration.descriptor_for_decision(descriptor)

    assert configuration.decision_fractile == 0.6
    assert [(binding.name, binding.value) for binding in configuration.applied_bindings] == [
        ("explicit_decision_fractile", 0.6)
    ]
    assert rewritten.type == GuaranteeType(
        claim=GuaranteeClaim.NONE,
        currency=None,
        declared_slack=None,
    )
    assert replace(rewritten, type=descriptor.type) == descriptor


def test_override_off_path_has_no_binding_and_returns_descriptor_unchanged() -> None:
    configuration = compile_ordering(_setup())
    descriptor = _descriptor()

    assert configuration.applied_bindings == ()
    assert configuration.descriptor_for_decision(descriptor) is descriptor


@pytest.mark.parametrize(
    "changes",
    [
        {"calibration_coverage": 0.0},
        {"calibration_coverage": 1.0},
        {"policy_coverage": -0.1},
        {"explicit_quantile": math.nan, "policy": "rs"},
        {"explicit_decision_fractile": math.inf},
        {"explicit_decision_fractile": True},
    ],
)
def test_all_configured_levels_must_be_strict_finite_probabilities(
    changes: dict[str, object],
) -> None:
    with pytest.raises(OrderingConfigError):
        compile_ordering(_setup(**changes))


def test_policy_coverage_inherits_or_must_match_calibration_coverage() -> None:
    inherited = compile_ordering(_setup(policy_coverage=None))
    explicit = compile_ordering(_setup(policy_coverage=0.9))

    assert inherited.coverage == explicit.coverage == 0.9
    with pytest.raises(OrderingConfigError, match="must match"):
        compile_ordering(_setup(policy_coverage=0.8))


def test_rs_explicit_quantile_is_the_only_coverage_sync_exemption() -> None:
    configuration = compile_ordering(
        _setup(
            policy="rs",
            explicit_quantile=0.7,
            calibration_coverage=0.9,
            policy_coverage=0.8,
        )
    )

    assert configuration.explicit_quantile == 0.7
    assert configuration.coverage is None

    for policy in ("newsvendor", "rss"):
        with pytest.raises(OrderingConfigError, match="only.*rs"):
            compile_ordering(_setup(policy=policy, explicit_quantile=0.7))


def test_explicit_decision_fractile_is_only_for_newsvendor() -> None:
    for policy in ("rs", "rss"):
        with pytest.raises(OrderingConfigError, match="only.*newsvendor"):
            compile_ordering(_setup(policy=policy, explicit_decision_fractile=0.6))


@pytest.mark.parametrize("policy", ["newsvendor", "rs", "rss"])
def test_every_ordering_run_requires_the_complete_protection_horizon(policy: str) -> None:
    with pytest.raises(OrderingConfigError, match="complete.*window"):
        compile_ordering(_setup(policy=policy, task_horizon=4))


def test_calibration_protection_period_must_equal_lead_plus_review() -> None:
    configuration = compile_ordering(_setup(calibration_protection_period=TIMING.protection_period))
    assert configuration.protection_period == TIMING.lead_time + TIMING.review_period

    with pytest.raises(OrderingConfigError, match="equal lead_time plus review_period"):
        compile_ordering(_setup(calibration_protection_period=TIMING.protection_period + 1))

    with pytest.raises(OrderingConfigError, match="positive integer"):
        compile_ordering(_setup(calibration_protection_period=True))


def test_window_policies_require_a_bound_source() -> None:
    for policy in ("rs", "rss"):
        with pytest.raises(OrderingConfigError, match="requires conformal coverage"):
            compile_ordering(_setup(policy=policy, calibration_coverage=None))

    configuration = compile_ordering(
        _setup(policy="rs", calibration_coverage=None, explicit_quantile=0.7)
    )
    assert configuration.explicit_quantile == 0.7


def test_setup_and_compiled_configuration_are_immutable() -> None:
    setup = _setup()
    configuration = compile_ordering(setup)

    with pytest.raises(FrozenInstanceError):
        cast(Any, setup).policy = "rs"
    with pytest.raises(FrozenInstanceError):
        cast(Any, configuration).policy = "rs"
    with pytest.raises(TypeError, match="compile_ordering"):
        OrderingConfiguration()


@pytest.mark.parametrize(
    "target",
    [True, "10", None, math.nan, math.inf, -math.inf],
)
def test_order_up_to_rejects_malformed_or_nonfinite_targets(target: object) -> None:
    with pytest.raises(OrderingInputError, match="target"):
        order_up_to(cast(Any, target), InventoryPosition(1.0, 0.0, 0.0))


def test_order_up_to_is_real_valued_pure_and_zero_at_or_above_target() -> None:
    position = InventoryPosition(on_hand=2.0, on_order=1.25, backorders=0.5)

    assert order_up_to(10.5, position) == 7.75
    assert order_up_to(2.75, position) == 0.0
    assert order_up_to(2.0, position) == 0.0
    assert position == InventoryPosition(on_hand=2.0, on_order=1.25, backorders=0.5)


def test_order_up_to_preserves_fractional_need_without_ceiling() -> None:
    quantity = order_up_to(4.2, InventoryPosition(1.0, 0.0, 0.0))

    assert quantity == pytest.approx(3.2)
    assert not quantity.is_integer()


def test_order_up_to_rejects_inventory_and_quantity_overflow() -> None:
    maximum = sys.float_info.max

    with pytest.raises(OrderingInputError, match="inventory position"):
        order_up_to(maximum, InventoryPosition(maximum, maximum, 0.0))
    with pytest.raises(OrderingInputError, match="order quantity"):
        order_up_to(maximum, InventoryPosition(0.0, 0.0, maximum))


def test_order_up_to_requires_the_domain_inventory_position() -> None:
    with pytest.raises(OrderingInputError, match="InventoryPosition"):
        order_up_to(1.0, cast(Any, 0.0))
