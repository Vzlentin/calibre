"""Exercise decision-cost configuration contracts at tier 1."""

import sys
from dataclasses import FrozenInstanceError

import pytest

from newcalibre.domain import CostStructure, CostStructureError


def test_cost_structure_preserves_four_independent_zero_components() -> None:
    costs = CostStructure(
        underage=0.0,
        overage=0.0,
        holding=0.0,
        shortage=0.0,
    )

    assert costs.underage == 0.0
    assert costs.overage == 0.0
    assert costs.holding == 0.0
    assert costs.shortage == 0.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_cost_structure_rejects_non_finite_or_negative_components(value: float) -> None:
    with pytest.raises(CostStructureError):
        CostStructure(
            underage=value,
            overage=1.0,
            holding=1.0,
            shortage=1.0,
        )


def test_cost_structure_is_immutable() -> None:
    costs = CostStructure(underage=1.0, overage=2.0, holding=3.0, shortage=4.0)

    with pytest.raises(FrozenInstanceError):
        costs.underage = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("underage", "overage", "expected"),
    [(0.0, 2.0, 0.0), (2.0, 0.0, 1.0), (1.0, 3.0, 0.25)],
)
def test_critical_ratio_includes_valid_boundaries(
    underage: float,
    overage: float,
    expected: float,
) -> None:
    costs = CostStructure(
        underage=underage,
        overage=overage,
        holding=17.0,
        shortage=19.0,
    )

    assert costs.critical_ratio == expected


def test_critical_ratio_refuses_only_a_non_positive_denominator_at_use() -> None:
    costs = CostStructure(underage=0.0, overage=0.0, holding=1.0, shortage=1.0)

    with pytest.raises(CostStructureError, match="positive"):
        _ = costs.critical_ratio


def test_critical_ratio_does_not_overflow_for_large_finite_costs() -> None:
    costs = CostStructure(
        underage=sys.float_info.max,
        overage=sys.float_info.max,
        holding=0.0,
        shortage=0.0,
    )

    assert costs.critical_ratio == 0.5
