"""Exercise decision-time vocabulary contracts at tier 1."""

import sys
from dataclasses import FrozenInstanceError

import pytest

from newcalibre.domain import DecisionError, DecisionTiming, InventoryPosition


def test_decision_timing_derives_the_inclusive_protection_window() -> None:
    timing = DecisionTiming(lead_time=2, review_period=3)

    assert timing.protection_period == 5
    assert tuple(timing.protection_window) == (1, 2, 3, 4, 5)


def test_zero_lead_time_still_has_a_positive_protection_period() -> None:
    timing = DecisionTiming(lead_time=0, review_period=1)

    assert timing.protection_period == 1
    assert tuple(timing.protection_window) == (1,)


@pytest.mark.parametrize(
    ("lead_time", "review_period"),
    [(-1, 1), (0, 0), (0, -1), (True, 1), (0, False), (1.0, 1)],
)
def test_decision_timing_rejects_invalid_periods(
    lead_time: object,
    review_period: object,
) -> None:
    with pytest.raises(DecisionError):
        DecisionTiming(lead_time=lead_time, review_period=review_period)  # type: ignore[arg-type]


def test_decision_timing_is_immutable() -> None:
    timing = DecisionTiming(lead_time=1, review_period=2)

    with pytest.raises(FrozenInstanceError):
        timing.lead_time = 3  # type: ignore[misc]


def test_inventory_position_is_on_hand_plus_on_order_minus_backorders() -> None:
    position = InventoryPosition(on_hand=7.5, on_order=4.0, backorders=3.0)

    assert position.value == 8.5


def test_inventory_position_can_be_negative() -> None:
    position = InventoryPosition(on_hand=0.0, on_order=1.0, backorders=2.5)

    assert position.value == -1.5


def test_inventory_position_cancels_before_adding_large_components() -> None:
    maximum = sys.float_info.max

    assert InventoryPosition(maximum, maximum, maximum).value == maximum


def test_inventory_position_rejects_a_genuinely_overflowing_result() -> None:
    maximum = sys.float_info.max

    with pytest.raises(DecisionError, match="finite float range"):
        _ = InventoryPosition(maximum, maximum, 0.0).value


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_inventory_position_rejects_invalid_components(value: float) -> None:
    with pytest.raises(DecisionError):
        InventoryPosition(on_hand=value, on_order=0.0, backorders=0.0)


def test_inventory_position_is_immutable() -> None:
    position = InventoryPosition(on_hand=1.0, on_order=2.0, backorders=3.0)

    with pytest.raises(FrozenInstanceError):
        position.on_order = 4.0  # type: ignore[misc]
