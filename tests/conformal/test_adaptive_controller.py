"""Tests for the adaptive conformal interval controller."""

from __future__ import annotations

from calibre.conformal.controllers import AdaptiveAlphaController
from calibre.conformal.types import IntervalPrediction


def _interval(
    lower: float, upper: float, center: float = 0.0, alpha: float = 0.1
) -> IntervalPrediction:
    return IntervalPrediction(
        center=center,
        lower=lower,
        upper=upper,
        radius=max(upper - center, center - lower),
        alpha=alpha,
    )


def test_error_history_starts_empty() -> None:
    controller = AdaptiveAlphaController(alpha=0.1, gamma=0.05)
    assert controller.error_history == []


def test_error_history_records_misses_and_hits() -> None:
    controller = AdaptiveAlphaController(alpha=0.1, gamma=0.05)
    controller.observe(5.0, _interval(0.0, 10.0), h=1)
    controller.observe(20.0, _interval(0.0, 10.0), h=1)
    controller.observe(7.0, _interval(0.0, 10.0), h=1)
    assert controller.error_history == [0, 1, 0]


def test_error_history_is_independent_copy() -> None:
    controller = AdaptiveAlphaController(alpha=0.1, gamma=0.05)
    controller.observe(5.0, _interval(0.0, 10.0), h=1)
    snapshot = controller.error_history
    snapshot.append(999)
    assert controller.error_history == [0]
