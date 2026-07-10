"""Define decision-time configuration and inventory-position vocabulary."""

import math
from dataclasses import dataclass, field
from numbers import Integral, Real


class DecisionError(ValueError):
    """Report invalid decision timing or inventory-position data."""


def _integer_period(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DecisionError(f"{name} must be an integer")

    normalized = int(value)
    if normalized < minimum:
        raise DecisionError(f"{name} must be at least {minimum}")
    return normalized


def _finite_nonnegative_quantity(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DecisionError(f"{name} must be a real number")

    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise DecisionError(f"{name} must be finite") from error

    if not math.isfinite(normalized):
        raise DecisionError(f"{name} must be finite")
    if normalized < 0.0:
        raise DecisionError(f"{name} must be non-negative")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True, slots=True)
class DecisionTiming:
    """Represent lead/review timing with its exact protection-period invariant."""

    lead_time: int
    review_period: int
    protection_period: int = field(init=False)

    def __post_init__(self) -> None:
        lead_time = _integer_period(self.lead_time, name="lead_time", minimum=0)
        review_period = _integer_period(
            self.review_period,
            name="review_period",
            minimum=1,
        )
        protection_period = lead_time + review_period

        object.__setattr__(self, "lead_time", lead_time)
        object.__setattr__(self, "review_period", review_period)
        object.__setattr__(self, "protection_period", protection_period)

    @property
    def protection_window(self) -> range:
        """The inclusive horizon steps ``1 .. protection_period``."""
        return range(1, self.protection_period + 1)


@dataclass(frozen=True, slots=True)
class InventoryPosition:
    """Represent the non-negative components read by a policy at decision time."""

    on_hand: float
    on_order: float
    backorders: float

    def __post_init__(self) -> None:
        for name in ("on_hand", "on_order", "backorders"):
            value = _finite_nonnegative_quantity(getattr(self, name), name=name)
            object.__setattr__(self, name, value)

    @property
    def value(self) -> float:
        """Return on-hand plus on-order quantity minus backorders."""
        terms = (self.on_hand, self.on_order, -self.backorders)
        try:
            value = math.fsum(terms)
        except OverflowError:
            scale = max(self.on_hand, self.on_order, self.backorders)
            value = math.fsum(term / scale for term in terms) * scale
        if not math.isfinite(value):
            raise DecisionError("inventory position exceeds finite float range")
        return value
