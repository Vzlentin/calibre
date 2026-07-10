"""Define decision-cost configuration without policy-specific derivations."""

import math
from dataclasses import dataclass
from numbers import Real


class CostStructureError(ValueError):
    """Report invalid decision-cost data or an undefined critical ratio."""


def _finite_nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CostStructureError(f"{name} must be a real number")

    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise CostStructureError(f"{name} must be finite") from error

    if not math.isfinite(normalized):
        raise CostStructureError(f"{name} must be finite")
    if normalized < 0.0:
        raise CostStructureError(f"{name} must be non-negative")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True, slots=True)
class CostStructure:
    """Represent independent per-decision and per-period cost components."""

    underage: float
    overage: float
    holding: float
    shortage: float

    def __post_init__(self) -> None:
        for name in ("underage", "overage", "holding", "shortage"):
            value = _finite_nonnegative_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)

    @property
    def critical_ratio(self) -> float:
        """Return the underage share, rejecting only an undefined ratio."""
        if self.underage == 0.0 and self.overage == 0.0:
            raise CostStructureError("critical ratio requires a positive denominator")

        scale = max(self.underage, self.overage)
        scaled_underage = self.underage / scale
        scaled_overage = self.overage / scale
        return scaled_underage / (scaled_underage + scaled_overage)
