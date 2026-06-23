"""Value type for single-step interval predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IntervalPrediction:
    """A single-step interval with its center, bounds, radius, and ``alpha``."""

    center: float
    lower: float
    upper: float
    radius: float
    alpha: float
    issued_at: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def contains(self, value: float) -> bool:
        return bool(self.lower <= value <= self.upper)
