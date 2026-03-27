from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class IntervalPrediction:
    center: float
    lower: float
    upper: float
    radius: float
    alpha: float
    issued_at: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def contains(self, value: float) -> bool:
        return bool(self.lower <= value <= self.upper)


@dataclass(slots=True)
class MultiStepIntervalPrediction:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    radius: np.ndarray
    alpha: np.ndarray
    issued_at: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.radius = np.asarray(self.radius, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)

    @property
    def horizon(self) -> int:
        return int(self.center.shape[0])

    def contains(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.logical_and(self.lower <= arr, arr <= self.upper)
