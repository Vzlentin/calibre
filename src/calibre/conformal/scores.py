"""Scores: how wrong a point was on a target column, and the bound a threshold gives."""

from abc import ABC, abstractmethod

import numpy as np


class Score(ABC):
    @abstractmethod
    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Nonconformity from targets and points of the same shape."""

    @abstractmethod
    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds `[N, C]` that hold exactly when score <= threshold."""


class Absolute(Score):
    """`|target - point|`. The bound is a two-sided band."""

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.abs(target - point)

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return point - threshold, point + threshold


class Signed(Score):
    """`target - point`. The bound is a one-sided upper bound.

    On a `LeadTime` target, it bounds the total. It is not a sum of per-step bounds.
    """

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(point, -np.inf), point + threshold
