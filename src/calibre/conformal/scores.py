"""Scores: nested families of bounds indexed by a threshold, and their inverse.

`bound(point, t)` is the bound that threshold t issues, and `score(target, point)` is the
smallest t whose bound holds the target. A quantile of scores is a threshold, and a
loss of bounds is a function of the threshold.
"""

from abc import ABC, abstractmethod

import numpy as np


class Score(ABC):
    @abstractmethod
    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Nonconformity from targets and points of the same shape."""

    @abstractmethod
    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds, broadcast together. They hold when score <= threshold."""


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
        upper = point + threshold
        return np.full_like(upper, -np.inf), upper
