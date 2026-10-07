"""Losses: the cost of issued bounds once their target is known.

A loss is a function of the bounds that a threshold issues, so a risk calibrator can
evaluate it at every candidate threshold. It need not be monotone in the threshold.
The threshold is one dimension that indexes a `Score`: a loss of a parameter that is
not a bound is out of scope.
"""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @property
    @abstractmethod
    def maximum(self) -> float:
        """The largest value the loss can take, inf when it has no bound."""

    @abstractmethod
    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        """Loss per cell, broadcast from bounds, known targets, and censored flags."""


class Miss(Loss):
    """1 when the target is outside the bounds. Its risk is the miscoverage."""

    @property
    def maximum(self) -> float:
        return 1.0

    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        return ((target < lower) | (target > upper)).astype(np.float64)


class Newsvendor(Loss):
    """`holding * (upper - target)+ + shortage * (target - upper)+`: the upper bound is
    an order and the target is the demand it serves.

    Costs are positive scalars or broadcast against `[K, N, G]`, so `[N, 1]` is one per
    node. The loss has no bound. A censored target is scored as observed.
    """

    def __init__(self, holding: float | np.ndarray, shortage: float | np.ndarray) -> None:
        holding, shortage = np.asarray(holding, float), np.asarray(shortage, float)
        if (holding <= 0).any() or (shortage <= 0).any():
            raise ValueError("holding and shortage costs must be positive")
        self.holding = holding
        self.shortage = shortage

    @property
    def maximum(self) -> float:
        return np.inf

    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        over = np.maximum(upper - target, 0)
        under = np.maximum(target - upper, 0)
        return self.holding * over + self.shortage * under
