"""Losses: the cost of issued bounds once their target is known.

A loss is a function of the bounds that a threshold issues, so a risk calibrator can
evaluate it at every candidate threshold. It need not be monotone in the threshold.
"""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @property
    @abstractmethod
    def maximum(self) -> float:
        """The largest value the loss can take. It must be finite."""

    @abstractmethod
    def loss(self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Loss per cell, broadcast from bounds and known targets."""


class Miss(Loss):
    """1 when the target is outside the bounds. Its risk is the miscoverage."""

    @property
    def maximum(self) -> float:
        return 1.0

    def loss(self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> np.ndarray:
        return ((target < lower) | (target > upper)).astype(np.float64)
