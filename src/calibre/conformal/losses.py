"""Losses: the cost of issued bounds once their target is known.

A risk calibrator averages a loss over known targets for each candidate threshold, so a
loss needs no monotonicity in the threshold.
"""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        """Loss per cell, broadcast from bounds, targets, and censored flags.

        A NaN target gives a NaN loss.
        """


class Miss(Loss):
    """1 when the target is outside the bounds. Its risk is the miscoverage."""

    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        miss = ((target < lower) | (target > upper)).astype(np.float64)
        return np.where(np.isnan(target), np.nan, miss)


class Newsvendor(Loss):
    """`holding * (upper - target)+ + shortage * (target - upper)+`: the upper bound is
    an order and the target is the demand it serves.

    Costs are scalars or broadcast against `[K, N, G]`, so `[N, 1]` is one per node. A
    censored target is scored as observed.
    """

    def __init__(self, holding: float | np.ndarray, shortage: float | np.ndarray) -> None:
        holding, shortage = np.asarray(holding, float), np.asarray(shortage, float)
        if (holding < 0).any() or (shortage < 0).any():
            raise ValueError("holding and shortage costs must not be negative")
        self.holding = holding
        self.shortage = shortage

    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        over = np.maximum(upper - target, 0)
        under = np.maximum(target - upper, 0)
        return self.holding * over + self.shortage * under
