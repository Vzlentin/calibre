"""Scores: how wrong a forecast was, per step or per window of steps."""

import numpy as np


class AbsoluteResidual:
    """`|actual - point|` per step. The interval is a two-sided band."""

    def cover(self, horizon: int) -> np.ndarray:
        return np.eye(horizon, dtype=bool)

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.abs(target - point)

    def interval(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return point - threshold, point + threshold


class SignedResidual:
    """`actual - point` per step. The interval is a one-sided upper bound."""

    def cover(self, horizon: int) -> np.ndarray:
        return np.eye(horizon, dtype=bool)

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def interval(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(point, -np.inf), point + threshold


class WindowSum:
    """`actual - point` summed over the first `steps` steps: one score per origin.

    The interval is a one-sided upper bound on the total. It is not a sum of per-step
    bounds, and it is known only when every step of the window is known.
    """

    def __init__(self, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        self.steps = steps

    def cover(self, horizon: int) -> np.ndarray:
        if self.steps > horizon:
            raise ValueError(f"window of {self.steps} steps is longer than horizon {horizon}")
        return (np.arange(horizon) < self.steps)[None, :]

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def interval(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(point, -np.inf), point + threshold
