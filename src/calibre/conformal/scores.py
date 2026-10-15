"""Residual scores and the bounds issued from their calibration parameters.

Scalar parameters index nested bounds: a score is the smallest threshold that holds
the target. Asymmetric intervals instead take a pair of signed residual quantiles;
coverage must be evaluated against both issued endpoints, not by score <= threshold.
"""

from abc import ABC, abstractmethod

import numpy as np


class Score(ABC):
    @abstractmethod
    def parameter_shape(self) -> tuple[int, ...]:
        """Trailing parameter axes: () for a scalar, (2,) for lower/upper corrections."""

    @abstractmethod
    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Nonconformity from targets and points of the same shape."""

    @abstractmethod
    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds with the point shape, from the issued parameters."""


class Absolute(Score):
    """`|target - point|`. The bound is a two-sided band."""

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.abs(target - point)

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return point - threshold, point + threshold


class Signed(Score):
    """`target - point`. The bound is a one-sided upper bound.

    On a `LeadTime` target, it bounds the total. It is not a sum of per-step bounds.
    """

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        upper = point + threshold
        return np.full_like(upper, -np.inf), upper


class Asymmetric(Score):
    """Signed residuals with independently calibrated lower/upper corrections.

    Parameters have shape point.shape + (2,). They are not a scalar nonconformity
    threshold and must not be used with scalar quantile or risk calibrators.
    """

    def parameter_shape(self) -> tuple[int, ...]:
        return (2,)

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if threshold.shape != point.shape + (2,):
            raise ValueError("asymmetric parameters must have shape point.shape + (2,)")
        return point + threshold[..., 0], point + threshold[..., 1]
