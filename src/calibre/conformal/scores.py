"""Nonconformity scores and the bounds indexed by their calibration parameters."""

from abc import ABC, abstractmethod

import numpy as np


class Score(ABC):
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Trailing forecast axes: () for points, (2,) for native quantiles."""

    @abstractmethod
    def parameter_shape(self) -> tuple[int, ...]:
        """Trailing parameter axes: () for a scalar, (2,) for paired corrections."""

    @abstractmethod
    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Nonconformity with the target shape from forecasts with input_shape()."""

    @abstractmethod
    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds from the forecast and issued parameters."""


class Absolute(Score):
    """Absolute residual for a two-sided band."""

    def input_shape(self) -> tuple[int, ...]:
        return ()

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.abs(target - point)

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return point - threshold, point + threshold


class Signed(Score):
    """Signed residual for an upper bound."""

    def input_shape(self) -> tuple[int, ...]:
        return ()

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        upper = point + threshold
        return np.full_like(upper, -np.inf), upper


class Asymmetric(Score):
    """Signed residuals with independently calibrated lower/upper corrections."""

    def input_shape(self) -> tuple[int, ...]:
        return ()

    def parameter_shape(self) -> tuple[int, ...]:
        return (2,)

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return target - point

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if threshold.shape != point.shape + (2,):
            raise ValueError("asymmetric parameters must have shape point.shape + (2,)")
        return point + threshold[..., 0], point + threshold[..., 1]


class CQR(Score):
    """Conformalized quantile regression from ordered native quantiles."""

    def input_shape(self) -> tuple[int, ...]:
        return (2,)

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.maximum(point[..., 0] - target, target - point[..., 1])

    def bound(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return point[..., 0] - threshold, point[..., 1] + threshold
