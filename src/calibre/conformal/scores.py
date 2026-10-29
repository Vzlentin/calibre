"""Nonconformity scores and the bounds indexed by their calibration parameters.

Scalar scores index nested bounds. Asymmetric residual corrections instead take a
pair of parameters. CQR takes a scalar correction and the forecast's native interval.
"""

from abc import ABC, abstractmethod

import numpy as np


class Score(ABC):
    @abstractmethod
    def parameter_shape(self) -> tuple[int, ...]:
        """Trailing parameter axes: () for a scalar, (2,) for paired corrections."""

    @abstractmethod
    def requires_interval(self) -> bool:
        """Whether scoring and bounding require an archived native interval."""

    @abstractmethod
    def score(
        self, target: np.ndarray, point: np.ndarray, *, interval: np.ndarray | None = None
    ) -> np.ndarray:
        """Nonconformity with the target shape. Native intervals add a final pair axis."""

    @abstractmethod
    def bound(
        self, point: np.ndarray, threshold: np.ndarray, *, interval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bounds from points, issued parameters, and optional native intervals."""


class Absolute(Score):
    """`|target - point|`. The bound is a two-sided band."""

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def requires_interval(self) -> bool:
        return False

    def score(
        self, target: np.ndarray, point: np.ndarray, *, interval: np.ndarray | None = None
    ) -> np.ndarray:
        return np.abs(target - point)

    def bound(
        self, point: np.ndarray, threshold: np.ndarray, *, interval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return point - threshold, point + threshold


class Signed(Score):
    """`target - point`. The bound is a one-sided upper bound.

    On a `LeadTime` target, it bounds the total. It is not a sum of per-step bounds.
    """

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def requires_interval(self) -> bool:
        return False

    def score(
        self, target: np.ndarray, point: np.ndarray, *, interval: np.ndarray | None = None
    ) -> np.ndarray:
        return target - point

    def bound(
        self, point: np.ndarray, threshold: np.ndarray, *, interval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        upper = point + threshold
        return np.full_like(upper, -np.inf), upper


class Asymmetric(Score):
    """Signed residuals with independently calibrated lower/upper corrections.

    Parameters have shape point.shape + (2,). Scalar quantile and risk calibrators
    cannot consume this parameterization.
    """

    def parameter_shape(self) -> tuple[int, ...]:
        return (2,)

    def requires_interval(self) -> bool:
        return False

    def score(
        self, target: np.ndarray, point: np.ndarray, *, interval: np.ndarray | None = None
    ) -> np.ndarray:
        return target - point

    def bound(
        self, point: np.ndarray, threshold: np.ndarray, *, interval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if threshold.shape != point.shape + (2,):
            raise ValueError("asymmetric parameters must have shape point.shape + (2,)")
        return point + threshold[..., 0], point + threshold[..., 1]


class CQR(Score):
    """Conformalized quantile regression: `max(lower - target, target - upper)`.

    Use with `SplitQuantile(level)`. Native quantiles are ordered before scoring.
    Negative scores and corrections are retained. Corrected endpoints are never
    reordered: a sufficiently negative correction can intentionally give an empty set.
    """

    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    def requires_interval(self) -> bool:
        return True

    def score(
        self, target: np.ndarray, point: np.ndarray, *, interval: np.ndarray | None = None
    ) -> np.ndarray:
        lower, upper = _endpoints(point, interval)
        return np.maximum(lower - target, target - upper)

    def bound(
        self, point: np.ndarray, threshold: np.ndarray, *, interval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = _endpoints(point, interval)
        return lower - threshold, upper + threshold


def _endpoints(point: np.ndarray, interval: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if interval is None or interval.shape != point.shape + (2,):
        raise ValueError("CQR needs native intervals with shape point.shape + (2,)")
    return np.minimum(interval[..., 0], interval[..., 1]), np.maximum(
        interval[..., 0], interval[..., 1]
    )
