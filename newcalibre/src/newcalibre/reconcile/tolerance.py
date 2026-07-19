"""Derive the sole coherence bound from floating-point problem facts."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


class CoherenceToleranceError(ValueError):
    """Report invalid inputs to coherence-bound derivation."""


def coherence_tolerance(
    *,
    reduction_width: int,
    vector_magnitude: float,
    solver_tolerance: float | None = None,
) -> float:
    """Derive an absolute coherence bound for one lattice evaluation.

    The roundoff component uses the standard floating-point reduction bound
    for the widest lattice row and scales it by the largest vector magnitude.
    A declared relative solver tolerance contributes on the same magnitude
    scale when projection strategies are added.
    """
    if (
        not isinstance(reduction_width, Integral)
        or isinstance(reduction_width, bool)
        or reduction_width < 0
    ):
        raise CoherenceToleranceError("reduction width must be a non-negative integer")
    if (
        not isinstance(vector_magnitude, Real)
        or isinstance(vector_magnitude, bool)
        or not math.isfinite(float(vector_magnitude))
        or vector_magnitude < 0
    ):
        raise CoherenceToleranceError("vector magnitude must be a finite non-negative real")
    if solver_tolerance is not None and (
        not isinstance(solver_tolerance, Real)
        or isinstance(solver_tolerance, bool)
        or not math.isfinite(float(solver_tolerance))
        or solver_tolerance < 0
    ):
        raise CoherenceToleranceError("solver tolerance must be a finite non-negative real")

    width = max(int(reduction_width), 1)
    magnitude = max(float(vector_magnitude), 1.0)
    epsilon = np.finfo(np.float64).eps
    denominator = 1.0 - width * epsilon
    if denominator <= 0.0:
        raise CoherenceToleranceError("reduction width exceeds the float64 error model")
    reduction_error = (width * epsilon / denominator) * width * magnitude
    solver_error = 0.0 if solver_tolerance is None else float(solver_tolerance) * magnitude
    return float(reduction_error + solver_error)
