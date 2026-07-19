"""Derive the sole coherence bound from floating-point problem facts."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


class CoherenceToleranceError(ValueError):
    """Report invalid inputs to coherence-bound derivation."""


def covariance_estimator_tolerance(*, n_nodes: int, residual_periods: int) -> float:
    """Derive the roundoff term for all node-pair residual products."""
    for value, name in ((n_nodes, "node count"), (residual_periods, "residual periods")):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise CoherenceToleranceError(f"{name} must be a positive integer")
    operations = int(n_nodes) * int(n_nodes) * int(residual_periods)
    epsilon = np.finfo(np.float64).eps
    denominator = 1.0 - operations * epsilon
    if denominator <= 0.0:
        raise CoherenceToleranceError("covariance estimator exceeds the float64 error model")
    return float(operations * epsilon / denominator)


def coherence_tolerance(
    *,
    reduction_width: int,
    vector_magnitude: float,
    solver_tolerance: float | None = None,
    condition_number: float | None = None,
    estimator_tolerance: float | None = None,
) -> float:
    """Derive an absolute coherence or projection-agreement bound.

    The roundoff component uses the standard floating-point reduction bound
    for the widest lattice row. Projection agreement additionally lifts the
    solver residual through an instance-derived normal-system condition.
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
    if condition_number is not None and (
        not isinstance(condition_number, Real)
        or isinstance(condition_number, bool)
        or not math.isfinite(float(condition_number))
        or condition_number < 1
    ):
        raise CoherenceToleranceError("condition number must be a finite real at least one")
    if estimator_tolerance is not None and (
        not isinstance(estimator_tolerance, Real)
        or isinstance(estimator_tolerance, bool)
        or not math.isfinite(float(estimator_tolerance))
        or estimator_tolerance < 0
    ):
        raise CoherenceToleranceError("estimator tolerance must be a finite non-negative real")

    width = max(int(reduction_width), 1)
    magnitude = max(float(vector_magnitude), 1.0)
    epsilon = np.finfo(np.float64).eps
    denominator = 1.0 - width * epsilon
    if denominator <= 0.0:
        raise CoherenceToleranceError("reduction width exceeds the float64 error model")
    reduction_error = (width * epsilon / denominator) * width * magnitude
    projection_error = 0.0
    if (
        solver_tolerance is not None
        or condition_number is not None
        or estimator_tolerance is not None
    ):
        solver = 0.0 if solver_tolerance is None else float(solver_tolerance)
        estimator = 0.0 if estimator_tolerance is None else float(estimator_tolerance)
        condition = 1.0 if condition_number is None else float(condition_number)
        projection_error = 2.0 * condition * (epsilon + solver + estimator) * width * magnitude
    return float(reduction_error + projection_error)
