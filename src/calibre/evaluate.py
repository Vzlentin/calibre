"""Metrics for intervals, quantiles, and order quantities over arrays.

Each metric averages over the cells whose target is known. A NaN target is not known
and does not count. `axis` keeps the other axes, for example `axis=(0, 2)` gives one
value per node of `[O, N, C]` arrays. An infinite bound, which means not ready, gives
an infinite width or score: slice the warm-up away instead of hiding it.
"""

import numpy as np

Axis = int | tuple[int, ...] | None


def coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, axis: Axis = None):
    """Share of known targets with lower <= target <= upper."""
    inside = (lower <= target) & (target <= upper)
    return _mean(inside.astype(np.float64), target, axis)


def width(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, axis: Axis = None):
    """Mean `upper - lower` over the cells with a known target."""
    return _mean(upper - lower, target, axis)


def interval_score(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float, axis: Axis = None
):
    """Winkler interval score at `level`: width plus 2 / (1 - level) times the miss."""
    alpha = 1 - level
    below = np.maximum(lower - target, 0)
    above = np.maximum(target - upper, 0)
    return _mean(upper - lower + (2 / alpha) * (below + above), target, axis)


def pinball(target: np.ndarray, quantile: np.ndarray, tau: float, axis: Axis = None):
    """Quantile loss of `quantile` as the `tau` quantile of the target."""
    error = target - quantile
    return _mean(np.maximum(tau * error, (tau - 1) * error), target, axis)


def newsvendor_cost(
    target: np.ndarray, order: np.ndarray, holding: float, shortage: float, axis: Axis = None
):
    """Mean cost `holding * (order - target)+ + shortage * (target - order)+`.

    It is the pinball loss at tau = shortage / (holding + shortage), times
    holding + shortage.
    """
    over = np.maximum(order - target, 0)
    under = np.maximum(target - order, 0)
    return _mean(holding * over + shortage * under, target, axis)


def _mean(values: np.ndarray, target: np.ndarray, axis: Axis):
    known = ~np.isnan(target)
    values = np.broadcast_to(values, known.shape)
    total = np.where(known, values, 0).sum(axis=axis)
    count = known.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count
