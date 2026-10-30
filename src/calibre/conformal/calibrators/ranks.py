"""Finite-sample score quantiles, with rolling retention and pooling.

These functions hold the rank rules of the library. Calibrators call them on the
scores they keep.
"""

import numpy as np


def score_quantile(scores: np.ndarray, level: float | np.ndarray) -> np.ndarray:
    """Finite-sample quantile of finite scores along axis 0, without interpolation.

    A vector is one pool; a matrix is independent pools in columns. `level` is a
    scalar or one value per output. The rank is ceil((n + 1) * level); insufficient
    samples give inf.
    """
    scores = np.asarray(scores)
    level = np.asarray(level)
    if scores.ndim not in (1, 2) or level.ndim > 1:
        raise ValueError("scores must be a vector or matrix and level a scalar or vector")
    if not np.isfinite(level).all() or ((level <= 0) | (level >= 1)).any():
        raise ValueError("level must be strictly between zero and one")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite; select only resolved observations")
    shape = scores.shape[1:] if scores.ndim == 2 else np.shape(level)
    ranks = np.ceil((len(scores) + 1) * np.broadcast_to(level, shape)).astype(np.int64)
    out = np.full(shape, np.inf, dtype=np.result_type(scores.dtype, np.float32))
    ready = ranks <= len(scores)
    if ready.any():
        ordered = np.sort(scores, axis=0)
        if scores.ndim == 1:
            out[ready] = ordered[ranks[ready] - 1]
        else:
            columns = np.nonzero(ready)[0]
            out[ready] = ordered[ranks[ready] - 1, columns]
    return out


def retained_quantile(
    scores: np.ndarray,
    level: float | np.ndarray,
    window: int | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Quantile per node `[N]` from scores `[K, N]` in origin order. NaN is no score.

    `window` keeps the last K resolved rows per pool, before pooling. A row is resolved
    for a pool when it has at least one finite score there, so holes do not consume
    rows. `groups` has one integer per node: equal labels pool their scores, and every
    member gets the pool quantile. `level` is a scalar or one value per node.
    """
    if window is not None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        # Fast path: when the last K rows are complete, they are the retained rows.
        if np.isfinite(scores[-window:]).all():
            scores = scores[-window:]
    if groups is None:
        if window is not None and len(scores) > window:
            finite = np.isfinite(scores)
            retained = np.cumsum(finite[::-1], axis=0)[::-1] <= window
            scores = np.where(finite & retained, scores, np.nan)
            scores = scores[np.isfinite(scores).any(axis=1)]
        return _finite_quantile(scores, level)
    levels = np.broadcast_to(level, (scores.shape[1],))
    out = np.full(scores.shape[1], np.inf, dtype=np.float32)
    order = np.argsort(groups)
    labels = groups[order]
    for columns in np.split(order, np.flatnonzero(labels[1:] != labels[:-1]) + 1):
        pool = scores[:, columns]
        pool = pool[np.isfinite(pool).any(axis=1)]
        if window is not None:
            pool = pool[-window:]
        out[columns] = score_quantile(pool[np.isfinite(pool)], levels[columns])
    return out


def weighted_score_quantile(
    scores: np.ndarray, weights: np.ndarray, level: float | np.ndarray
) -> np.ndarray:
    """Weighted inverse CDF per column [N], with a unit test mass at +inf.

    Scores are [K, N], finite or NaN. Weights are [K] or [K, N], fixed in [0, 1];
    missing scores carry no mass. Select the first score with cumulative weight
    >= level * (1 + total weight), without interpolation. Otherwise return +inf.
    Accumulate weights and levels in float64; do not renormalize the finite scores.
    """
    scores = np.asarray(scores)
    weights = np.asarray(weights, dtype=np.float64)
    levels = np.asarray(level, dtype=np.float64)
    if scores.ndim != 2 or levels.ndim > 1:
        raise ValueError("scores must be [K, N] and level a scalar or vector")
    if not np.isfinite(levels).all() or ((levels <= 0) | (levels >= 1)).any():
        raise ValueError("level must be strictly between zero and one")
    if np.isinf(scores).any():
        raise ValueError("scores must be finite or NaN")
    if weights.shape not in ((len(scores),), scores.shape):
        raise ValueError("weights must be [K] or [K, N] matching scores")
    if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("weights must be finite and between zero and one")
    levels = np.broadcast_to(levels, (scores.shape[1],))
    out = np.full(scores.shape[1], np.inf, dtype=np.result_type(scores.dtype, np.float32))
    if len(scores) == 0:
        return out
    if weights.ndim == 1:
        weights = weights[:, None]
    weights = np.where(np.isfinite(scores), weights, 0.0)
    order = np.argsort(scores, axis=0, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=0)
    cumulative = np.cumsum(np.take_along_axis(weights, order, axis=0), axis=0)
    reached = cumulative >= levels * (1.0 + cumulative[-1])
    ready = reached.any(axis=0)
    columns = np.flatnonzero(ready)
    out[columns] = ordered[np.argmax(reached[:, ready], axis=0), columns]
    return out


def empirical_quantile(scores: np.ndarray, level: float | np.ndarray) -> np.ndarray:
    """Empirical inverse CDF per column [N] of scores [K, N], with NaN holes.

    Use ceil(n * level), clamped to ranks 1 through n, without interpolation or
    an n+1 correction. Levels include 0 and 1; an empty column returns inf.
    Computation and output are float64 for controller initialization.
    """
    scores = np.asarray(scores, dtype=np.float64)
    levels = np.asarray(level, dtype=np.float64)
    if scores.ndim != 2 or levels.ndim > 1:
        raise ValueError("scores must be [K, N] and level a scalar or vector")
    if not np.isfinite(levels).all() or ((levels < 0) | (levels > 1)).any():
        raise ValueError("level must be between zero and one inclusive")
    if np.isinf(scores).any():
        raise ValueError("scores must be finite or NaN")
    levels = np.broadcast_to(levels, (scores.shape[1],))
    counts = np.isfinite(scores).sum(axis=0)
    out = np.full(scores.shape[1], np.inf, dtype=np.float64)
    ready = counts > 0
    if ready.any():
        ordered = np.sort(scores[:, ready], axis=0)
        ranks = np.clip(np.ceil(counts[ready] * levels[ready]).astype(np.int64), 1, counts[ready])
        out[ready] = ordered[ranks - 1, np.arange(ready.sum())]
    return out


def empirical_interval(
    scores: np.ndarray, level: float | np.ndarray, beta_grid_size: int = 21
) -> np.ndarray:
    """Shortest empirical signed-residual interval per column, returned as [N, 2].

    Scores are [K, N], with NaN holes. Quantiles use ceil(n * q), clamped to the
    finite endpoints, not the conformal n+1 correction. Search beta on an inclusive
    grid from zero to 1-level; ties choose the first beta. No scores gives [-inf, inf].
    """
    scores = np.asarray(scores)
    levels = np.asarray(level, dtype=np.float64)
    if scores.ndim != 2 or levels.ndim > 1:
        raise ValueError("scores must be [K, N] and level a scalar or vector")
    if not np.isfinite(levels).all() or ((levels <= 0) | (levels >= 1)).any():
        raise ValueError("level must be strictly between zero and one")
    if (
        isinstance(beta_grid_size, (bool, np.bool_))
        or not isinstance(beta_grid_size, (int, np.integer))
        or beta_grid_size < 2
    ):
        raise ValueError("beta_grid_size must be an integer of at least 2")
    if np.isinf(scores).any():
        raise ValueError("scores must be finite or NaN")
    levels = np.broadcast_to(levels, (scores.shape[1],))
    counts = np.isfinite(scores).sum(axis=0)
    out = np.tile([-np.inf, np.inf], (scores.shape[1], 1))
    ready = counts > 0
    if not ready.any():
        return out.astype(np.float32)
    ordered = np.sort(scores[:, ready].astype(np.float64), axis=0)
    counts = counts[ready]
    levels = levels[ready]
    beta = np.linspace(0, 1, beta_grid_size)[:, None] * (1 - levels)
    low = np.clip(np.ceil(counts * beta).astype(np.int64) - 1, 0, counts - 1)
    high = np.clip(np.ceil(counts * (levels + beta)).astype(np.int64) - 1, 0, counts - 1)
    columns = np.arange(len(counts))
    lower, upper = ordered[low, columns], ordered[high, columns]
    best = np.argmin(upper - lower, axis=0)
    out[ready, 0] = lower[best, columns]
    out[ready, 1] = upper[best, columns]
    return out.astype(np.float32)


def _finite_quantile(scores: np.ndarray, level: float | np.ndarray) -> np.ndarray:
    """Independent column ranks from finite counts; NaN cells supply no score."""
    if np.isfinite(scores).all():
        return score_quantile(scores, level).astype(np.float32)
    counts = np.isfinite(scores).sum(axis=0)
    levels = np.broadcast_to(level, counts.shape)
    # Match score_quantile's level dtype arithmetic, including float32 levels.
    ranks = np.ceil((counts + 1).astype(levels.dtype) * levels).astype(np.int64)
    out = np.full(counts.shape, np.inf, dtype=np.float32)
    ready = (ranks <= counts) & (counts > 0)
    if ready.any():
        ordered = np.sort(np.where(np.isfinite(scores), scores, np.nan), axis=0)
        columns = np.flatnonzero(ready)
        out[columns] = ordered[ranks[columns] - 1, columns]
    return out
