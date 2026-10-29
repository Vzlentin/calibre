"""Finite-sample score quantiles, with rolling retention and pooling.

These functions hold the rank rules of the library. Calibrators call them on the
scores they keep.
"""

import numpy as np

from calibre.conformal.calibrators.base import check_integer


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
    levels = _interval_levels(scores, level, beta_grid_size)
    counts = np.isfinite(scores).sum(axis=0)
    out = np.tile([-np.inf, np.inf], (scores.shape[1], 1)).astype(np.float32)
    ready = counts > 0
    if ready.any():
        ordered = np.sort(scores[:, ready].astype(np.float64), axis=0)
        out[ready] = _shortest_interval(ordered, counts[ready], levels[ready], beta_grid_size)
    return out


def weighted_interval(
    scores: np.ndarray, weights: np.ndarray, level: float, beta_grid_size: int = 21
) -> np.ndarray:
    """Shortest weighted empirical interval [2] from signed scores and weights [K].

    Weights must be finite and nonnegative. NaN scores and zero weights supply
    no mass. Invert the positive support's CDF without interpolation or an n+1
    correction. Empty support gives [-inf, inf]; width ties choose the first beta.
    """
    scores = np.asarray(scores)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 1 or weights.shape != scores.shape:
        raise ValueError("scores and weights must be matching [K] vectors")
    levels = _interval_levels(scores[:, None], level, beta_grid_size)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    support = np.isfinite(scores) & (weights > 0)
    if not support.any():
        return np.array([-np.inf, np.inf], dtype=np.float32)
    values, masses = scores[support].astype(np.float64), weights[support]
    order = np.argsort(values, kind="stable")
    ordered, masses = values[order, None], masses[order]
    cumulative = None
    # Equal mass uses counts to preserve ceil(n*q) at floating-point boundaries.
    if not (masses == masses[0]).all():
        with np.errstate(over="ignore"):
            cumulative = np.cumsum(masses)
        if not np.isfinite(cumulative[-1]):
            cumulative = np.cumsum(masses / masses.max())
        cumulative /= cumulative[-1]
    counts = np.array([len(masses)])
    return _shortest_interval(ordered, counts, levels, beta_grid_size, cumulative)[0]


def _interval_levels(
    scores: np.ndarray, level: float | np.ndarray, beta_grid_size: int
) -> np.ndarray:
    levels = np.asarray(level, dtype=np.float64)
    if scores.ndim != 2 or levels.ndim > 1:
        raise ValueError("scores must be [K, N] and level a scalar or vector")
    if not np.isfinite(levels).all() or ((levels <= 0) | (levels >= 1)).any():
        raise ValueError("level must be strictly between zero and one")
    check_integer("beta_grid_size", beta_grid_size, 2)
    if np.isinf(scores).any():
        raise ValueError("scores must be finite or NaN")
    return np.broadcast_to(levels, (scores.shape[1],))


def _shortest_interval(
    ordered: np.ndarray,
    counts: np.ndarray,
    levels: np.ndarray,
    beta_grid_size: int,
    cumulative: np.ndarray | None = None,
) -> np.ndarray:
    beta = np.linspace(0, 1, beta_grid_size)[:, None] * (1 - levels)
    quantiles = np.clip(np.stack([beta, levels + beta]), 0, 1)
    if cumulative is None:
        ranks = np.clip(np.ceil(counts * quantiles).astype(np.int64) - 1, 0, counts - 1)
    else:
        ranks = np.minimum(np.searchsorted(cumulative, quantiles), counts - 1)
    columns = np.arange(len(counts))
    lower, upper = ordered[ranks, columns]
    best = np.argmin(upper - lower, axis=0)
    return np.stack([lower[best, columns], upper[best, columns]], axis=-1).astype(np.float32)


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
