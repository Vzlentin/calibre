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
