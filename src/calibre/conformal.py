"""Model-independent score quantiles, per-step bands, and window-sum bounds."""

import numpy as np


def score_quantile(scores: np.ndarray, level: float | np.ndarray) -> np.ndarray:
    """Finite-sample quantile of finite scores along axis 0, without interpolation.

    A vector is one pool; a matrix is independent pools in columns. `level` is a
    scalar or one value per output. The rank is ceil((n + 1) * level); insufficient
    samples give inf. The caller owns score construction, scaling, grouping, and
    which observations are eligible. No model or forecast history is needed.
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


def _finite_quantile(scores: np.ndarray, level: float | np.ndarray) -> np.ndarray:
    """Independent column ranks from finite counts; NaN cells supply no score."""
    if np.isfinite(scores).all():
        return score_quantile(scores, level)
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


def _correction(
    scores: np.ndarray,
    level: float | np.ndarray,
    groups: np.ndarray | None,
    window: int | None,
) -> np.ndarray:
    """Keep last K resolved origin rows per pool, then pool their finite scores.

    A resolved row has at least one finite score in that pool. For a single-node
    pool this means that node is resolved. Holes do not consume rows. A late old
    score can enter only by its original row position, never its arrival order.
    """
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


def _check_window(window: int | None) -> None:
    if window is not None and window < 1:
        raise ValueError(f"window must be at least 1, got {window}")


def resolved_rows(origins: np.ndarray, index: int, horizon: int) -> np.ndarray:
    """Count target-complete prior origins at each step, not finite scores.

    Origin j's step s is target-complete when origins[j] + s <= origins[index].
    Sorted origins make these candidate rows a prefix. Missing or late actuals
    leave holes inside it. Daily origins give index - s + 1 candidate rows.
    """
    steps = np.arange(1, horizon + 1)
    return np.searchsorted(origins, origins[index] - steps, side="right")


def width(
    resid: np.ndarray,
    rows: np.ndarray,
    level: float,
    window: int | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Half-width per node and step from resolved absolute residuals.

    `window` retains the last resolved origins per pool and step, before pooling. `groups`
    has one integer per node: equal labels pool scores, None keeps nodes separate.
    Steps never share scores. Pooling does not imply per-node coverage.

    Sort one [K, N] slice per step, not a full [K, N, H] copy. On M5 (63 rows) sort is
    3x faster than introselect (120 ms against 350 ms per origin), and a per-origin
    [K, N, H] copy made macOS malloc hold 8 GB of freed blocks.
    """
    _check_window(window)
    _, n_nodes, horizon = resid.shape
    out = np.full((n_nodes, horizon), np.inf, dtype=np.float32)
    for step in range(horizon):
        stop = int(rows[step])
        scores = resid[:stop, :, step]
        # Fast path: when the last K rows are complete, they are the retained rows.
        if window is not None and np.isfinite(scores[-window:]).all():
            scores = scores[-window:]
        out[:, step] = _correction(np.abs(scores), level, groups, window)
    return out


def window_bound(
    point: np.ndarray,
    resid: np.ndarray,
    rows: int,
    level: float | np.ndarray,
    steps: int,
    window: int | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """One-sided upper bound on the total of the first `steps` forecast steps.

    Scores are signed residual sums over those steps, one per origin and row of `point`.
    Steps are summed before calibration, so this is not a sum of per-step bounds. Only
    the first `rows` origins are eligible, and a score needs every step resolved. Rows of
    `point` are the first nodes of `resid`. `groups` has one integer per row of `point`.
    `level` is a scalar or one value per row, also within a pool. No scores or an
    unsupported rank gives inf. The bound is not projected onto the target support.
    """
    _check_window(window)
    n_rows = point.shape[0]
    residuals = resid[:rows, :n_rows, :steps]
    # Select finite window scores, not finite cells: a finite row can sum to inf.
    scores = residuals[-window:].sum(axis=2) if window is not None else residuals.sum(axis=2)
    if window is not None and not np.isfinite(scores).all():
        scores = residuals.sum(axis=2)
    return point[:, :steps].sum(axis=1) + _correction(scores, level, groups, window)
