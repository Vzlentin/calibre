"""combine: one row of points, the models weighted by the losses known at each origin."""

import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import RollingMean

from calibre.data.panel import Panel
from calibre.forecast.forecasts import Forecasts


def _truth(truth: Panel, origin: np.ndarray, horizon: int) -> np.ndarray:
    """The truth of each series, step and origin, as `[S, H, O]` float64, NaN outside a series."""
    period = origin[None, :] + np.arange(1, horizon + 1)[:, None]
    offset = period[None] - truth.starts[:, None, None]
    length = np.diff(truth.indptr)[:, None, None]
    inside = (offset >= 0) & (offset < length)
    # Each series has at least one value, so the row of a cell outside it is a valid row.
    rows = truth.indptr[:-1, None, None] + np.where(inside, offset, 0)
    return np.where(inside, truth.data[rows].astype(np.float64), np.nan)


def combine(forecasts: Forecasts, truth: Panel, *, window: int, best: bool = False) -> Forecasts:
    """One row named `combined`: the mean of the models, weighted by the inverse known loss.

    The loss of a cell is its absolute error. The known loss of a model is the mean of its
    finite losses at the last `window` origins whose truth is known at the origin. With
    `best`, the models with the lowest known loss take all the weight. Before a loss is
    known, the weights are equal. A model with no point is out.
    """
    if not isinstance(forecasts, Forecasts):
        raise TypeError("forecasts must be a Forecasts")
    if not isinstance(truth, Panel):
        raise TypeError("truth must be a Panel")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError("window must be an int")
    if not isinstance(best, bool):
        raise TypeError("best must be a bool")
    if window < 1:
        raise ValueError("window must be at least 1")
    if not np.array_equal(forecasts.series, truth.series):
        raise ValueError("the forecast series must be the series of truth, in order")
    origin = truth.periods.get_indexer(forecasts.origins)
    if (origin < 0).any():
        raise ValueError("each origin must be a period of truth")
    combined = np.array(["combined"])
    points = forecasts.points
    if len(forecasts.models) == 1:
        return Forecasts(points, combined, forecasts.series, forecasts.origins)

    horizon, n_origins = points.shape[2], points.shape[3]
    loss = np.abs(_truth(truth, origin, horizon)[None] - points.astype(np.float64))
    rows = GroupedArray(loss.ravel(), np.arange(0, loss.size + 1, n_origins), 1)
    mean = RollingMean(lag=0, window_size=int(window), min_samples=1, skipna=True).transform(rows)
    mean = mean.reshape(points.shape)
    # The truth of step h at origin k is known at origin o when origin[k] + h <= origin[o].
    last = np.searchsorted(origin, origin - np.arange(1, horizon + 1)[:, None], side="right") - 1
    known = np.take_along_axis(mean, np.broadcast_to(np.maximum(last, 0), mean.shape), axis=3)
    missing = np.isnan(points)
    known_loss = np.where((last >= 0) & ~missing, known, np.nan)

    low = np.where(np.isnan(known_loss), np.inf, known_loss).min(axis=0, keepdims=True)
    with np.errstate(divide="ignore"):
        # The lowest known loss takes all the weight with `best`, or when it is 0.
        weight = np.where(best | (low == 0), known_loss == low, 1.0 / known_loss)
    weight = np.where(np.isnan(known_loss), 0.0, weight)
    weight = np.where(weight.sum(axis=0, keepdims=True) > 0, weight, ~missing)
    total = weight.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        out = (weight * np.nan_to_num(points.astype(np.float64))).sum(axis=0, keepdims=True) / total
    return Forecasts(out.astype(np.float32), combined, forecasts.series, forecasts.origins)
