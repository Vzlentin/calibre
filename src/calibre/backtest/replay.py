"""replay: the thresholds of split calibration at each origin, with their bands and bounds."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from conformalforecast._quantiles import _ceil
from conformalforecast.nonconformity import AbsoluteResidual

from calibre.conformal.split import Split
from calibre.conformal.targets import Target
from calibre.data.panel import Panel
from calibre.forecast.forecasts import Forecasts

_NODES = 4096  # nodes sorted at once: bounds the copy of their score windows


@dataclass(frozen=True)
class Replay:
    """The cells of each node, column and origin: `[N, C, O]`."""

    origins: pd.DatetimeIndex  # [O]: the origins of the forecasts
    point: np.ndarray  # [N, C, O] float32
    truth: np.ndarray  # [N, C, O] float32
    score: np.ndarray  # [N, C, O] float32
    threshold: np.ndarray  # [N, C, O] float32
    lower: np.ndarray  # [N, C, O] float32
    upper: np.ndarray  # [N, C, O] float32
    censored: np.ndarray  # [N, C, O] bool

    def __post_init__(self) -> None:
        if not isinstance(self.origins, pd.DatetimeIndex):
            raise TypeError("origins must be a pandas DatetimeIndex")
        origins = self.origins.copy(deep=True)
        origins.freq = None
        if len(origins) == 0 or origins.hasnans or not (origins[1:] > origins[:-1]).all():
            raise ValueError("origins must be non-empty and strictly increasing")
        shape = None
        for name in ("point", "truth", "score", "threshold", "lower", "upper", "censored"):
            value = getattr(self, name)
            kinds = "b" if name == "censored" else "fiu"
            if not isinstance(value, np.ndarray) or value.dtype.kind not in kinds:
                raise TypeError(f"{name} must be a numpy array of the required kind")
            dtype = bool if name == "censored" else np.float32
            array = np.array(value, dtype=dtype)
            if array.ndim != 3 or 0 in array.shape or array.shape[2] != len(origins):
                raise ValueError(f"{name} must be [N, C, O]")
            if shape is not None and array.shape != shape:
                raise ValueError("the arrays of a Replay must have one shape")
            shape = array.shape
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, "origins", origins)

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            object.__setattr__(self, name, value)


def _dense(panel: Panel) -> tuple[np.ndarray, np.ndarray | None]:
    """Values `[S, T]` float32, NaN outside each series, and the censored flags `[S, T]`."""
    n_series, n_periods = len(panel.series), len(panel.periods)
    lengths = np.diff(panel.indptr)
    rows = np.repeat(np.arange(n_series), lengths)
    cols = (
        np.repeat(panel.starts, lengths)
        + np.arange(len(panel.data))
        - np.repeat(panel.indptr[:-1], lengths)
    )
    values = np.full((n_series, n_periods), np.nan, np.float32)
    values[rows, cols] = panel.data
    censored = None
    if panel.censored is not None:
        censored = np.zeros((n_series, n_periods), bool)
        censored[rows, cols] = panel.censored
    return values, censored


def _column(values: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """Sums `[N, O]` of `values` `[N, U, O]` over the covered steps `cover` `[N, U]`, in
    float64 stored as float32. NaN wins."""
    mask = cover[:, :, None]
    total = np.where(mask, values, 0).sum(axis=1, dtype=np.float64)
    total[(mask & np.isnan(values)).any(axis=1)] = np.nan
    return total.astype(np.float32)


def _level(level: object, n_nodes: int) -> np.ndarray:
    if isinstance(level, bool) or not isinstance(level, (float, np.ndarray)):
        raise TypeError("level must be a float or a numpy array")
    if isinstance(level, np.ndarray) and level.dtype.kind not in "fiu":
        raise TypeError("level must be numeric")
    array = np.asarray(level, dtype=np.float64)
    if array.ndim > 1 or (array.ndim == 1 and len(array) != n_nodes):
        raise ValueError("level must be one value, or one value per node")
    array = np.broadcast_to(array, (n_nodes,))
    if not ((array > 0) & (array < 1)).all():
        raise ValueError("each level must be strictly between 0 and 1")
    return array


def _thresholds(
    scores: np.ndarray, known: np.ndarray, window: int, alpha: np.ndarray
) -> np.ndarray:
    """Thresholds `[n, O]` of one column. Origin o reads the scores at `known[o] - window`
    to `known[o]`, the latest known origins."""
    n_nodes, n_origins = scores.shape
    need = _ceil(1.0 / alpha)
    out = np.full((n_nodes, n_origins), np.inf, np.float32)
    live = np.flatnonzero(known >= need.min())
    if len(live) == 0:
        return out
    padded = np.concatenate([np.full((n_nodes, window), np.nan, np.float32), scores], axis=1)
    view = np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)
    for first in range(0, n_nodes, _NODES):
        rows = slice(first, first + _NODES)
        ordered = np.sort(view[rows][:, known[live], :], axis=2)
        count = (~np.isnan(ordered)).sum(axis=2)
        rank = np.minimum(_ceil((1.0 - alpha[rows])[:, None] * (count + 1)), count)
        index = np.maximum(rank - 1, 0)[:, :, None]
        chosen = np.take_along_axis(ordered, index, axis=2)[:, :, 0]
        ready = count >= need[rows, None]
        out[rows, live] = np.where(ready, chosen, np.float32(np.inf))
    return out


def replay(
    forecasts: Forecasts,
    truth: Panel,
    *,
    target: Target,
    method: Split,
    level: float | np.ndarray,
) -> Replay:
    """Issue, at each origin, the threshold of each node and column from its known scores."""
    if not isinstance(forecasts, Forecasts):
        raise TypeError("forecasts must be a Forecasts")
    if not isinstance(truth, Panel):
        raise TypeError("truth must be a Panel")
    if not isinstance(target, Target):
        raise TypeError("target must be a Target")
    if not isinstance(method, Split):
        raise TypeError("method must be a Split")
    n_models, n_nodes, horizon, n_origins = forecasts.points.shape
    levels = _level(level, n_nodes)
    if n_models != 1:
        raise ValueError("replay takes forecasts of one model: use select")
    if not np.array_equal(forecasts.series, truth.series):
        raise ValueError("the forecast series must be the series of truth, in order")
    origin = truth.periods.get_indexer(forecasts.origins)
    if (origin < 0).any():
        raise ValueError("each origin must be a period of truth")
    cover = np.asarray(target.cover(n_nodes, horizon), dtype=bool)
    if cover.ndim != 3 or cover.shape[0] != n_nodes or cover.shape[2] != horizon:
        raise ValueError("the target cover must be [N, C, H]")
    if not cover.any(axis=2).all():
        raise ValueError("each column must cover at least one step")
    n_columns = cover.shape[1]
    alpha = 1.0 - levels

    values, flags = _dense(truth)
    n_periods = values.shape[1]
    period = origin[None, :] + np.arange(1, horizon + 1)[:, None]
    after = period >= n_periods
    period = np.minimum(period, n_periods - 1)
    points = forecasts.points[0]

    shape = (n_nodes, n_columns, n_origins)
    point = np.empty(shape, np.float32)
    observed = np.empty(shape, np.float32)
    censored = np.zeros(shape, bool)
    for column in range(n_columns):
        used = np.flatnonzero(cover[:, column, :].any(axis=0))
        covered = cover[:, column, used]
        point[:, column] = _column(points[:, used, :], covered)
        steps = values[:, period[used]]
        steps[:, after[used]] = np.nan
        observed[:, column] = _column(steps, covered)
        if flags is not None:
            marks = flags[:, period[used]] & ~after[used]
            censored[:, column] = (marks & covered[:, :, None]).any(axis=1)
    del values, flags
    censored &= ~np.isnan(observed)
    score = np.asarray(method.score.score(point, observed), dtype=np.float32)

    last = horizon - np.argmax(cover[:, :, ::-1], axis=2)
    threshold = np.empty(shape, np.float32)
    for column in range(n_columns):
        for lag in np.unique(last[:, column]):
            nodes = np.flatnonzero(last[:, column] == lag)
            known = np.searchsorted(origin, origin - lag, side="right")
            threshold[nodes, column] = _thresholds(
                score[nodes, column], known, method.window, alpha[nodes]
            )

    missing = np.isnan(point)
    upper = point + threshold
    if isinstance(method.score, AbsoluteResidual):
        lower = point - threshold
    else:
        lower = np.full(shape, -np.inf, np.float32)
        lower[missing] = np.nan
    upper[missing] = np.nan
    return Replay(forecasts.origins, point, observed, score, threshold, lower, upper, censored)
