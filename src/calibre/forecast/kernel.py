"""Kernel models: local `Forecaster` models on the arrays of the panel, for all origins at once."""

from collections.abc import Sequence

import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import RollingMean

from calibre.data.panel import Panel
from calibre.forecast.window import Forecaster, Window


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


def _edges(windows: Sequence[Window]) -> tuple[np.ndarray, np.ndarray]:
    """The `start` and `stop` of each window, as `[S, O]` arrays."""
    start = np.stack([window.start for window in windows], axis=1)
    stop = np.stack([window.stop for window in windows], axis=1)
    return start, stop


class WindowAverage(Forecaster):
    """The mean of the last `window_size` values of the window, at every origin.

    A window with fewer than `window_size` values gives NaN.
    """

    local = True

    def __init__(self, window_size: int, num_threads: int = 1) -> None:
        self.window_size = _positive_int("window_size", window_size)
        self.num_threads = _positive_int("num_threads", num_threads)

    @property
    def models(self) -> Sequence[str]:
        return ("WindowAverage",)

    def forecast(self, panel: Panel, windows: Sequence[Window], horizon: int) -> np.ndarray:
        start, stop = _edges(windows)
        seen = stop - start >= self.window_size
        size = self.window_size
        # Each window is its own group, so its mean reads only its own values. The sum runs
        # in float64: in float32 it drifts, and a run of zeros does not give 0.
        last = stop[seen]
        rows = (last[:, None] - size + np.arange(size)[None, :]).ravel()
        groups = GroupedArray(
            panel.data[rows].astype(np.float64),
            np.arange(0, len(rows) + 1, size, dtype=np.int64),
            self.num_threads,
        )
        mean = RollingMean(lag=0, window_size=size).transform(groups)
        point = np.full(stop.shape, np.nan)
        point[seen] = mean[size - 1 :: size]
        shape = (1, len(panel.series), horizon, len(windows))
        return np.broadcast_to(point[None, :, None, :], shape).astype(np.float32)


class SeasonalNaive(Forecaster):
    """The value one season before each step, at every origin.

    A window with fewer than `season_length` values gives NaN.
    """

    local = True

    def __init__(self, season_length: int) -> None:
        self.season_length = _positive_int("season_length", season_length)

    @property
    def models(self) -> Sequence[str]:
        return ("SeasonalNaive",)

    def forecast(self, panel: Panel, windows: Sequence[Window], horizon: int) -> np.ndarray:
        start, stop = _edges(windows)
        back = np.arange(horizon) % self.season_length - self.season_length
        rows = stop[:, None, :] + back[None, :, None]
        seen = (stop - start >= self.season_length)[:, None, :]
        points = np.where(seen, panel.data[np.maximum(rows, 0)], np.nan)
        return points[None].astype(np.float32)
