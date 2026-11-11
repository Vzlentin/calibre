"""The statsforecast engine: runs the models of a `StatsForecast` object on windows."""

import multiprocessing
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.core import _controller

from calibre.data.panel import Panel
from calibre.forecast.window import Forecaster, Window


def _n_jobs(engine: StatsForecast, n_series: int) -> int:
    n_jobs = engine.n_jobs
    if n_jobs is None or n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive int or -1, got {engine.n_jobs!r}")
    return max(1, min(n_jobs, n_series))


def _run_chunk(
    models: Sequence[Any],
    fallback: Any,
    data: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Points `[M, s, H, O]` of a chunk of series: `start` and `stop` are `[s, O]`."""
    n_series, n_origins = start.shape
    out = np.full((len(models), n_series, horizon, n_origins), np.nan, dtype=np.float32)
    with _controller.limit(limits=1):
        for series in range(n_series):
            for origin in range(n_origins):
                lo, hi = start[series, origin], stop[series, origin]
                if lo == hi:
                    continue
                values = data[lo:hi]
                for index, model in enumerate(models):
                    try:
                        result = model.forecast(y=values, h=horizon)
                    except Exception:
                        if fallback is None:
                            raise
                        result = fallback.forecast(y=values, h=horizon)
                    out[index, series, :, origin] = result["mean"]
    return out


def _same_freq(freq: object, panel_freq: str) -> bool:
    if not isinstance(freq, str):
        return False
    try:
        return pd.tseries.frequencies.to_offset(freq) == pd.tseries.frequencies.to_offset(
            panel_freq
        )
    except ValueError:
        return False


class StatsForecastEngine(Forecaster):
    """A `Forecaster` that runs the models of a `StatsForecast` object, one series at a time."""

    local = True

    def __init__(self, engine: object, panel_freq: str) -> None:
        if type(engine) is not StatsForecast:
            raise TypeError(
                f"an engine must be a StatsForecast object, got {type(engine).__name__}"
            )
        freq = getattr(engine, "freq", None)
        if not _same_freq(freq, panel_freq):
            raise ValueError(f"engine freq {freq!r} is not the panel freq {panel_freq!r}")
        self.engine = engine

    @property
    def models(self) -> list[str]:
        """The names of the models, as statsforecast names its columns."""
        return [repr(model) for model in self.engine.models]

    def forecast(self, panel: Panel, windows: Sequence[Window], horizon: int) -> np.ndarray:
        """Points `[M, S, H, O]` float32 of the models of the engine at each window."""
        engine = self.engine
        models = list(engine.models)
        fallback = engine.fallback_model
        start = np.stack([window.start for window in windows], axis=1)
        stop = np.stack([window.stop for window in windows], axis=1)
        n_series = len(panel.series)
        n_jobs = _n_jobs(engine, n_series)
        if n_jobs == 1:
            return _run_chunk(models, fallback, panel.data, start, stop, horizon)
        chunks = np.array_split(np.arange(n_series), n_jobs)
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(n_jobs)
        try:
            pending = []
            for chunk in chunks:
                first, last = panel.indptr[chunk[0]], panel.indptr[chunk[-1] + 1]
                pending.append(
                    pool.apply_async(
                        _run_chunk,
                        (
                            models,
                            fallback,
                            np.array(panel.data[first:last]),
                            start[chunk] - first,
                            stop[chunk] - first,
                            horizon,
                        ),
                    )
                )
            parts = [task.get() for task in pending]
            pool.close()
        finally:
            pool.terminate()
            pool.join()
        return np.concatenate(parts, axis=1)
