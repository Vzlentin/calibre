"""The mlforecast engine: fits the models of an `MLForecast` object at each origin."""

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd
from mlforecast import MLForecast

from calibre.data.panel import Panel
from calibre.forecast.window import Forecaster, Window


def _same_freq(freq: object, panel_freq: str) -> bool:
    if isinstance(freq, bool) or not isinstance(freq, (str, pd.offsets.BaseOffset)):
        return False
    try:
        return pd.tseries.frequencies.to_offset(freq) == pd.tseries.frequencies.to_offset(
            panel_freq
        )
    except ValueError:
        return False


class MLForecastEngine(Forecaster):
    """A `Forecaster` that fits a copy of an `MLForecast` object on every series at each
    origin, with the values through the origin, then predicts the horizon."""

    local = False

    def __init__(self, engine: object, panel_freq: str) -> None:
        if type(engine) is not MLForecast:
            raise TypeError(f"an engine must be an MLForecast object, got {type(engine).__name__}")
        freq = engine.freq
        if not _same_freq(freq, panel_freq):
            raise ValueError(f"engine freq {freq!r} is not the panel freq {panel_freq!r}")
        # fit sets attributes, and Calibre changes no attribute of another package's object.
        self.engine = copy.deepcopy(engine)
        self.offset = pd.tseries.frequencies.to_offset(panel_freq)

    @property
    def models(self) -> list[str]:
        """The names of the models, as mlforecast names its columns."""
        return list(self.engine.models)

    def forecast(self, panel: Panel, windows: Sequence[Window], horizon: int) -> np.ndarray:
        """Points `[M, S, H, O]` float32: one fit and one prediction at each window."""
        names = self.models
        out = np.full((len(names), len(panel.series), horizon, len(windows)), np.nan, np.float32)
        position = pd.Index(panel.series)
        for o, window in enumerate(windows):
            active = np.flatnonzero(window.stop > window.start)
            if len(active) == 0:
                continue
            lengths = window.stop[active] - window.start[active]
            rows = np.repeat(window.start[active] - np.cumsum(lengths) + lengths, lengths)
            rows = rows + np.arange(lengths.sum())
            period = rows + np.repeat(panel.starts[active] - panel.indptr[active], lengths)
            train = pd.DataFrame(
                {
                    "unique_id": np.repeat(panel.series[active], lengths),
                    "ds": panel.periods[period],
                    "y": panel.data[rows],
                }
            )
            self.engine.fit(train)
            predicted = self.engine.predict(h=horizon)
            series = position.get_indexer(predicted["unique_id"].astype(str))
            steps = pd.date_range(
                panel.periods[window.origin], periods=horizon + 1, freq=self.offset
            )
            step = steps.get_indexer(pd.DatetimeIndex(predicted["ds"]))
            if (series < 0).any() or (step < 1).any():
                raise ValueError("mlforecast predicted a series or a date outside the window")
            for m, name in enumerate(names):
                out[m, series, step - 1, o] = predicted[name].to_numpy().astype(np.float32)
        return out
