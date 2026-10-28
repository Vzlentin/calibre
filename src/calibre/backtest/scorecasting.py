"""Compose a forecaster with the array-only scorecaster contract."""

import numpy as np
import pandas as pd

from calibre.conformal.calibrators.base import Scorecaster
from calibre.forecast.models.base import Forecaster, Window


class ForecastScorecaster(Scorecaster):
    """Refit by retained history length; use the mean below min_history."""

    def __init__(self, forecaster: Forecaster, *, min_history: int = 3) -> None:
        if not isinstance(forecaster, Forecaster):
            raise ValueError("forecaster must implement Forecaster")
        if (
            isinstance(min_history, (bool, np.bool_))
            or not isinstance(min_history, (int, np.integer))
            or min_history < 1
        ):
            raise ValueError("min_history must be a positive integer")
        self.forecaster = forecaster
        self.min_history = int(min_history)

    def predict(self, history: np.ndarray, ahead: np.ndarray) -> np.ndarray:
        cells = history.reshape(-1, history.shape[-1])
        counts = np.isfinite(cells).sum(axis=-1)
        out = np.nansum(cells, axis=-1) / np.maximum(counts, 1)
        horizons = np.broadcast_to(ahead, history.shape[:2]).ravel()
        horizon = int(ahead.max())
        for length in np.unique(counts[counts >= self.min_history]):
            indices = np.flatnonzero(counts == length)
            window = Window(
                y=cells[indices, -int(length) :],
                x={},
                # Window requires dates; these index retained scores, not calendar time.
                periods=pd.date_range("1970-01-01", periods=int(length) + horizon, freq="D"),
                horizon=horizon,
            )
            prediction = np.asarray(self.forecaster.fit(window).predict(window), dtype=np.float64)
            if prediction.shape != (len(indices), horizon) or not np.isfinite(prediction).all():
                raise ValueError("forecaster must return finite predictions [S, horizon]")
            out[indices] = prediction[np.arange(len(indices)), horizons[indices] - 1]
        return out.reshape(history.shape[:2])
