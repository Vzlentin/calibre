"""Compose a forecaster with the array-only scorecaster contract."""

import numpy as np
import pandas as pd

from calibre.conformal.calibrators.base import Scorecaster
from calibre.forecast.models.base import Forecaster, Window


class ForecastScorecaster(Scorecaster):
    """Refit on each cell history, batching nodes with equal retained counts.

    Below min_history use the retained mean (zero if empty). With scale_floor,
    divide by max(seed scale, scale_floor) before fitting and undo it afterwards.
    Dates are positional daily indexes, not the original calendar. No fitted model
    is retained: the caller saves only the PID arrays and recreates configuration.
    """

    def __init__(
        self,
        forecaster: Forecaster,
        *,
        min_history: int = 3,
        scale_floor: float | None = None,
    ) -> None:
        if not isinstance(forecaster, Forecaster):
            raise ValueError("forecaster must implement Forecaster")
        if (
            isinstance(min_history, (bool, np.bool_))
            or not isinstance(min_history, (int, np.integer))
            or min_history < 1
        ):
            raise ValueError("min_history must be a positive integer")
        if scale_floor is not None:
            floor = np.asarray(scale_floor)
            if floor.ndim != 0 or not np.isfinite(floor) or float(floor) <= 0:
                raise ValueError("scale_floor must be a positive finite scalar or None")
        self.forecaster = forecaster
        self.min_history = int(min_history)
        self.scale_floor = None if scale_floor is None else float(scale_floor)

    def predict(
        self, scores: np.ndarray, counts: np.ndarray, scale: np.ndarray, ahead: np.ndarray
    ) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        counts, scale, ahead = np.asarray(counts), np.asarray(scale), np.asarray(ahead)
        if scores.ndim != 3 or counts.shape != scores.shape[:2] or scale.shape != counts.shape:
            raise ValueError("scores [N, C, K], counts and scale [N, C] must align")
        if counts.dtype.kind not in "iu" or ((counts < 0) | (counts > scores.shape[-1])).any():
            raise ValueError("counts must be integers between zero and K")
        if ahead.shape != (scores.shape[1],) or ahead.dtype.kind not in "iu" or (ahead < 1).any():
            raise ValueError("ahead must be positive integers [C]")
        valid = np.arange(scores.shape[-1]) >= scores.shape[-1] - counts[..., None]
        if not np.array_equal(np.isfinite(scores), valid) or np.isinf(scores).any():
            raise ValueError("scores must be finite right-aligned histories with leading NaN")
        if not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("scale must be finite and nonnegative")
        out = np.where(valid, scores, 0).sum(axis=-1) / np.maximum(counts, 1)
        for column, horizon in enumerate(ahead):
            for length in np.unique(counts[:, column]):
                if length < self.min_history:
                    continue
                nodes = np.flatnonzero(counts[:, column] == length)
                normalizer = (
                    np.ones(len(nodes))
                    if self.scale_floor is None
                    else np.maximum(scale[nodes, column], self.scale_floor)
                )
                history = scores[nodes, column, -int(length) :] / normalizer[:, None]
                history.flags.writeable = False
                window = Window(
                    y=history,
                    x={},
                    periods=pd.date_range("1970-01-01", periods=int(length + horizon), freq="D"),
                    horizon=int(horizon),
                )
                fitted = self.forecaster.fit(window)
                prediction = np.asarray(fitted.predict(window), dtype=np.float64)
                if (
                    prediction.shape != (len(nodes), int(horizon))
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError("forecaster must return finite predictions [S, horizon]")
                out[nodes, column] = prediction[:, -1] * normalizer
        if not np.isfinite(out).all():
            raise ValueError("scorecaster must return finite predictions in score units")
        return out
