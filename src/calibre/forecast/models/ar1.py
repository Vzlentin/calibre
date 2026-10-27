"""Regularized AR(1) forecasts with an intercept and bounded persistence."""

import numpy as np

from calibre.forecast.models.base import Fitted, Forecaster, Window


class AR1(Forecaster, Fitted):
    """Local AR(1) requiring three values, with variance-relative ridge.

    Forecasts are not clipped to support.
    """

    def __init__(self, ridge: float = 1e-6, slope_limit: float = 0.99) -> None:
        for name, value in (("ridge", ridge), ("slope_limit", slope_limit)):
            array = np.asarray(value)
            if array.ndim != 0 or not np.isfinite(array) or float(array) <= 0:
                raise ValueError(f"{name} must be a positive finite scalar")
        if slope_limit >= 1:
            raise ValueError("slope_limit must be strictly below one")
        self.ridge = float(ridge)
        self.slope_limit = float(slope_limit)

    def fit(self, window: Window) -> "AR1":
        return self

    def predict(self, window: Window) -> np.ndarray:
        history = np.asarray(window.y, dtype=np.float64)
        if history.shape[1] < 3:
            raise ValueError("AR1 requires at least three history values")
        if window.x:
            raise ValueError("AR1 does not support covariates")
        x, y = history[:, :-1], history[:, 1:]
        mean_x, mean_y = x.mean(axis=1), y.mean(axis=1)
        centered_x, centered_y = x - mean_x[:, None], y - mean_y[:, None]
        variance = np.mean(centered_x**2, axis=1)
        denominator = np.sum(centered_x**2, axis=1) + self.ridge * variance
        slope = np.divide(
            np.sum(centered_x * centered_y, axis=1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        slope = np.clip(slope, -self.slope_limit, self.slope_limit)
        equilibrium = (mean_y - slope * mean_x) / (1 - slope)
        steps = np.arange(1, window.horizon + 1)
        return (
            equilibrium[:, None] + slope[:, None] ** steps * (history[:, -1] - equilibrium)[:, None]
        )
