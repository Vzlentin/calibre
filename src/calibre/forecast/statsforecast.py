"""statsforecast adapter: one local model from `statsforecast.models`, series by series.

It calls `model.forecast` on each series and skips the `StatsForecast` class. That class
needs a long frame and an output frame at each origin and stores one fitted model per
series. On 500 series with seasonal naive, the direct loop is 1.1 ms against 5.2 ms for
`StatsForecast.fit` and `predict`, with identical numbers. Heavy models such as AutoARIMA
spend their time in the model itself, and need process parallelism over series instead.
"""

from typing import Any

import numpy as np

from calibre.forecast import Window


class StatsForecastModel:
    """A local model: `predict` fits each series on the window. `fit` returns the model."""

    def __init__(self, model: Any) -> None:
        if not callable(getattr(model, "forecast", None)):
            raise ValueError(f"{model!r} is not a statsforecast.models instance")
        self.model = model

    def fit(self, window: Window) -> "StatsForecastModel":
        return self

    def predict(self, window: Window) -> np.ndarray:
        points = np.empty((window.y.shape[0], window.horizon), dtype=np.float32)
        for index, series in enumerate(window.y):
            points[index] = self.model.forecast(y=series, h=window.horizon)["mean"]
        return points
