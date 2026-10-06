"""mlforecast adapter: one global regressor behind `fit` and `predict`.

`fit` reads the last `fit_periods` periods and `predict` the last `lookback` periods.
`lookback` must cover the longest lag and lag-transform window. When it does not, the
lag features have NaN and `predict` fails before the model sees them, because models
such as LightGBM accept NaN and return finite, wrong points.

mlforecast reads future values of every dynamic feature, so a covariate that is not
known ahead is rejected. Lag it before the call if the model needs it.
"""

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from mlforecast import MLForecast as Engine

from calibre.forecast.models import frames
from calibre.forecast.models.base import Fitted, Forecaster, Window


class MLForecast(Forecaster):
    """A global model: one regressor fitted on every forecast series at once."""

    def __init__(
        self,
        model: Any,
        lags: Sequence[int] = (),
        lag_transforms: dict[int, list] | None = None,
        date_features: Sequence[Any] = (),
        target_transforms: list | None = None,
        features: Sequence[str] = (),
        fit_periods: int | None = None,
        lookback: int | None = None,
        num_threads: int = 1,
    ) -> None:
        if not (lags or lag_transforms or date_features or features):
            raise ValueError("give at least one of lags, lag_transforms, date_features, features")
        for name, value in (("fit_periods", fit_periods), ("lookback", lookback)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if lookback is not None and lags and lookback < max(lags):
            raise ValueError(f"lookback {lookback} is shorter than the longest lag {max(lags)}")
        self.model = model
        self.lags = list(lags)
        self.lag_transforms = lag_transforms
        self.date_features = list(date_features)
        self.target_transforms = target_transforms
        self.features = list(features)
        self.fit_periods = fit_periods
        self.lookback = lookback
        self.num_threads = num_threads

    def fit(self, window: Window) -> "FittedMLForecast":
        static, future = _features(window, self.features)
        engine = Engine(
            models=[copy.deepcopy(self.model)],
            freq=frames.freq(window),
            lags=self.lags or None,
            lag_transforms=copy.deepcopy(self.lag_transforms),
            date_features=self.date_features or None,
            num_threads=self.num_threads,
            target_transforms=copy.deepcopy(self.target_transforms),
        )
        engine.fit(
            frames.history(window, self.fit_periods, static + future), static_features=static
        )
        return FittedMLForecast(engine, self.features, self.lookback)


class FittedMLForecast(Fitted):
    def __init__(self, engine: Engine, features: list[str], lookback: int | None) -> None:
        self.engine = engine
        self.features = features
        self.lookback = lookback

    def predict(self, window: Window) -> np.ndarray:
        static, future = _features(window, self.features)
        out = self.engine.predict(
            h=window.horizon,
            before_predict_callback=_reject_missing,
            new_df=frames.history(window, self.lookback, static + future),
            X_df=frames.future(window, future) if future else None,
        )
        return frames.points(out, window)


def _features(window: Window, names: list[str]) -> tuple[list[str], list[str]]:
    static, historical, future = frames.split_features(window, names)
    if historical:
        raise ValueError(f"features {historical} are not known ahead; lag them before the call")
    return static, future


def _reject_missing(features: pd.DataFrame) -> pd.DataFrame:
    if features.isna().to_numpy().any():
        raise ValueError("lag features have NaN; lookback is shorter than a lag window")
    return features
