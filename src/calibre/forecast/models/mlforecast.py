"""mlforecast adapter: one global regressor behind `fit` and `predict`.

`fit` reads the last `fit_periods` periods and `predict` the last `lookback` periods.
`lookback` must cover the longest lag and lag-transform window. When it does not, the
lag features have NaN and `predict` fails before the model sees them, because models
such as LightGBM accept NaN and return finite, wrong points.

mlforecast reads future values of every dynamic feature, so a covariate that is not
known ahead is rejected. Lag it before the call if the model needs it.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from mlforecast import MLForecast as Engine

from calibre.forecast.models import frames
from calibre.forecast.models.base import (
    Fitted,
    FittedQuantiles,
    Forecaster,
    QuantileForecast,
    Quantiles,
    Window,
    check_quantiles,
)


class MLForecast(Forecaster):
    """A global point regressor, with optional direct quantile models per horizon.

    `quantile_models` maps two levels to unfitted quantile estimators. The caller
    configures their objectives. Quantile mode rejects target transforms.
    """

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
        *,
        quantile_models: Mapping[float, Any] | None = None,
    ) -> None:
        if not (lags or lag_transforms or date_features or features):
            raise ValueError("give at least one of lags, lag_transforms, date_features, features")
        for name, value in (("fit_periods", fit_periods), ("lookback", lookback)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if lookback is not None and lags and lookback < max(lags):
            raise ValueError(f"lookback {lookback} is shorter than the longest lag {max(lags)}")
        if quantile_models is not None and target_transforms:
            raise ValueError("native quantiles do not support target_transforms")
        self.quantile_models = (
            None
            if quantile_models is None
            else tuple((q, quantile_models[q]) for q in check_quantiles(sorted(quantile_models)))
        )
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
        data = frames.history(window, self.fit_periods, static + future)
        engine = self._engine([copy.deepcopy(self.model)], frames.freq(window))
        engine.fit(data, static_features=static)
        if self.quantile_models is not None:
            (lower, lower_model), (upper, upper_model) = self.quantile_models
            quantile_engine = self._engine(
                {
                    "lower": copy.deepcopy(lower_model),
                    "upper": copy.deepcopy(upper_model),
                },
                frames.freq(window),
            )
            quantile_engine.fit(
                data,
                static_features=static,
                max_horizon=window.horizon if window.horizon > 1 else None,
            )
            return FittedQuantileMLForecast(
                engine, self.features, self.lookback, quantile_engine, (lower, upper)
            )
        return FittedMLForecast(engine, self.features, self.lookback)

    def _engine(self, models: Any, frequency: str) -> Engine:
        return Engine(
            models=models,
            freq=frequency,
            lags=self.lags or None,
            lag_transforms=copy.deepcopy(self.lag_transforms),
            date_features=self.date_features or None,
            num_threads=self.num_threads,
            target_transforms=copy.deepcopy(self.target_transforms),
        )


class FittedMLForecast(Fitted):
    def __init__(self, engine: Engine, features: list[str], lookback: int | None) -> None:
        self.engine = engine
        self.features = features
        self.lookback = lookback

    def predict(self, window: Window) -> np.ndarray:
        return frames.points(self._frame(window, self.engine), window)

    def _frame(self, window: Window, engine: Engine) -> pd.DataFrame:
        static, future = _features(window, self.features)
        return engine.predict(
            h=window.horizon,
            before_predict_callback=_reject_missing,
            new_df=frames.history(window, self.lookback, static + future),
            X_df=frames.future(window, future) if future else None,
        )


class FittedQuantileMLForecast(FittedMLForecast, FittedQuantiles):
    def __init__(
        self,
        engine: Engine,
        features: list[str],
        lookback: int | None,
        quantile_engine: Engine,
        levels: tuple[float, float],
    ) -> None:
        super().__init__(engine, features, lookback)
        self.quantile_engine = quantile_engine
        self.levels = levels

    def predict_quantiles(self, window: Window) -> QuantileForecast:
        interval = frames.predictions(self._frame(window, self.quantile_engine), window)
        return QuantileForecast(self.predict(window), Quantiles(interval, self.levels))


def _features(window: Window, names: list[str]) -> tuple[list[str], list[str]]:
    static, historical, future = frames.split_features(window, names)
    if historical:
        raise ValueError(f"features {historical} are not known ahead; lag them before the call")
    return static, future


def _reject_missing(features: pd.DataFrame) -> pd.DataFrame:
    if features.isna().to_numpy().any():
        raise ValueError("lag features have NaN; lookback is shorter than a lag window")
    return features
