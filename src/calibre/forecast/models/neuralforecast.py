"""neuralforecast adapter: one global neural model behind `fit` and `predict`.

The adapter sets the horizon and the covariate lists from the window. Static
covariates become `stat_exog_list`, historical ones `hist_exog_list`, and known-ahead
ones `futr_exog_list`. `predict` reads the last `input_size` periods, which is what the
model reads. `fit` reads the last `fit_periods` periods.
"""

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast as Engine
from neuralforecast.losses.pytorch import MQLoss

from calibre.forecast.models import frames
from calibre.forecast.models.base import (
    Fitted,
    FittedQuantiles,
    Forecaster,
    QuantileForecast,
    Window,
    check_quantiles,
)

_OWNED = {"h", "stat_exog_list", "hist_exog_list", "futr_exog_list"}
# Lightning prints progress, summaries, and logs by default. A library call stays quiet.
_QUIET = {"enable_progress_bar": False, "enable_model_summary": False, "logger": False}


class NeuralForecast(Forecaster):
    """A global network with optional native quantiles and a median point.

    `quantiles=(low, high)` owns the multi-quantile loss. Do not also set `loss`
    or `valid_loss` in config. Without quantiles, the point-only contract is unchanged.
    """

    def __init__(
        self,
        model_cls: type,
        config: dict[str, Any],
        features: Sequence[str] = (),
        fit_periods: int | None = None,
        *,
        quantiles: tuple[float, float] | None = None,
    ) -> None:
        owned = _OWNED & config.keys()
        if owned:
            raise ValueError(f"config keys {sorted(owned)} are set by the adapter")
        if "input_size" not in config:
            raise ValueError("config needs input_size")
        if fit_periods is not None and fit_periods < 1:
            raise ValueError(f"fit_periods must be at least 1, got {fit_periods}")
        self.quantiles = None if quantiles is None else check_quantiles(quantiles)
        if self.quantiles is not None and {"loss", "valid_loss"} & config.keys():
            raise ValueError("loss and valid_loss are set by the adapter in quantile mode")
        self.model_cls = model_cls
        self.config = dict(config)
        self.features = list(features)
        self.fit_periods = fit_periods

    def fit(self, window: Window) -> "FittedNeuralForecast":
        static, historical, future = frames.split_features(window, self.features)
        config = copy.deepcopy(self.config)
        quantiles = [] if self.quantiles is None else sorted({0.5, *self.quantiles})
        if quantiles:
            loss = MQLoss(quantiles=quantiles)
            # Vendor probability labels are rounded and can collide for nearby levels.
            loss.output_names = [f"-q{i}" for i in range(len(quantiles))]
            config["loss"] = loss
        model = self.model_cls(
            **_QUIET,
            **config,
            h=window.horizon,
            stat_exog_list=static or None,
            hist_exog_list=historical or None,
            futr_exog_list=future or None,
        )
        engine = Engine(models=[model], freq=frames.freq(window))
        engine.fit(
            frames.history(window, self.fit_periods, historical + future),
            static_df=frames.static(window, static) if static else None,
        )
        input_size = self.config["input_size"]
        lookback = input_size if input_size > 0 else None
        if self.quantiles is not None:
            names = [
                repr(engine.models[0]) + f"-q{quantiles.index(q)}" for q in (0.5, *self.quantiles)
            ]
            return FittedQuantileNeuralForecast(
                engine, self.features, lookback, self.quantiles, names
            )
        return FittedNeuralForecast(engine, self.features, lookback)


class FittedNeuralForecast(Fitted):
    def __init__(self, engine: Engine, features: list[str], lookback: int | None) -> None:
        self.engine = engine
        self.features = features
        self.lookback = lookback

    def predict(self, window: Window) -> np.ndarray:
        return frames.points(self._frame(window), window)

    def _frame(self, window: Window) -> pd.DataFrame:
        static, historical, future = frames.split_features(window, self.features)
        return self.engine.predict(
            df=frames.history(window, self.lookback, historical + future),
            static_df=frames.static(window, static) if static else None,
            futr_df=frames.future(window, future) if future else None,
        )


class FittedQuantileNeuralForecast(FittedNeuralForecast, FittedQuantiles):
    def __init__(
        self,
        engine: Engine,
        features: list[str],
        lookback: int | None,
        levels: tuple[float, float],
        names: list[str],
    ) -> None:
        super().__init__(engine, features, lookback)
        self.levels = levels
        self.names = names

    def predict(self, window: Window) -> np.ndarray:
        return self.predict_quantiles(window).point

    def predict_quantiles(self, window: Window) -> QuantileForecast:
        values = frames.predictions(self._frame(window), window, self.names)
        return QuantileForecast(values[..., 0], values[..., 1:], self.levels)
