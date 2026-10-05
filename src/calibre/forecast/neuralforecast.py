"""neuralforecast adapter: one global neural model behind `fit` and `predict`.

The adapter sets the horizon and the covariate lists from the window. Static
covariates become `stat_exog_list`, historical ones `hist_exog_list`, and known-ahead
ones `futr_exog_list`. `predict` reads the last `input_size` periods, which is what the
model reads. `fit` reads the last `fit_periods` periods.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from neuralforecast import NeuralForecast as Engine

from calibre.forecast import Window, _frames

_OWNED = {"h", "stat_exog_list", "hist_exog_list", "futr_exog_list"}
# Lightning prints progress, summaries, and logs by default. A library call stays quiet.
_QUIET = {"enable_progress_bar": False, "enable_model_summary": False, "logger": False}


class NeuralForecast:
    """A global model: one `neuralforecast.models` network fitted on every forecast series."""

    def __init__(
        self,
        model_cls: type,
        config: dict[str, Any],
        features: Sequence[str] = (),
        fit_periods: int | None = None,
    ) -> None:
        owned = _OWNED & config.keys()
        if owned:
            raise ValueError(f"config keys {sorted(owned)} are set by the adapter")
        if "input_size" not in config:
            raise ValueError("config needs input_size")
        if fit_periods is not None and fit_periods < 1:
            raise ValueError(f"fit_periods must be at least 1, got {fit_periods}")
        self.model_cls = model_cls
        self.config = dict(config)
        self.features = list(features)
        self.fit_periods = fit_periods

    def fit(self, window: Window) -> "FittedNeuralForecast":
        static, historical, future = _frames.split_features(window, self.features)
        model = self.model_cls(
            **_QUIET,
            **self.config,
            h=window.horizon,
            stat_exog_list=static or None,
            hist_exog_list=historical or None,
            futr_exog_list=future or None,
        )
        engine = Engine(models=[model], freq=_frames.freq(window))
        engine.fit(
            _frames.history(window, self.fit_periods, historical + future),
            static_df=_frames.static(window, static) if static else None,
        )
        input_size = self.config["input_size"]
        return FittedNeuralForecast(engine, self.features, input_size if input_size > 0 else None)


class FittedNeuralForecast:
    def __init__(self, engine: Engine, features: list[str], lookback: int | None) -> None:
        self.engine = engine
        self.features = features
        self.lookback = lookback

    def predict(self, window: Window) -> np.ndarray:
        static, historical, future = _frames.split_features(window, self.features)
        out = self.engine.predict(
            df=_frames.history(window, self.lookback, historical + future),
            static_df=_frames.static(window, static) if static else None,
            futr_df=_frames.future(window, future) if future else None,
        )
        return _frames.points(out, window)
