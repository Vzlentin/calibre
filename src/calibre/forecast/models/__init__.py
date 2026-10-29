"""Forecasting models. A model is one module that implements `base.Forecaster`.

The vendor adapters `statsforecast`, `mlforecast`, and `neuralforecast` are not imported
here: each needs its extra, `calibre[stats]`, `calibre[ml]`, or `calibre[neural]`.
"""

from calibre.forecast.models.ar1 import AR1
from calibre.forecast.models.base import (
    Covariate,
    FitResiduals,
    Fitted,
    FittedQuantiles,
    FittedResiduals,
    Forecaster,
    QuantileForecast,
    Window,
)
from calibre.forecast.models.naive import SeasonalNaive

__all__ = [
    "AR1",
    "Covariate",
    "FitResiduals",
    "Fitted",
    "FittedQuantiles",
    "FittedResiduals",
    "Forecaster",
    "QuantileForecast",
    "SeasonalNaive",
    "Window",
]
