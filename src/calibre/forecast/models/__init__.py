"""Forecasting models. A model is one module that implements `base.Forecaster`.

The vendor adapters `statsforecast`, `mlforecast`, and `neuralforecast` are not imported
here: each needs its extra, `calibre[stats]`, `calibre[ml]`, or `calibre[neural]`.
"""

from calibre.forecast.models.base import Covariate, Fitted, Forecaster, Window
from calibre.forecast.models.naive import SeasonalNaive

__all__ = ["Covariate", "Fitted", "Forecaster", "SeasonalNaive", "Window"]
