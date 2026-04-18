"""Composable feature-engineering transforms for forecasting pipelines."""

from calibre.features.calendar import add_calendar_features
from calibre.features.censoring import add_stockout_features
from calibre.features.lags import add_lag_features, add_rolling_features
from calibre.features.pipeline import build_training_frame
from calibre.features.scaling import add_series_scaling
from calibre.features.static import add_static_features
from calibre.features.weights import add_time_weights

__all__ = [
    "add_calendar_features",
    "add_lag_features",
    "add_rolling_features",
    "add_series_scaling",
    "add_static_features",
    "add_stockout_features",
    "add_time_weights",
    "build_training_frame",
]
