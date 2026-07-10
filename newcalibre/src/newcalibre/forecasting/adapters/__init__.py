"""Provide forecasting adapter implementations."""

from newcalibre.forecasting.adapters.seasonal_naive import (
    SEASONAL_NAIVE_BACKEND,
    SeasonalNaiveAdapter,
)

__all__ = ["SEASONAL_NAIVE_BACKEND", "SeasonalNaiveAdapter"]
