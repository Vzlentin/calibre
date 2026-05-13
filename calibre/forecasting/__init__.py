"""Forecasting: adapters, ensembles, and feature transforms."""

from calibre.forecasting.adapter_base import ModelAdapter
from calibre.forecasting.adapter_registry import get_adapter_cls, get_scope, resolve_adapter
from calibre.forecasting.ensemble import (
    ensemble_inverse_error,
    ensemble_median,
    ensemble_weighted,
)

__all__ = [
    "ModelAdapter",
    "get_adapter_cls",
    "get_scope",
    "resolve_adapter",
    "ensemble_inverse_error",
    "ensemble_median",
    "ensemble_weighted",
]
