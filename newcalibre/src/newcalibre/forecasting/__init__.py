"""Define forecasting adapter contracts and implementations."""

from collections.abc import Mapping

from newcalibre.forecasting.adapters import SEASONAL_NAIVE_BACKEND, SeasonalNaiveAdapter
from newcalibre.forecasting.protocol import (
    AdapterCapability,
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterDataError,
    AdapterError,
    AdapterLifecycleError,
    ForecastAdapter,
)
from newcalibre.forecasting.registry import AdapterRegistry, AdapterRegistryError

_BUILTIN_ADAPTERS = AdapterRegistry()
_BUILTIN_ADAPTERS.register(SEASONAL_NAIVE_BACKEND, SeasonalNaiveAdapter)


def available_backends() -> tuple[str, ...]:
    """Return the immutable built-in backend view for explicit diagnostics."""
    return _BUILTIN_ADAPTERS.available_backends


def resolve_adapter(model_config: Mapping[str, object]) -> ForecastAdapter:
    """Resolve one explicitly selected built-in without exposing registry mutation."""
    return _BUILTIN_ADAPTERS.resolve(model_config)


__all__ = [
    "SEASONAL_NAIVE_BACKEND",
    "AdapterCapability",
    "AdapterCapabilityError",
    "AdapterConfigurationError",
    "AdapterDataError",
    "AdapterError",
    "AdapterLifecycleError",
    "AdapterRegistry",
    "AdapterRegistryError",
    "ForecastAdapter",
    "SeasonalNaiveAdapter",
    "available_backends",
    "resolve_adapter",
]
