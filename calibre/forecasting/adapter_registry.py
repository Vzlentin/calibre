from __future__ import annotations

import importlib
from typing import Literal, get_args

from calibre.forecasting.adapter_base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}
_ADAPTERS: dict[str, tuple[str, str]] = {
    "statsforecast": (
        "calibre.forecasting.statsforecast_adapter",
        "StatsForecastAdapter",
    ),
    "mlforecast": (
        "calibre.forecasting.mlforecast_adapter",
        "MLForecastAdapter",
    ),
    "neuralforecast": (
        "calibre.forecasting.neuralforecast_adapter",
        "NeuralForecastAdapter",
    ),
}

ScopeType = Literal["local", "global"]
VALID_SCOPES: frozenset[ScopeType] = frozenset(get_args(ScopeType))
DEFAULT_SCOPE: ScopeType = "local"


def _available_backends() -> list[str]:
    return sorted(_ADAPTERS)


def get_adapter_cls(model_config: dict) -> type[ModelAdapter]:
    """Return the adapter class for the given config without instantiating it."""
    backend = model_config.get("backend")
    if not backend:
        raise ValueError(
            f"model_config must include a 'backend' key. "
            f"Available backends: {_available_backends()}"
        )
    if backend not in _ADAPTERS:
        raise ValueError(f"Unknown backend: {backend!r}. Available: {_available_backends()}")
    if backend not in _REGISTRY:
        module_name, class_name = _ADAPTERS[backend]
        module = importlib.import_module(module_name)
        _REGISTRY[backend] = getattr(module, class_name)
    return _REGISTRY[backend]


def get_scope(model_config: dict) -> ScopeType:
    """Return the dispatch scope for a model config.

    'local' fits one model per unique_id.
    'global' fits one model per config across all unique_ids.
    """
    scope = model_config.get("scope", DEFAULT_SCOPE)
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unknown scope: {scope!r}. Valid scopes: {sorted(VALID_SCOPES)}")
    return scope


def resolve_adapter(model_config: dict) -> ModelAdapter:
    """Instantiate and return an adapter for the given model config."""
    return get_adapter_cls(model_config)(model_config)
