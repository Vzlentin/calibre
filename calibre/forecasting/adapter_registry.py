from __future__ import annotations

from typing import Literal, get_args

from calibre.forecasting.adapter_base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}

ScopeType = Literal["local", "global"]
VALID_SCOPES: frozenset[ScopeType] = frozenset(get_args(ScopeType))
DEFAULT_SCOPE: ScopeType = "local"


def _ensure_registry() -> None:
    if _REGISTRY:
        return
    from calibre.forecasting.mlforecast_adapter import MLForecastAdapter
    from calibre.forecasting.neuralforecast_adapter import NeuralForecastAdapter
    from calibre.forecasting.statsforecast_adapter import StatsForecastAdapter

    # NeuralForecastAdapter imports neuralforecast lazily; instantiation raises
    # a clear RuntimeError when the optional extra isn't installed.
    _REGISTRY.update(
        {
            "statsforecast": StatsForecastAdapter,
            "mlforecast": MLForecastAdapter,
            "neuralforecast": NeuralForecastAdapter,
        }
    )


def get_adapter_cls(model_config: dict) -> type[ModelAdapter]:
    """Return the adapter class for the given config without instantiating it."""
    _ensure_registry()
    backend = model_config.get("backend")
    if not backend:
        raise ValueError(
            f"model_config must include a 'backend' key. "
            f"Available backends: {list(_REGISTRY.keys())}"
        )
    if backend not in _REGISTRY:
        raise ValueError(f"Unknown backend: {backend!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[backend]


def get_scope(model_config: dict) -> ScopeType:
    """Return the dispatch scope for a model config.

    'local' fits one model per unique_id (Fugue-partitioned).
    'global' fits one model per config across all unique_ids.
    """
    scope = model_config.get("scope", DEFAULT_SCOPE)
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unknown scope: {scope!r}. Valid scopes: {sorted(VALID_SCOPES)}")
    return scope


def resolve_adapter(model_config: dict) -> ModelAdapter:
    """Instantiate and return an adapter for the given model config."""
    return get_adapter_cls(model_config)(model_config)
