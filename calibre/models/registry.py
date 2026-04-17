from __future__ import annotations

from calibre.models.base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}

VALID_SCOPES = frozenset({"local", "global"})
DEFAULT_SCOPE = "local"


def _ensure_registry() -> None:
    if _REGISTRY:
        return
    from calibre.models.mlforecast import MLForecastAdapter
    from calibre.models.neuralforecast import NeuralForecastAdapter
    from calibre.models.statsforecast import StatsForecastAdapter

    _REGISTRY.update(
        {
            "statsforecast": StatsForecastAdapter,
            "neuralforecast": NeuralForecastAdapter,
            "mlforecast": MLForecastAdapter,
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


def get_scope(model_config: dict) -> str:
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
