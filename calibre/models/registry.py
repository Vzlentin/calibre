from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibre.models.base import ModelAdapter

_BACKEND_REGISTRY: dict[str, type] = {}


def _ensure_registry() -> None:
    if _BACKEND_REGISTRY:
        return
    from calibre.models.mlforecast import MLForecastAdapter
    from calibre.models.neuralforecast import NeuralForecastAdapter
    from calibre.models.statsforecast import StatsForecastAdapter

    _BACKEND_REGISTRY.update(
        {
            "statsforecast": StatsForecastAdapter,
            "neuralforecast": NeuralForecastAdapter,
            "mlforecast": MLForecastAdapter,
        }
    )


def resolve_adapter(model_config: dict) -> ModelAdapter:
    _ensure_registry()
    backend = model_config.get("backend")
    if not backend:
        raise ValueError(
            "model_config must include a 'backend' key. "
            f"Available backends: {list(_BACKEND_REGISTRY.keys())}"
        )
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend!r}. Available: {list(_BACKEND_REGISTRY.keys())}"
        )
    adapter_cls = _BACKEND_REGISTRY[backend]
    return adapter_cls(model_config)
