from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibre.models.base import ModelAdapter

_ADAPTER_REGISTRY: dict[str, type] = {}


def _ensure_registry() -> None:
    if _ADAPTER_REGISTRY:
        return
    from calibre.models.mlforecast import MLForecastAdapter
    from calibre.models.nixtla import StatsForecastAdapter

    _ADAPTER_REGISTRY.update(
        {
            "SeasonalNaive": StatsForecastAdapter,
            "AutoARIMA": StatsForecastAdapter,
            "LightGBM": MLForecastAdapter,
            "XGBoost": MLForecastAdapter,
        }
    )


def resolve_adapter(model_config: dict) -> ModelAdapter:
    _ensure_registry()
    model_name = model_config["model"]
    if model_name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    adapter_cls = _ADAPTER_REGISTRY[model_name]
    return adapter_cls(model_config)
