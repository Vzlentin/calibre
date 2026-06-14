"""Name-to-class registry for dataset adapters with lazy built-in loading."""

from __future__ import annotations

from collections.abc import Callable

from calibre.execution.dataset import DatasetAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {}
_BUILTINS_LOADED = False


def register_dataset_adapter(name: str) -> Callable[[type[DatasetAdapter]], type[DatasetAdapter]]:
    """Return a class decorator registering a :class:`DatasetAdapter` under ``name``."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("dataset adapter name must be non-empty")

    def _decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        _REGISTRY[key] = cls
        return cls

    return _decorator


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from calibre.execution.m5_adapter import M5DatasetAdapter
    from calibre.execution.vn2_adapter import VN2DatasetAdapter

    _REGISTRY.setdefault("vn2", VN2DatasetAdapter)
    _REGISTRY.setdefault("m5", M5DatasetAdapter)
    _BUILTINS_LOADED = True


def get_dataset_adapter_cls(name: str) -> type[DatasetAdapter]:
    """Look up a registered dataset adapter class by name (loading built-ins first)."""
    _ensure_builtins()
    key = str(name).strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset adapter: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from exc


def resolve_dataset_adapter(name: str) -> DatasetAdapter:
    """Instantiate the dataset adapter registered under ``name``."""
    return get_dataset_adapter_cls(name)()


def available_dataset_adapters() -> list[str]:
    """List the names of all registered dataset adapters, sorted."""
    _ensure_builtins()
    return sorted(_REGISTRY)
