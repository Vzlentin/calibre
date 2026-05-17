from __future__ import annotations

from collections.abc import Callable

from calibre.execution.dataset import DatasetAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {}
_BUILTINS_LOADED = False


def register_dataset_adapter(name: str) -> Callable[[type[DatasetAdapter]], type[DatasetAdapter]]:
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
    try:
        from benchmarks.vn2.dataset import VN2DatasetAdapter
    except ImportError:
        pass
    else:
        _REGISTRY.setdefault("vn2", VN2DatasetAdapter)
    _BUILTINS_LOADED = True


def get_dataset_adapter_cls(name: str) -> type[DatasetAdapter]:
    _ensure_builtins()
    key = str(name).strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset adapter: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from exc


def resolve_dataset_adapter(name: str) -> DatasetAdapter:
    return get_dataset_adapter_cls(name)()


def available_dataset_adapters() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)
