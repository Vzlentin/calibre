from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibre.reconciliation.protocols import Reconciler

ReconcilerBuilder = Callable[..., Reconciler]

DEFAULT_STRATEGY = "none"

_REGISTRY: dict[str, ReconcilerBuilder] = {}
_BUILTINS_LOADED = False


def register_reconciler(name: str) -> Callable[[ReconcilerBuilder], ReconcilerBuilder]:
    """Register a reconciler builder under ``name`` (mirrors the dataset registry)."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("reconciler name must be non-empty")

    def _decorator(builder: ReconcilerBuilder) -> ReconcilerBuilder:
        _REGISTRY[key] = builder
        return builder

    return _decorator


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from calibre.reconciliation.noop import NoOpReconciler
    from calibre.reconciliation.strategies import BottomUpReconciler, TopDownReconciler

    _REGISTRY.setdefault("none", NoOpReconciler)
    _REGISTRY.setdefault("bottom_up", BottomUpReconciler)
    _REGISTRY.setdefault("top_down", TopDownReconciler)
    _BUILTINS_LOADED = True


def get_reconciler_builder(name: str) -> ReconcilerBuilder:
    _ensure_builtins()
    key = str(name).strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reconciliation strategy: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from exc


def resolve_reconciler(name: str, **kwargs: Any) -> Reconciler:
    """Build a reconciler instance from a registered strategy name."""
    return get_reconciler_builder(name)(**kwargs)


def available_reconcilers() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)
