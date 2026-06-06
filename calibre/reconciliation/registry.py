from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibre.reconciliation.mint import build_mint_reconciler
from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler
from calibre.reconciliation.strategies import BottomUpReconciler, TopDownReconciler

ReconcilerBuilder = Callable[..., Reconciler]

_REGISTRY: dict[str, ReconcilerBuilder] = {
    "none": NoOpReconciler,
    "bottom_up": BottomUpReconciler,
    "top_down": TopDownReconciler,
    "mint": build_mint_reconciler,
}


def resolve_reconciler(name: str, **kwargs: Any) -> Reconciler:
    """Build a reconciler instance from a registered strategy name."""
    key = str(name).strip().lower()
    try:
        builder = _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reconciliation strategy: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from exc
    return builder(**kwargs)


def available_reconcilers() -> list[str]:
    return sorted(_REGISTRY)
