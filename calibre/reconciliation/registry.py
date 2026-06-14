"""Name-to-builder registry resolving reconciliation strategy names to instances."""

from __future__ import annotations

from collections.abc import Callable

from calibre.reconciliation.bottom_up import BottomUpReconciler
from calibre.reconciliation.nixtla_adapter import (
    NIXTLA_POINT_STRATEGIES,
    NixtlaReconciler,
    NixtlaStrategy,
)
from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler

ReconcilerBuilder = Callable[[], Reconciler]


def _nixtla_builder(strategy: NixtlaStrategy) -> ReconcilerBuilder:
    return lambda: NixtlaReconciler(strategy)


_REGISTRY: dict[str, ReconcilerBuilder] = {
    "none": NoOpReconciler,
    "bottom_up": BottomUpReconciler,
    **{strategy: _nixtla_builder(strategy) for strategy in NIXTLA_POINT_STRATEGIES},
}


def resolve_reconciler(name: str) -> Reconciler:
    """Build a reconciler instance from a registered strategy name."""
    key = str(name).strip().lower()
    try:
        builder = _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reconciliation strategy: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from exc
    return builder()


def available_reconcilers() -> list[str]:
    """List the names of all registered reconciliation strategies, sorted."""
    return sorted(_REGISTRY)
