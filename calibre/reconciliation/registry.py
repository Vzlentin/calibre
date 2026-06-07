from __future__ import annotations

from collections.abc import Callable

from calibre.reconciliation.nixtla_adapter import NixtlaReconciler, NixtlaStrategy
from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler

ReconcilerBuilder = Callable[[], Reconciler]


def _nixtla_builder(strategy: NixtlaStrategy) -> ReconcilerBuilder:
    return lambda: NixtlaReconciler(strategy)


_REGISTRY: dict[str, ReconcilerBuilder] = {
    "none": NoOpReconciler,
    "bottom_up": _nixtla_builder("bottom_up"),
    "ols": _nixtla_builder("ols"),
    "wls_struct": _nixtla_builder("wls_struct"),
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
    return sorted(_REGISTRY)
