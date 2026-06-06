"""Point-forecast reconciliation seam (architecture §9).

Makes point forecasts coherent across a cross-sectional hierarchy. The
pipeline-facing surface is the :class:`Reconciler` Protocol plus the strategy
registry; concrete strategies (no-op, bottom-up, top-down, MinT) are resolved by
name and default to a no-op pass-through.
"""

from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler
from calibre.reconciliation.registry import (
    DEFAULT_STRATEGY,
    available_reconcilers,
    get_reconciler_builder,
    register_reconciler,
    resolve_reconciler,
)

__all__ = [
    "DEFAULT_STRATEGY",
    "NoOpReconciler",
    "Reconciler",
    "available_reconcilers",
    "get_reconciler_builder",
    "register_reconciler",
    "resolve_reconciler",
]
