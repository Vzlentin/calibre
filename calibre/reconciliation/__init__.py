"""Point-forecast reconciliation seam (architecture §9).

Makes point forecasts coherent across a cross-sectional hierarchy. The
pipeline-facing surface is the :class:`Reconciler` Protocol plus the strategy
registry; concrete strategies (no-op, bottom-up, top-down, MinT) are resolved by
name and default to a no-op pass-through.
"""

from calibre.reconciliation.apply import NODE_LABEL, VectorReconciler
from calibre.reconciliation.mint import (
    MinTReconciler,
    build_mint_reconciler,
    mint_projection,
    ols_projection,
    schafer_strimmer_shrinkage,
)
from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler
from calibre.reconciliation.registry import (
    DEFAULT_STRATEGY,
    available_reconcilers,
    get_reconciler_builder,
    register_reconciler,
    resolve_reconciler,
)
from calibre.reconciliation.strategies import BottomUpReconciler, TopDownReconciler
from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    SummingMatrix,
    build_summing_matrix,
)

__all__ = [
    "DEFAULT_STRATEGY",
    "NODE_LABEL",
    "TOTAL_LABEL",
    "BottomUpReconciler",
    "MinTReconciler",
    "NoOpReconciler",
    "Reconciler",
    "SummingMatrix",
    "TopDownReconciler",
    "VectorReconciler",
    "available_reconcilers",
    "build_mint_reconciler",
    "build_summing_matrix",
    "get_reconciler_builder",
    "mint_projection",
    "ols_projection",
    "register_reconciler",
    "resolve_reconciler",
    "schafer_strimmer_shrinkage",
]
