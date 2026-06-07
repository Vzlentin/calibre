"""Point-forecast reconciliation seam (architecture §9).

Makes point forecasts coherent across a cross-sectional hierarchy. The
pipeline-facing surface is the :class:`Reconciler` Protocol plus the strategy
registry; concrete strategies are resolved by name and default to a no-op
pass-through.
"""

from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.nixtla_adapter import NixtlaReconciler
from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler, ReconciliationContext
from calibre.reconciliation.registry import (
    available_reconcilers,
    resolve_reconciler,
)
from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    SummingMatrix,
    build_summing_matrix,
)

__all__ = [
    "TOTAL_LABEL",
    "NixtlaReconciler",
    "NoOpReconciler",
    "ReconciliationContext",
    "Reconciler",
    "SummingMatrix",
    "VectorReconciler",
    "available_reconcilers",
    "build_summing_matrix",
    "resolve_reconciler",
]
