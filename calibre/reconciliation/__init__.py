"""Hierarchy reconciliation seam (architecture §9).

Makes point forecasts coherent across a cross-sectional hierarchy. The
pipeline-facing surface is the :class:`Reconciler` Protocol plus the strategy
registry; concrete strategies are resolved by name and default to a no-op
pass-through.
"""

from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.bottom_up import BottomUpReconciler
from calibre.reconciliation.nixtla_adapter import NixtlaReconciler
from calibre.reconciliation.noop import NoOpReconciler
from calibre.reconciliation.protocols import Reconciler, ReconciliationContext
from calibre.reconciliation.registry import (
    available_reconcilers,
    resolve_reconciler,
)
from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    SparseSummingMatrix,
    SummingMatrix,
    build_summing_matrix,
    sparse_summing_matrix_from_index,
    summing_matrix_from_index,
)

__all__ = [
    "TOTAL_LABEL",
    "BottomUpReconciler",
    "NixtlaReconciler",
    "NoOpReconciler",
    "ReconciliationContext",
    "Reconciler",
    "SparseSummingMatrix",
    "SummingMatrix",
    "VectorReconciler",
    "available_reconcilers",
    "build_summing_matrix",
    "resolve_reconciler",
    "sparse_summing_matrix_from_index",
    "summing_matrix_from_index",
]
