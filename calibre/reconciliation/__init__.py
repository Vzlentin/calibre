"""Hierarchy reconciliation and fused interval seams (architecture §9).

Makes point forecasts coherent across a cross-sectional hierarchy. The
pipeline-facing surface is the :class:`Reconciler` Protocol plus the strategy
registry; concrete strategies are resolved by name and default to a no-op
pass-through. Interim hierarchical conformal intervals use a separate fused
phase because they own both hierarchy and interval output for that run.
"""

from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.bottom_up import BottomUpReconciler
from calibre.reconciliation.hierarchical_intervals import (
    HierarchicalIntervalContext,
    HierarchicalIntervalOptions,
    HierarchicalIntervalPhase,
    NixtlaHierarchicalIntervalPhase,
)
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
    "BottomUpReconciler",
    "NixtlaReconciler",
    "NixtlaHierarchicalIntervalPhase",
    "NoOpReconciler",
    "HierarchicalIntervalContext",
    "HierarchicalIntervalOptions",
    "HierarchicalIntervalPhase",
    "ReconciliationContext",
    "Reconciler",
    "SummingMatrix",
    "VectorReconciler",
    "available_reconcilers",
    "build_summing_matrix",
    "resolve_reconciler",
]
