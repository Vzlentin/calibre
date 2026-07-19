"""Expose point-reconciliation contracts, matrices, and native strategies."""

from newcalibre.reconcile.apply import ReconciliationError
from newcalibre.reconcile.native import (
    BOTTOM_UP,
    BOTTOM_UP_DECLARATION,
    NONE,
    NONE_DECLARATION,
    BottomUpReconciler,
    NoReconciliation,
    build_bottom_up,
    build_none,
)
from newcalibre.reconcile.protocol import (
    MatrixCapability,
    Reconciler,
    ReconcilerDeclaration,
    ReconciliationContext,
    ReconciliationInputFamily,
)
from newcalibre.reconcile.registry import ReconcilerRegistry, ReconciliationRegistryError
from newcalibre.reconcile.summing import (
    DenseSummingMatrix,
    SparseSummingMatrix,
    SummingMatrix,
    SummingMatrixError,
    build_dense_summing_matrix,
    build_sparse_summing_matrix,
)
from newcalibre.reconcile.tolerance import CoherenceToleranceError, coherence_tolerance

_BUILTIN_STRATEGIES = ReconcilerRegistry()
_BUILTIN_STRATEGIES.register(NONE_DECLARATION, build_none)
_BUILTIN_STRATEGIES.register(BOTTOM_UP_DECLARATION, build_bottom_up)


def available_strategies() -> tuple[str, ...]:
    """Return canonical built-in reconciliation names in deterministic order."""
    return _BUILTIN_STRATEGIES.available_strategies


def strategy_declaration(name: str) -> ReconcilerDeclaration:
    """Return one normalized built-in strategy declaration."""
    return _BUILTIN_STRATEGIES.declaration(name)


def resolve_strategy(name: str) -> Reconciler:
    """Resolve one normalized built-in strategy to a fresh instance."""
    return _BUILTIN_STRATEGIES.resolve(name)


__all__ = [
    "BOTTOM_UP",
    "NONE",
    "BOTTOM_UP_DECLARATION",
    "NONE_DECLARATION",
    "BottomUpReconciler",
    "CoherenceToleranceError",
    "DenseSummingMatrix",
    "MatrixCapability",
    "NoReconciliation",
    "Reconciler",
    "ReconcilerDeclaration",
    "ReconcilerRegistry",
    "ReconciliationContext",
    "ReconciliationError",
    "ReconciliationInputFamily",
    "ReconciliationRegistryError",
    "SparseSummingMatrix",
    "SummingMatrix",
    "SummingMatrixError",
    "available_strategies",
    "build_bottom_up",
    "build_dense_summing_matrix",
    "build_none",
    "build_sparse_summing_matrix",
    "coherence_tolerance",
    "resolve_strategy",
    "strategy_declaration",
]
