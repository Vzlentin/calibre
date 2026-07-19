"""Expose point-reconciliation contracts, matrices, and built-in strategies."""

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
from newcalibre.reconcile.nixtla import (
    MINT_SHRINK,
    MINT_SHRINK_DECLARATION,
    SPARSE_SOLVER_TOLERANCE,
    WLS_STRUCT,
    WLS_STRUCT_DECLARATION,
    WLS_VAR,
    WLS_VAR_DECLARATION,
    NixtlaLayout,
    ProjectionConvergenceError,
    ProjectionReconciler,
    VarianceWeights,
    build_mint_shrink,
    build_wls_struct,
    build_wls_var,
    derive_variance_weights,
)
from newcalibre.reconcile.preflight import (
    DENSE_PERMITTED,
    DENSE_WORKSPACE_CEILING_BYTES,
    REJECTED_AT_SCALE,
    SPARSE_REQUIRED,
    ProjectionMetadata,
    ProjectionPreflight,
    WorkspaceComponent,
    metadata_from_hierarchy,
    preflight_projection,
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
from newcalibre.reconcile.tolerance import (
    CoherenceToleranceError,
    coherence_tolerance,
    covariance_estimator_tolerance,
)

_BUILTIN_STRATEGIES = ReconcilerRegistry()
_BUILTIN_STRATEGIES.register(NONE_DECLARATION, build_none)
_BUILTIN_STRATEGIES.register(BOTTOM_UP_DECLARATION, build_bottom_up)
_BUILTIN_STRATEGIES.register(WLS_STRUCT_DECLARATION, build_wls_struct)
_BUILTIN_STRATEGIES.register(WLS_VAR_DECLARATION, build_wls_var)
_BUILTIN_STRATEGIES.register(MINT_SHRINK_DECLARATION, build_mint_shrink)


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
    "DENSE_PERMITTED",
    "DENSE_WORKSPACE_CEILING_BYTES",
    "MINT_SHRINK",
    "NONE",
    "REJECTED_AT_SCALE",
    "SPARSE_REQUIRED",
    "SPARSE_SOLVER_TOLERANCE",
    "WLS_STRUCT",
    "WLS_VAR",
    "BOTTOM_UP_DECLARATION",
    "MINT_SHRINK_DECLARATION",
    "NONE_DECLARATION",
    "WLS_STRUCT_DECLARATION",
    "WLS_VAR_DECLARATION",
    "BottomUpReconciler",
    "CoherenceToleranceError",
    "DenseSummingMatrix",
    "MatrixCapability",
    "NixtlaLayout",
    "NoReconciliation",
    "ProjectionConvergenceError",
    "ProjectionMetadata",
    "ProjectionPreflight",
    "ProjectionReconciler",
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
    "VarianceWeights",
    "WorkspaceComponent",
    "available_strategies",
    "build_bottom_up",
    "build_dense_summing_matrix",
    "build_mint_shrink",
    "build_none",
    "build_sparse_summing_matrix",
    "build_wls_struct",
    "build_wls_var",
    "coherence_tolerance",
    "covariance_estimator_tolerance",
    "derive_variance_weights",
    "metadata_from_hierarchy",
    "preflight_projection",
    "resolve_strategy",
    "strategy_declaration",
]
