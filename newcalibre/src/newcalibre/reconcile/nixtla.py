"""Adapt point projections to pinned hierarchicalforecast matrix conventions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from hierarchicalforecast.methods import MinTrace, MinTraceSparse
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from newcalibre.domain import (
    ACTUAL_VALUE,
    FITTED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    HierarchyIndex,
)
from newcalibre.reconcile.apply import (
    ReconciledValues,
    ReconciliationError,
    _ProjectionSection,
    apply_projection,
)
from newcalibre.reconcile.preflight import (
    DENSE_WORKSPACE_CEILING_BYTES,
    REJECTED_AT_SCALE,
    SPARSE_REQUIRED,
    ProjectionMetadata,
    ProjectionPreflight,
    preflight_projection,
)
from newcalibre.reconcile.protocol import (
    MatrixCapability,
    ReconcilerDeclaration,
    ReconciliationContext,
    ReconciliationInputFamily,
)
from newcalibre.reconcile.summing import (
    DenseSummingMatrix,
    SparseSummingMatrix,
    SummingMatrix,
    build_dense_summing_matrix,
    build_sparse_summing_matrix,
)
from newcalibre.reconcile.tolerance import coherence_tolerance

WLS_STRUCT: Final = "wls_struct"
WLS_VAR: Final = "wls_var"
MINT_SHRINK: Final = "mint_shrink"
SPARSE_SOLVER_TOLERANCE: Final = 1e-5

WLS_STRUCT_DECLARATION: Final = ReconcilerDeclaration(
    name=WLS_STRUCT,
    input_family=ReconciliationInputFamily.PROJECTION,
    requires_fitted_values=False,
    matrix_capability=MatrixCapability.SPARSE_CAPABLE,
)
WLS_VAR_DECLARATION: Final = ReconcilerDeclaration(
    name=WLS_VAR,
    input_family=ReconciliationInputFamily.PROJECTION,
    requires_fitted_values=True,
    matrix_capability=MatrixCapability.SPARSE_CAPABLE,
)
MINT_SHRINK_DECLARATION: Final = ReconcilerDeclaration(
    name=MINT_SHRINK,
    input_family=ReconciliationInputFamily.PROJECTION,
    requires_fitted_values=True,
    matrix_capability=MatrixCapability.DENSE_ONLY,
)
_PROJECTION_DECLARATIONS = (
    WLS_STRUCT_DECLARATION,
    WLS_VAR_DECLARATION,
    MINT_SHRINK_DECLARATION,
)


class ProjectionConvergenceError(RuntimeError):
    """Report a non-converged sparse point-projection solve."""


class _NormalOperator(sparse_linalg.LinearOperator):
    def __init__(
        self,
        matrix: sparse.csr_matrix,
        weighted_transpose: sparse.csr_matrix,
    ) -> None:
        self._matrix = matrix
        self._weighted_transpose = weighted_transpose
        super().__init__(dtype=np.dtype(np.float64), shape=(matrix.shape[1], matrix.shape[1]))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._weighted_transpose @ (self._matrix @ x))


class _CheckedProjectionOperator(sparse_linalg.LinearOperator):
    def __init__(
        self,
        matrix: sparse.csr_matrix,
        weighted_transpose: sparse.csr_matrix,
        *,
        maxiter: int | None,
    ) -> None:
        self._matrix = matrix
        self._weighted_transpose = weighted_transpose
        self._maxiter = maxiter
        super().__init__(dtype=np.dtype(np.float64), shape=(matrix.shape[1], matrix.shape[0]))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        right_hand = np.asarray(self._weighted_transpose @ x)
        solution, exit_code = sparse_linalg.bicgstab(
            _NormalOperator(self._matrix, self._weighted_transpose),
            right_hand,
            rtol=SPARSE_SOLVER_TOLERANCE,
            atol=SPARSE_SOLVER_TOLERANCE,
            maxiter=self._maxiter,
        )
        if exit_code != 0:
            reason = (
                "no convergence within maxiter" if exit_code > 0 else "breakdown or illegal input"
            )
            raise ProjectionConvergenceError(
                f"bicgstab solve failed: exit_code={exit_code} ({reason}); "
                "the coherent output shape cannot expose non-convergence"
            )
        return solution


@dataclass(frozen=True, slots=True)
class VarianceWeights:
    """Carry floored diagonal residual weights and their derived floor."""

    values: np.ndarray
    floor: float


@dataclass(frozen=True, slots=True)
class NixtlaLayout:
    """Convert identity-first project rows to aggregate-first Nixtla rows."""

    permutation: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    n_bottom: int
    n_nodes: int

    @classmethod
    def from_matrix(cls, matrix: SummingMatrix) -> NixtlaLayout:
        """Derive the row permutation from one validated summing matrix."""
        if not isinstance(matrix, SummingMatrix):
            raise TypeError("Nixtla layout requires a SummingMatrix")
        permutation = (*range(matrix.n_bottom, matrix.n_nodes), *range(matrix.n_bottom))
        inverse = tuple(int(index) for index in np.argsort(permutation))
        return cls(
            permutation=tuple(permutation),
            inverse_permutation=inverse,
            n_bottom=matrix.n_bottom,
            n_nodes=matrix.n_nodes,
        )

    def to_nixtla_vector(self, values: np.ndarray) -> np.ndarray:
        """Permute one project-order node vector to Nixtla order."""
        vector = _layout_vector(values, expected=self.n_nodes)
        return vector[list(self.permutation)]

    def to_project_vector(self, values: np.ndarray) -> np.ndarray:
        """Restore one Nixtla-order node vector to project order."""
        vector = _layout_vector(values, expected=self.n_nodes)
        return vector[list(self.inverse_permutation)]

    def dense_matrix(self, matrix: DenseSummingMatrix) -> np.ndarray:
        """Return an aggregate-first dense summing matrix."""
        if not isinstance(matrix, DenseSummingMatrix):
            raise TypeError("dense Nixtla layout requires DenseSummingMatrix")
        self._require_shape(matrix)
        return matrix.to_dense()[list(self.permutation)]

    def sparse_matrix(self, matrix: SparseSummingMatrix) -> sparse.csr_array:
        """Return aggregate-first CSR rows without a dense intermediate."""
        if not isinstance(matrix, SparseSummingMatrix):
            raise TypeError("sparse Nixtla layout requires SparseSummingMatrix")
        self._require_shape(matrix)
        return sparse.csr_array(matrix.to_csr()[list(self.permutation)])

    def _require_shape(self, matrix: SummingMatrix) -> None:
        if matrix.n_bottom != self.n_bottom or matrix.n_nodes != self.n_nodes:
            raise ValueError("Nixtla layout and summing matrix shapes do not match")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ProjectionReconciler:
    """Run one selected Nixtla point projection behind Calibre-owned guards."""

    _declaration: ReconcilerDeclaration
    dense_workspace_ceiling_bytes: int = DENSE_WORKSPACE_CEILING_BYTES
    bicgstab_maxiter: int | None = None

    def __post_init__(self) -> None:
        if self._declaration not in _PROJECTION_DECLARATIONS:
            raise ValueError("projection reconciler requires a built-in projection declaration")
        if (
            not isinstance(self.dense_workspace_ceiling_bytes, Integral)
            or isinstance(self.dense_workspace_ceiling_bytes, bool)
            or self.dense_workspace_ceiling_bytes < 0
        ):
            raise ValueError("dense workspace ceiling bytes must be a non-negative integer")
        if self.bicgstab_maxiter is not None and (
            not isinstance(self.bicgstab_maxiter, Integral)
            or isinstance(self.bicgstab_maxiter, bool)
            or self.bicgstab_maxiter < 1
        ):
            raise ValueError("bicgstab maxiter must be a positive integer or None")

    @property
    def declaration(self) -> ReconcilerDeclaration:
        """Return immutable projection run-preparation requirements."""
        return self._declaration

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        """Reconcile every complete node cross-section in place by row key."""
        model_frames: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
        section_matrices: dict[
            tuple[bool, tuple[str, ...]], DenseSummingMatrix | SparseSummingMatrix
        ] = {}
        aligned_fitted: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]] = {}
        variance_weights: dict[tuple[str, tuple[str, ...]], VarianceWeights] = {}

        def kernel(
            section: _ProjectionSection,
            hierarchy: HierarchyIndex,
            context: ReconciliationContext,
            base_forecast: np.ndarray,
        ) -> ReconciledValues:
            try:
                preflight = _preflight_section(
                    self.declaration.name,
                    section,
                    hierarchy,
                    residual_periods=0,
                    ceiling_bytes=self.dense_workspace_ceiling_bytes,
                )
                _require_admitted(
                    preflight,
                    strategy=self.declaration.name,
                    section=section,
                )
                model_frame: pd.DataFrame | None = None
                if self.declaration.requires_fitted_values:
                    fitted_values = context.fitted_values
                    if fitted_values is None:
                        raise ReconciliationError(
                            f"strategy {self.declaration.name!r} requires fitted values"
                        )
                    model = section.identity[0]
                    residual_periods = fitted_values.residual_periods_for(
                        model,
                        section.node_labels,
                    )
                    if residual_periods is None:
                        raise ReconciliationError(
                            f"{section.description} has no fitted values for model {model!r}"
                        )
                    preflight = _preflight_section(
                        self.declaration.name,
                        section,
                        hierarchy,
                        residual_periods=residual_periods,
                        ceiling_bytes=self.dense_workspace_ceiling_bytes,
                    )
                    _require_admitted(
                        preflight,
                        strategy=self.declaration.name,
                        section=section,
                    )
                    residual_key = (model, section.node_labels)
                    if residual_key not in model_frames:
                        model_frames[residual_key] = fitted_values.select_model_series(
                            model,
                            section.node_labels,
                        )
                    model_frame = model_frames[residual_key]
                return self._reconcile_section(
                    section,
                    hierarchy,
                    base_forecast,
                    preflight=preflight,
                    model_frame=model_frame,
                    section_matrices=section_matrices,
                    aligned_fitted=aligned_fitted,
                    variance_weights=variance_weights,
                )
            except ReconciliationError:
                raise
            except Exception as error:
                raise ReconciliationError(
                    f"strategy {self.declaration.name!r} {section.description} failed: {error}"
                ) from error

        return apply_projection(
            frame,
            hierarchy,
            context,
            declaration=self.declaration,
            kernel=kernel,
        )

    def _reconcile_section(
        self,
        section: _ProjectionSection,
        hierarchy: HierarchyIndex,
        base_forecast: np.ndarray,
        *,
        preflight: ProjectionPreflight,
        model_frame: pd.DataFrame | None,
        section_matrices: dict[
            tuple[bool, tuple[str, ...]], DenseSummingMatrix | SparseSummingMatrix
        ],
        aligned_fitted: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]],
        variance_weights: dict[tuple[str, tuple[str, ...]], VarianceWeights],
    ) -> ReconciledValues:
        use_sparse = self.declaration.name == WLS_VAR or preflight.decision == SPARSE_REQUIRED
        matrix_key = (use_sparse, section.bottom_ids)
        matrix = section_matrices.get(matrix_key)
        if matrix is None:
            matrix = (
                build_sparse_summing_matrix(hierarchy, bottom_ids=section.bottom_ids)
                if use_sparse
                else build_dense_summing_matrix(hierarchy, bottom_ids=section.bottom_ids)
            )
            section_matrices[matrix_key] = matrix
        if matrix.node_labels != section.node_labels:
            raise ReconciliationError(
                f"{section.description} projection nodes drifted from the summing matrix"
            )

        layout = NixtlaLayout.from_matrix(matrix)
        nixtla_forecast = layout.to_nixtla_vector(base_forecast)[:, None]
        actuals: np.ndarray | None = None
        fitted: np.ndarray | None = None
        variance: VarianceWeights | None = None
        if model_frame is not None:
            residual_key = (section.identity[0], section.node_labels)
            aligned = aligned_fitted.get(residual_key)
            if aligned is None:
                aligned = _aligned_fitted_matrices(
                    model_frame,
                    node_labels=section.node_labels,
                    section=section,
                )
                aligned_fitted[residual_key] = aligned
            project_actuals, project_fitted = aligned
            if self.declaration.name == WLS_VAR:
                project_variance = variance_weights.get(residual_key)
                if project_variance is None:
                    project_variance = derive_variance_weights(project_actuals - project_fitted)
                    variance_weights[residual_key] = project_variance
                variance = VarianceWeights(
                    values=project_variance.values[list(layout.permutation)],
                    floor=project_variance.floor,
                )
            else:
                actuals = project_actuals[list(layout.permutation)]
                fitted = project_fitted[list(layout.permutation)]

        if use_sparse:
            if not isinstance(matrix, SparseSummingMatrix):
                raise ReconciliationError("sparse projection requires a sparse summing matrix")
            method = _CheckedMinTraceSparse(
                self.declaration.name,
                weight_diagonal=None if variance is None else variance.values,
                bicgstab_maxiter=self.bicgstab_maxiter,
            )
            nixtla_matrix = layout.sparse_matrix(matrix)
        else:
            if not isinstance(matrix, DenseSummingMatrix):
                raise ReconciliationError("dense projection requires a dense summing matrix")
            method = MinTrace(
                method=self.declaration.name,
                mint_shr_ridge=0.0,
                num_threads=1,
            )
            nixtla_matrix = layout.dense_matrix(matrix)

        # The pinned sparse subclass accepts CSR, but its inherited annotation
        # still declares only ndarray. Keep the vendor mismatch at this boundary.
        result = cast(Any, method).fit_predict(
            S=nixtla_matrix,
            y_hat=nixtla_forecast,
            y_insample=actuals,
            y_hat_insample=fitted,
        )
        reconciled = layout.to_project_vector(np.asarray(result["mean"])[:, 0])
        coherent, coherence_bound, support_bound = _coherent_projection_bound(
            matrix,
            reconciled,
            use_sparse=use_sparse,
        )
        _verify_coherence(reconciled, coherent, coherence_bound, section=section)
        return ReconciledValues(reconciled, support_bound)


def _preflight_section(
    strategy: str,
    section: _ProjectionSection,
    hierarchy: HierarchyIndex,
    *,
    residual_periods: int,
    ceiling_bytes: int,
) -> ProjectionPreflight:
    metadata = ProjectionMetadata(
        n_bottom=len(section.bottom_ids),
        n_nodes=len(section.node_labels),
        n_attributes=len(hierarchy.attribute_names),
        residual_periods=residual_periods,
    )
    return preflight_projection(strategy, metadata, ceiling_bytes=ceiling_bytes)


def _require_admitted(
    preflight: ProjectionPreflight,
    *,
    strategy: str,
    section: _ProjectionSection,
) -> None:
    if preflight.decision == REJECTED_AT_SCALE:
        raise ReconciliationError(
            f"strategy {strategy!r} {section.description}: {preflight.reason}"
        )


def derive_variance_weights(residuals: np.ndarray) -> VarianceWeights:
    """Derive ddof-one variances with the selected scale-relative floor."""
    try:
        array = np.asarray(residuals, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReconciliationError("residual history must contain real numeric values") from error
    if array.ndim != 2:
        raise ReconciliationError("residual history must be a two-dimensional node matrix")
    if array.shape[1] < 2:
        raise ReconciliationError("residual history requires at least two aligned periods")
    if not np.all(np.isfinite(array)):
        raise ReconciliationError("residual history must be finite")
    variances = np.var(array, axis=1, ddof=1)
    floor = float(variances.max()) * array.shape[1] * np.finfo(np.float64).eps
    if not np.isfinite(floor) or floor <= 0.0:
        raise ReconciliationError(
            "residual variance floor is not positive; at least one node must vary"
        )
    values = np.maximum(variances, floor)
    values.setflags(write=False)
    return VarianceWeights(values=values, floor=floor)


def build_wls_struct(
    *,
    dense_workspace_ceiling_bytes: int = DENSE_WORKSPACE_CEILING_BYTES,
    bicgstab_maxiter: int | None = None,
) -> ProjectionReconciler:
    """Build a structural-weights projection adapter."""
    return ProjectionReconciler(
        WLS_STRUCT_DECLARATION,
        dense_workspace_ceiling_bytes=dense_workspace_ceiling_bytes,
        bicgstab_maxiter=bicgstab_maxiter,
    )


def build_wls_var(
    *,
    dense_workspace_ceiling_bytes: int = DENSE_WORKSPACE_CEILING_BYTES,
    bicgstab_maxiter: int | None = None,
) -> ProjectionReconciler:
    """Build a residual-variance projection adapter."""
    return ProjectionReconciler(
        WLS_VAR_DECLARATION,
        dense_workspace_ceiling_bytes=dense_workspace_ceiling_bytes,
        bicgstab_maxiter=bicgstab_maxiter,
    )


def build_mint_shrink(
    *,
    dense_workspace_ceiling_bytes: int = DENSE_WORKSPACE_CEILING_BYTES,
) -> ProjectionReconciler:
    """Build a ceiling-gated shrunk-covariance projection adapter."""
    return ProjectionReconciler(
        MINT_SHRINK_DECLARATION,
        dense_workspace_ceiling_bytes=dense_workspace_ceiling_bytes,
    )


class _CheckedMinTraceSparse(MinTraceSparse):
    """Surface bicgstab status while mirroring the pinned upstream P-action."""

    def __init__(
        self,
        method: str,
        *,
        weight_diagonal: np.ndarray | None,
        bicgstab_maxiter: int | None,
    ) -> None:
        super().__init__(method=method, nonnegative=False, num_threads=1)
        self._weight_diagonal = weight_diagonal
        self._bicgstab_maxiter = bicgstab_maxiter

    def _get_PW_matrices(
        self,
        S,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ):
        del y_insample, y_hat_insample
        matrix = sparse.csr_matrix(S)
        n_nodes, n_bottom = matrix.shape
        if self.method == WLS_STRUCT:
            weights = np.asarray(matrix @ np.ones(n_bottom), dtype=np.float64)
        elif self.method == WLS_VAR and self._weight_diagonal is not None:
            weights = np.asarray(self._weight_diagonal, dtype=np.float64)
        else:
            raise ValueError(f"unsupported checked sparse method {self.method!r}")
        if weights.shape != (n_nodes,) or not np.all(np.isfinite(weights)):
            raise ValueError("sparse projection weights must be one finite value per node")
        if np.any(weights <= 0.0):
            raise ValueError("sparse projection weights must be strictly positive")

        precision = sparse.spdiags(np.reciprocal(weights), 0, n_nodes, n_nodes)
        weighted_transpose = sparse.csr_matrix(matrix.T @ precision)
        projection = _CheckedProjectionOperator(
            matrix,
            weighted_transpose,
            maxiter=self._bicgstab_maxiter,
        )
        weight_matrix = sparse.spdiags(weights, 0, n_nodes, n_nodes)
        return projection, weight_matrix


def _aligned_fitted_matrices(
    model_frame: pd.DataFrame,
    *,
    node_labels: tuple[str, ...],
    section: _ProjectionSection,
) -> tuple[np.ndarray, np.ndarray]:
    applicable = model_frame.loc[model_frame[SERIES_KEY].isin(node_labels)]
    present = set(applicable[SERIES_KEY])
    missing = sorted(set(node_labels) - present, key=str.encode)
    if missing:
        raise ReconciliationError(f"{section.description} is missing fitted-value nodes: {missing}")
    if applicable.duplicated(subset=[SERIES_KEY, TIMESTAMP]).any():
        raise ReconciliationError(f"{section.description} has duplicate fitted-value keys")

    timestamps: tuple[pd.Timestamp, ...] | None = None
    by_node: dict[str, pd.DataFrame] = {}
    for label in node_labels:
        rows = applicable.loc[applicable[SERIES_KEY] == label].sort_values(TIMESTAMP)
        node_timestamps = tuple(pd.Timestamp(value) for value in rows[TIMESTAMP])
        if timestamps is None:
            timestamps = node_timestamps
        elif node_timestamps != timestamps:
            raise ReconciliationError(
                f"{section.description} fitted-value timestamp sets are misaligned"
            )
        by_node[label] = rows
    if timestamps is None or len(timestamps) < 2:
        raise ReconciliationError(
            f"{section.description} fitted values require at least two aligned periods"
        )

    actual = np.vstack(
        [by_node[label][ACTUAL_VALUE].to_numpy(dtype=np.float64) for label in node_labels]
    )
    fitted = np.vstack(
        [by_node[label][FITTED_VALUE].to_numpy(dtype=np.float64) for label in node_labels]
    )
    residuals = actual - fitted
    if not np.all(np.isfinite(residuals)):
        raise ReconciliationError(f"{section.description} residual history must be finite")
    return actual, fitted


def _verify_coherence(
    reconciled: np.ndarray,
    coherent: np.ndarray,
    bound: float,
    *,
    section: _ProjectionSection,
) -> None:
    if not np.allclose(reconciled, coherent, rtol=0.0, atol=bound):
        raise ReconciliationError(
            f"{section.description} failed the derived summing-matrix coherence check"
        )


def _coherent_projection_bound(
    matrix: DenseSummingMatrix | SparseSummingMatrix,
    reconciled: np.ndarray,
    *,
    use_sparse: bool,
) -> tuple[np.ndarray, float, float]:
    coherent = matrix.matvec(reconciled[: matrix.n_bottom])
    magnitude = float(np.max(np.abs(np.concatenate((reconciled, coherent)))))
    coherence_bound = coherence_tolerance(
        reduction_width=matrix.reduction_width,
        vector_magnitude=magnitude,
        solver_tolerance=SPARSE_SOLVER_TOLERANCE if use_sparse else None,
    )
    support_bound = _support_canonicalization_bound(reconciled)
    return coherent, coherence_bound, support_bound


def _support_canonicalization_bound(reconciled: np.ndarray) -> float:
    magnitude = float(np.max(np.abs(reconciled))) if reconciled.size else 0.0
    return float(8.0 * np.finfo(np.float64).eps * max(magnitude, 1.0))


def _layout_vector(values: np.ndarray, *, expected: int) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Nixtla layout vector must contain real numeric values") from error
    if vector.shape != (expected,):
        raise ValueError(f"Nixtla layout vector must have shape ({expected},)")
    return vector


__all__ = [
    "MINT_SHRINK",
    "MINT_SHRINK_DECLARATION",
    "SPARSE_SOLVER_TOLERANCE",
    "WLS_STRUCT",
    "WLS_STRUCT_DECLARATION",
    "WLS_VAR",
    "WLS_VAR_DECLARATION",
    "NixtlaLayout",
    "ProjectionConvergenceError",
    "ProjectionReconciler",
    "VarianceWeights",
    "build_mint_shrink",
    "build_wls_struct",
    "build_wls_var",
    "derive_variance_weights",
]
