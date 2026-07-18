"""Metadata-only memory preflight and the dense-workspace ceiling.

Computes dimensions/nnz/workspace estimates from run-constant metadata —
``(n_bottom, n_nodes, n_attributes, residual_periods)`` — and decides
dense-permitted / sparse-required / rejected *before any array is allocated*
(`[REC-16]`, `[PRF-2]` O(metadata), `[PRF-21]`). The estimate names every
array and its dtype rather than reporting only ``S.nbytes``, and rejects a
forbidden dense structure instead of materializing it.

Recommended normative default: ``DENSE_WORKSPACE_CEILING_BYTES = 2**30``
(1 GiB = 1/32 of Stage 3's 32 GiB process budget, `[PRF-20]`), a configured
threshold per `[PRF-21]` — deterministic and machine-independent, unlike a
detected-available-memory comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from gate_b_proto.lattice import FLOAT64_BYTES, INT32_BYTES

DENSE_WORKSPACE_CEILING_BYTES = 1 << 30

DENSE_PERMITTED = "dense-permitted"
SPARSE_REQUIRED = "sparse-required"
REJECTED = "rejected"

# Factorization of the dense normal matrix is assumed to need one extra
# n_bottom^2 temporary beyond the normal matrix itself.
_FACTOR_TEMPORARIES = 1
# bicgstab/CG hold a handful of n_bottom- and n_nodes-sized work vectors.
_ITERATIVE_VECTOR_COUNT_BOTTOM = 4
_ITERATIVE_VECTOR_COUNT_NODES = 2
# Residual-requiring formulations widen the fitted-values sidecar into two
# aligned (n_nodes, T) matrices: actuals and fitted values.
_RESIDUAL_WIDE_MATRICES = 2


@dataclass(frozen=True, slots=True)
class LatticeMetadata:
    """Carry the run-constant facts the preflight estimates from — nothing else."""

    n_bottom: int
    n_nodes: int
    n_attributes: int
    residual_periods: int
    dtype_bytes: int = FLOAT64_BYTES


@dataclass(frozen=True, slots=True)
class FormulationSpec:
    """Declare one candidate formulation's estimator and representation reach."""

    name: str
    requires_residuals: bool
    full_covariance: bool
    sparse_capable: bool
    rejected_by_name: str | None


FORMULATIONS: dict[str, FormulationSpec] = {
    "wls_struct": FormulationSpec(
        name="wls_struct",
        requires_residuals=False,
        full_covariance=False,
        sparse_capable=True,
        rejected_by_name=None,
    ),
    "wls_var": FormulationSpec(
        name="wls_var",
        requires_residuals=True,
        full_covariance=False,
        sparse_capable=True,
        rejected_by_name=None,
    ),
    "mint_shrink": FormulationSpec(
        name="mint_shrink",
        requires_residuals=True,
        full_covariance=True,
        sparse_capable=False,
        rejected_by_name=None,
    ),
    "mint_cov": FormulationSpec(
        name="mint_cov",
        requires_residuals=True,
        full_covariance=True,
        sparse_capable=False,
        rejected_by_name=(
            "raw full sample covariance is rank-deficient for T < n_nodes and "
            "ill-conditioned on retail-sized lattices; use wls_var or wls_struct"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class Estimate:
    """Report one formulation's itemized byte estimates and the preflight decision."""

    formulation: str
    decision: str
    reason: str
    dense_components: tuple[tuple[str, int], ...]
    sparse_components: tuple[tuple[str, int], ...] | None
    dense_workspace_bytes: int
    sparse_workspace_bytes: int | None
    ceiling_bytes: int


def summing_matrix_nnz(meta: LatticeMetadata) -> int:
    """Return the exact summing-matrix nonzero count: ``n_bottom * (A + 2)``."""
    return meta.n_bottom * (meta.n_attributes + 2)


def dense_summing_matrix_bytes(meta: LatticeMetadata) -> int:
    """Return the dense summing-matrix bytes: ``n_nodes * n_bottom * dtype``."""
    return meta.n_nodes * meta.n_bottom * meta.dtype_bytes


def csr_summing_matrix_bytes(meta: LatticeMetadata) -> int:
    """Return the csr summing-matrix bytes: float64 data + int32 indices/indptr."""
    nnz = summing_matrix_nnz(meta)
    return nnz * meta.dtype_bytes + nnz * INT32_BYTES + (meta.n_nodes + 1) * INT32_BYTES


def estimate_formulation(
    formulation: str,
    meta: LatticeMetadata,
    *,
    ceiling_bytes: int = DENSE_WORKSPACE_CEILING_BYTES,
) -> Estimate:
    """Estimate one formulation's workspaces from metadata and decide its path.

    Peak assumptions, stated rather than implied: the sparse summing matrix is
    built directly from coordinates (no dense intermediate — an eager
    densify-first build would double the dense term transiently); a dense
    factorization holds one extra ``n_bottom^2`` temporary; an iterative solve
    holds a fixed handful of work vectors; residual-requiring formulations
    hold both wide sidecar matrices for the current origin.
    """
    try:
        spec = FORMULATIONS[formulation]
    except KeyError as error:
        raise ValueError(
            f"unknown formulation {formulation!r}; available: {sorted(FORMULATIONS)}"
        ) from error

    residual_term = (
        _RESIDUAL_WIDE_MATRICES * meta.n_nodes * meta.residual_periods * meta.dtype_bytes
        if spec.requires_residuals
        else 0
    )
    residual_components: list[tuple[str, int]] = (
        [("residual_wide", residual_term)] if spec.requires_residuals else []
    )

    dense_components = [("summing_matrix_dense", dense_summing_matrix_bytes(meta))]
    if spec.full_covariance:
        dense_components.append(
            ("covariance_dense", meta.n_nodes * meta.n_nodes * meta.dtype_bytes)
        )
    else:
        dense_components.append(("weights_diagonal", meta.n_nodes * meta.dtype_bytes))
    dense_components.append(("normal_equations", meta.n_bottom * meta.n_bottom * meta.dtype_bytes))
    dense_components.append(
        (
            "factor_temporary",
            _FACTOR_TEMPORARIES * meta.n_bottom * meta.n_bottom * meta.dtype_bytes,
        )
    )
    dense_components.extend(residual_components)
    dense_workspace = sum(bytes for _, bytes in dense_components)

    sparse_components: list[tuple[str, int]] | None = None
    sparse_workspace: int | None = None
    if spec.sparse_capable:
        sparse_components = [
            ("summing_matrix_csr", csr_summing_matrix_bytes(meta)),
            ("weights_diagonal", meta.n_nodes * meta.dtype_bytes),
            (
                "iterative_vectors",
                (
                    _ITERATIVE_VECTOR_COUNT_NODES * meta.n_nodes
                    + _ITERATIVE_VECTOR_COUNT_BOTTOM * meta.n_bottom
                )
                * meta.dtype_bytes,
            ),
            *residual_components,
        ]
        sparse_workspace = sum(bytes for _, bytes in sparse_components)

    if spec.rejected_by_name is not None:
        decision, reason = REJECTED, f"rejected by name: {spec.rejected_by_name}"
    elif dense_workspace <= ceiling_bytes:
        decision = DENSE_PERMITTED
        reason = f"dense workspace {dense_workspace} B within ceiling {ceiling_bytes} B"
    elif spec.sparse_capable:
        decision = SPARSE_REQUIRED
        reason = (
            f"dense workspace {dense_workspace} B exceeds ceiling {ceiling_bytes} B; "
            "the sparse operator path never materializes it"
        )
    else:
        decision = REJECTED
        reason = (
            f"dense-only formulation needs {dense_workspace} B, exceeding ceiling "
            f"{ceiling_bytes} B, and has no sparse path; use wls_var or wls_struct"
        )

    return Estimate(
        formulation=spec.name,
        decision=decision,
        reason=reason,
        dense_components=tuple(dense_components),
        sparse_components=(tuple(sparse_components) if sparse_components is not None else None),
        dense_workspace_bytes=dense_workspace,
        sparse_workspace_bytes=sparse_workspace,
        ceiling_bytes=ceiling_bytes,
    )


# Full-M5 metadata from checked-in facts only: `[M5-H2]` (30,490 bottom,
# 33,563 nodes), `[M5-H1]`/`[M5-D1]` (five attribute columns), `[M5-D3]`
# (evaluation phase: 1,941 days). Nothing is materialized and nothing is
# downloaded.
M5_METADATA = LatticeMetadata(
    n_bottom=30_490,
    n_nodes=33_563,
    n_attributes=5,
    residual_periods=1_941,
)
