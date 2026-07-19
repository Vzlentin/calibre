"""Estimate projection workspaces from hierarchy metadata before allocation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

from newcalibre.domain import HierarchyIndex

DENSE_WORKSPACE_CEILING_BYTES: Final = 1 << 30
DENSE_PERMITTED: Final = "dense_permitted"
SPARSE_REQUIRED: Final = "sparse_required"
REJECTED_AT_SCALE: Final = "rejected_at_scale"

_INT32_BYTES = 4
_FACTOR_TEMPORARIES = 1
_ITERATIVE_VECTOR_COUNT_BOTTOM = 4
_ITERATIVE_VECTOR_COUNT_NODES = 2
_RESIDUAL_WIDE_MATRICES = 2


@dataclass(frozen=True, slots=True)
class ProjectionMetadata:
    """Carry the scalar lattice facts used by workspace estimation."""

    n_bottom: int
    n_nodes: int
    n_attributes: int
    residual_periods: int
    dtype_bytes: int = 8

    def __post_init__(self) -> None:
        for name in ("n_bottom", "n_nodes", "n_attributes", "residual_periods", "dtype_bytes"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"projection metadata {name} must be an integer")
        if self.n_bottom < 1:
            raise ValueError("projection metadata requires at least one bottom series")
        if self.n_nodes < self.n_bottom:
            raise ValueError("projection node count cannot be smaller than bottom count")
        if self.n_attributes < 0:
            raise ValueError("projection attribute count cannot be negative")
        if self.residual_periods < 0:
            raise ValueError("projection residual periods cannot be negative")
        if self.dtype_bytes < 1:
            raise ValueError("projection dtype bytes must be positive")


@dataclass(frozen=True, slots=True)
class WorkspaceComponent:
    """Name one strategy-owned workspace allocation in bytes."""

    name: str
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("workspace component name must be a non-empty string")
        if not isinstance(self.bytes, Integral) or isinstance(self.bytes, bool) or self.bytes < 0:
            raise ValueError("workspace component bytes must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ProjectionPreflight:
    """Report an itemized dense/sparse estimate and representation decision."""

    strategy: str
    decision: str
    reason: str
    dense_components: tuple[WorkspaceComponent, ...]
    sparse_components: tuple[WorkspaceComponent, ...] | None
    dense_workspace_bytes: int
    sparse_workspace_bytes: int | None
    ceiling_bytes: int


@dataclass(frozen=True, slots=True)
class _Formulation:
    requires_residuals: bool
    full_covariance: bool
    sparse_capable: bool


_FORMULATIONS: Final = {
    "wls_struct": _Formulation(
        requires_residuals=False,
        full_covariance=False,
        sparse_capable=True,
    ),
    "wls_var": _Formulation(
        requires_residuals=True,
        full_covariance=False,
        sparse_capable=True,
    ),
    "mint_shrink": _Formulation(
        requires_residuals=True,
        full_covariance=True,
        sparse_capable=False,
    ),
}


def metadata_from_hierarchy(
    hierarchy: HierarchyIndex,
    *,
    residual_periods: int,
    dtype_bytes: int = 8,
) -> ProjectionMetadata:
    """Derive workspace dimensions only from canonical hierarchy metadata."""
    if not isinstance(hierarchy, HierarchyIndex):
        raise TypeError("projection metadata requires a HierarchyIndex")
    return ProjectionMetadata(
        n_bottom=len(hierarchy.bottom_series),
        n_nodes=len(hierarchy.node_labels),
        n_attributes=len(hierarchy.attribute_names),
        residual_periods=residual_periods,
        dtype_bytes=dtype_bytes,
    )


def preflight_projection(
    strategy: str,
    metadata: ProjectionMetadata,
    *,
    ceiling_bytes: int = DENSE_WORKSPACE_CEILING_BYTES,
) -> ProjectionPreflight:
    """Select dense, sparse, or rejection from scalar metadata only."""
    if not isinstance(strategy, str):
        raise TypeError("projection strategy must be a string")
    normalized = strategy.strip().casefold()
    try:
        formulation = _FORMULATIONS[normalized]
    except KeyError as error:
        available = ", ".join(sorted(_FORMULATIONS))
        raise ValueError(
            f"unknown projection strategy {normalized!r}; available strategies: {available}"
        ) from error
    if not isinstance(metadata, ProjectionMetadata):
        raise TypeError("projection preflight metadata must be ProjectionMetadata")
    if (
        not isinstance(ceiling_bytes, Integral)
        or isinstance(ceiling_bytes, bool)
        or ceiling_bytes < 0
    ):
        raise ValueError("dense workspace ceiling bytes must be a non-negative integer")

    dense = _dense_components(metadata, formulation=formulation)
    dense_total = sum(component.bytes for component in dense)
    sparse = _sparse_components(metadata, formulation=formulation)
    sparse_total = None if sparse is None else sum(component.bytes for component in sparse)

    if dense_total <= ceiling_bytes:
        decision = DENSE_PERMITTED
        reason = f"dense workspace {dense_total} B is within ceiling {ceiling_bytes} B"
    elif formulation.sparse_capable:
        decision = SPARSE_REQUIRED
        reason = (
            f"dense workspace {dense_total} B exceeds ceiling {ceiling_bytes} B; "
            "the sparse path must be used before any matrix allocation"
        )
    else:
        decision = REJECTED_AT_SCALE
        itemized = ", ".join(f"{item.name}={item.bytes} B" for item in dense)
        reason = (
            f"strategy {normalized!r} is dense-only and needs {dense_total} B, exceeding "
            f"ceiling {ceiling_bytes} B ({itemized}); reject this run before allocation; "
            "use wls_var or wls_struct as scalable alternatives"
        )

    return ProjectionPreflight(
        strategy=normalized,
        decision=decision,
        reason=reason,
        dense_components=dense,
        sparse_components=sparse,
        dense_workspace_bytes=dense_total,
        sparse_workspace_bytes=sparse_total,
        ceiling_bytes=int(ceiling_bytes),
    )


def _dense_components(
    metadata: ProjectionMetadata,
    *,
    formulation: _Formulation,
) -> tuple[WorkspaceComponent, ...]:
    items = [
        WorkspaceComponent(
            "summing_matrix_dense",
            metadata.n_nodes * metadata.n_bottom * metadata.dtype_bytes,
        )
    ]
    if formulation.full_covariance:
        items.append(
            WorkspaceComponent(
                "covariance_dense",
                metadata.n_nodes * metadata.n_nodes * metadata.dtype_bytes,
            )
        )
    else:
        items.append(
            WorkspaceComponent("weights_diagonal", metadata.n_nodes * metadata.dtype_bytes)
        )
    normal_bytes = metadata.n_bottom * metadata.n_bottom * metadata.dtype_bytes
    items.extend(
        (
            WorkspaceComponent("normal_equations", normal_bytes),
            WorkspaceComponent("factor_temporary", _FACTOR_TEMPORARIES * normal_bytes),
        )
    )
    if formulation.requires_residuals:
        items.append(WorkspaceComponent("residual_wide", _residual_wide_bytes(metadata)))
    return tuple(items)


def _sparse_components(
    metadata: ProjectionMetadata,
    *,
    formulation: _Formulation,
) -> tuple[WorkspaceComponent, ...] | None:
    if not formulation.sparse_capable:
        return None
    nonzero = metadata.n_bottom * (metadata.n_attributes + 2)
    items = [
        WorkspaceComponent(
            "summing_matrix_csr",
            nonzero * metadata.dtype_bytes
            + nonzero * _INT32_BYTES
            + (metadata.n_nodes + 1) * _INT32_BYTES,
        ),
        WorkspaceComponent("weights_diagonal", metadata.n_nodes * metadata.dtype_bytes),
        WorkspaceComponent(
            "iterative_vectors",
            (
                _ITERATIVE_VECTOR_COUNT_NODES * metadata.n_nodes
                + _ITERATIVE_VECTOR_COUNT_BOTTOM * metadata.n_bottom
            )
            * metadata.dtype_bytes,
        ),
    ]
    if formulation.requires_residuals:
        items.append(WorkspaceComponent("residual_wide", _residual_wide_bytes(metadata)))
    return tuple(items)


def _residual_wide_bytes(metadata: ProjectionMetadata) -> int:
    return (
        _RESIDUAL_WIDE_MATRICES
        * metadata.n_nodes
        * metadata.residual_periods
        * metadata.dtype_bytes
    )


__all__ = [
    "DENSE_PERMITTED",
    "DENSE_WORKSPACE_CEILING_BYTES",
    "REJECTED_AT_SCALE",
    "SPARSE_REQUIRED",
    "ProjectionMetadata",
    "ProjectionPreflight",
    "WorkspaceComponent",
    "metadata_from_hierarchy",
    "preflight_projection",
]
