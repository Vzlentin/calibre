"""Crossed-attribute lattice metadata, summing matrices, and the shared derived tolerance.

Prototype machinery for the Gate B reconciliation reaction asset. This module
mirrors the spec's summing-matrix contract (`docs/spec/07-reconciliation.md`
`[REC-11]`..`[REC-13]`) at fixture scale: rows are the bottom identity block,
then one row per distinct value of each attribute column, then a single total
row; columns are the bottom series in sorted order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

TOTAL_LABEL = "__total__"
FLOAT64_BYTES = 8
INT32_BYTES = 4


@dataclass(frozen=True, slots=True)
class Lattice:
    """Hold one crossed-attribute aggregation lattice as validated metadata.

    Attributes:
        bottom_ids: bottom series keys in canonical sorted order (S columns).
        attribute_names: attribute column names in canonical sorted order.
        attribute_values: per attribute name, that attribute's value for every
            bottom series, aligned to ``bottom_ids``.
        node_labels: every node label in row order: bottom ids, then
            ``"<attribute>=<value>"`` per attribute (values sorted), then the
            total label.
    """

    bottom_ids: tuple[str, ...]
    attribute_names: tuple[str, ...]
    attribute_values: dict[str, tuple[str, ...]]
    node_labels: tuple[str, ...]

    @property
    def n_bottom(self) -> int:
        """Return the number of bottom series (S columns)."""
        return len(self.bottom_ids)

    @property
    def n_nodes(self) -> int:
        """Return the number of lattice nodes (S rows)."""
        return len(self.node_labels)

    @property
    def n_attributes(self) -> int:
        """Return the number of attribute columns."""
        return len(self.attribute_names)

    @property
    def nnz(self) -> int:
        """Return the exact nonzero count of S: ``n_bottom * (n_attributes + 2)``.

        Every bottom series appears once in the identity block, once per
        attribute column's membership rows, and once in the total row.
        """
        return self.n_bottom * (self.n_attributes + 2)


def build_lattice(
    bottom_ids: tuple[str, ...], attribute_values: dict[str, tuple[str, ...]]
) -> Lattice:
    """Validate raw lattice facts into a :class:`Lattice` with canonical ordering.

    Raises:
        ValueError: on duplicate bottom ids, an empty lattice, or an attribute
            column whose values do not align with the bottom ids.
    """
    if not bottom_ids:
        raise ValueError("lattice requires at least one bottom series")
    if len(set(bottom_ids)) != len(bottom_ids):
        raise ValueError("bottom ids must be unique")
    if not attribute_values:
        raise ValueError("lattice requires at least one attribute column")
    ordered_bottom = tuple(sorted(bottom_ids))
    attribute_names = tuple(sorted(attribute_values))
    aligned: dict[str, tuple[str, ...]] = {}
    for name in attribute_names:
        values = attribute_values[name]
        if len(values) != len(bottom_ids):
            raise ValueError(
                f"attribute {name!r} has {len(values)} values for {len(bottom_ids)} bottom series"
            )
        if any(not value for value in values):
            raise ValueError(f"attribute {name!r} has a missing value")
        # Re-align values to the sorted bottom order using the submitted order.
        aligned[name] = tuple(value for _, value in sorted(zip(bottom_ids, values, strict=True)))

    labels: list[str] = list(ordered_bottom)
    for name in attribute_names:
        labels.extend(f"{name}={value}" for value in sorted(set(aligned[name])))
    labels.append(TOTAL_LABEL)
    if len(set(labels)) != len(labels):
        raise ValueError("aggregate node labels collide with bottom series keys")
    return Lattice(
        bottom_ids=ordered_bottom,
        attribute_names=attribute_names,
        attribute_values=aligned,
        node_labels=tuple(labels),
    )


def dense_summing_matrix(lattice: Lattice) -> np.ndarray:
    """Build the dense float64 summing matrix ``(n_nodes, n_bottom)`` of 0/1 entries."""
    n_bottom = lattice.n_bottom
    rows: list[np.ndarray] = list(np.eye(n_bottom, dtype=np.float64))
    for name in lattice.attribute_names:
        values = lattice.attribute_values[name]
        for value in sorted(set(values)):
            rows.append(np.array([member == value for member in values], dtype=np.float64))
    rows.append(np.ones(n_bottom, dtype=np.float64))
    S = np.vstack(rows)
    if S.shape != (lattice.n_nodes, n_bottom):
        raise ValueError("dense summing matrix shape drifted from lattice metadata")
    return S


def sparse_summing_matrix(lattice: Lattice) -> sparse.csr_array:
    """Build the csr summing matrix directly from membership coordinates.

    No dense intermediate is materialized: coordinates come from the identity
    block, the per-attribute memberships, and the total row, exactly the
    construction the full-scale engine must use so the ~7.6 GiB dense matrix
    is never a transient of the sparse build.
    """
    n_bottom = lattice.n_bottom
    bottom_positions = np.arange(n_bottom, dtype=np.int32)
    row_parts: list[np.ndarray] = [bottom_positions]
    next_row = n_bottom
    for name in lattice.attribute_names:
        values = np.asarray(lattice.attribute_values[name])
        unique_values, codes = np.unique(values, return_inverse=True)
        row_parts.append(np.asarray(next_row + codes, dtype=np.int32))
        next_row += len(unique_values)
    row_parts.append(np.full(n_bottom, next_row, dtype=np.int32))
    if next_row + 1 != lattice.n_nodes:
        raise ValueError("sparse row derivation drifted from the lattice node labels")
    rows = np.concatenate(row_parts)
    cols = np.tile(bottom_positions, lattice.n_attributes + 2)
    data = np.ones(rows.size, dtype=np.float64)
    S = sparse.csr_array((data, (rows, cols)), shape=(lattice.n_nodes, n_bottom))
    if S.nnz != rows.size or not np.all(S.data == 1.0):
        raise ValueError("sparse summing matrix produced duplicate membership coordinates")
    return S


def structural_weights(lattice: Lattice) -> np.ndarray:
    """Return the structural weight vector: the bottom-member count of every node.

    ``w_i = (S S.T)_ii``, which for a 0/1 summing matrix is the row sum: one
    for bottom rows, the member count for aggregate rows, ``n_bottom`` for the
    total row. This is the whole estimator — the structural-weights projection
    needs no residuals or covariance estimate.
    """
    weights = [1.0] * lattice.n_bottom
    for name in lattice.attribute_names:
        values = lattice.attribute_values[name]
        weights.extend(float(values.count(value)) for value in sorted(set(values)))
    weights.append(float(lattice.n_bottom))
    return np.asarray(weights, dtype=np.float64)


def derived_projection_tolerance(
    *, max_members: int, condition: float, magnitude: float, solver_rtol: float
) -> float:
    """Compute the coherence/equivalence tolerance from the problem instance.

    This is the single shared tolerance function the runtime coherence check
    and the tests both read (`[REC-12]`); no literal tolerance appears at a
    verification site. Derivation: the bottom block of a projection carries
    the solve's forward error, bounded by ``2 * condition * (eps +
    solver_rtol)`` relative to the vector magnitude (direct solve:
    ``solver_rtol = 0``; iterative solve with relative residual tolerance
    ``solver_rtol``: the residual bound lifts to the solution through the
    condition number). A node's coherence residual sums at most
    ``max_members`` such bottom errors, plus the 0/1 summation's own rounding
    of one rounding unit per added term. The tolerance is therefore::

        magnitude * max_members * (2 * condition * (eps + solver_rtol) + eps)

    evaluated in float64. At full scale the production engine substitutes a
    condition *estimate* for the exact fixture-scale ``condition``; the
    derivation is unchanged.
    """
    eps = np.finfo(np.float64).eps
    return magnitude * max_members * (2.0 * condition * (eps + solver_rtol) + eps)


def coherence_residual(S: np.ndarray, reconciled: np.ndarray, n_bottom: int) -> np.ndarray:
    """Return ``r - S @ r[:n_bottom]``: zero exactly when ``r`` is coherent."""
    return reconciled - S @ reconciled[:n_bottom]
