"""Build immutable dense and CSR-style summing matrices from hierarchy facts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

import numpy as np
from scipy import sparse

from newcalibre.domain import HierarchyIndex, HierarchyNode


class SummingMatrixError(ValueError):
    """Report invalid summing-matrix construction or evaluation input."""


@runtime_checkable
class SummingMatrix(Protocol):
    """Expose one representation-blind, label-indexed summing interface."""

    @property
    def bottom_ids(self) -> tuple[str, ...]:
        """Return canonical bottom labels indexing matrix columns."""
        ...

    @property
    def node_labels(self) -> tuple[str, ...]:
        """Return canonical hierarchy labels indexing matrix rows."""
        ...

    @property
    def n_bottom(self) -> int:
        """Return the number of bottom columns."""
        ...

    @property
    def n_nodes(self) -> int:
        """Return the number of hierarchy rows."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Return matrix row and column dimensions."""
        ...

    @property
    def reduction_width(self) -> int:
        """Return the largest number of bottom members reduced by one row."""
        ...

    @property
    def bottom_identity(self) -> np.ndarray:
        """Materialize the leading bottom identity block."""
        ...

    def subset(self, present_ids: Iterable[str]) -> Self:
        """Keep selected bottoms and every row with at least one selected member."""
        ...

    def matvec(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate the hierarchy vector for one bottom vector."""
        ...

    def to_dense(self) -> np.ndarray:
        """Materialize an isolated dense float64 value matrix."""
        ...


class _SummingMatrixOps:
    __slots__ = ()

    bottom_ids: tuple[str, ...]
    node_labels: tuple[str, ...]

    def to_dense(self) -> np.ndarray:
        """Materialize an isolated dense float64 matrix."""
        raise NotImplementedError

    @property
    def n_bottom(self) -> int:
        return len(self.bottom_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_nodes, self.n_bottom

    @property
    def bottom_identity(self) -> np.ndarray:
        return self.to_dense()[: self.n_bottom].copy()

    @property
    def total_index(self) -> int:
        """Return the canonical total-row index."""
        if not self.node_labels:
            raise SummingMatrixError("an empty summing subset has no total row")
        return self.n_nodes - 1

    def _bottom_vector(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise SummingMatrixError("bottom vector must contain real numeric values") from error
        if vector.ndim != 1:
            raise SummingMatrixError("bottom vector must be one-dimensional")
        if len(vector) != self.n_bottom:
            raise SummingMatrixError(
                f"bottom vector length must be {self.n_bottom}, received {len(vector)}"
            )
        return vector

    def _selected_positions(self, present_ids: Iterable[str]) -> tuple[int, ...]:
        if isinstance(present_ids, (str, bytes)):
            raise SummingMatrixError("present bottom ids must be an iterable of labels")
        try:
            supplied = tuple(present_ids)
        except TypeError as error:
            raise SummingMatrixError("present bottom ids must be iterable") from error
        if any(not isinstance(label, str) or not label for label in supplied):
            raise SummingMatrixError("present bottom ids must be non-empty strings")
        if len(set(supplied)) != len(supplied):
            raise SummingMatrixError("present bottom ids must be unique")
        unknown = sorted(set(supplied) - set(self.bottom_ids), key=str.encode)
        if unknown:
            raise SummingMatrixError(f"unknown bottom ids: {unknown}")
        selected = set(supplied)
        return tuple(index for index, label in enumerate(self.bottom_ids) if label in selected)


@dataclass(frozen=True, slots=True)
class DenseSummingMatrix(_SummingMatrixOps):
    """Store exact dense matrix values in immutable row tuples."""

    _values: tuple[tuple[float, ...], ...]
    bottom_ids: tuple[str, ...]
    node_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self._values) != len(self.node_labels):
            raise SummingMatrixError("dense row count must match node labels")
        if any(len(row) != len(self.bottom_ids) for row in self._values):
            raise SummingMatrixError("dense column count must match bottom labels")
        if any(value not in (0.0, 1.0) for row in self._values for value in row):
            raise SummingMatrixError("summing matrix entries must be exactly zero or one")
        _validate_labels(self.bottom_ids, self.node_labels)
        _validate_bottom_identity(self.to_dense(), n_bottom=self.n_bottom)

    @property
    def reduction_width(self) -> int:
        """Return the largest dense row membership count."""
        return max((sum(value == 1.0 for value in row) for row in self._values), default=0)

    def subset(self, present_ids: Iterable[str]) -> DenseSummingMatrix:
        """Keep selected columns and canonically ordered nonempty rows."""
        positions = self._selected_positions(present_ids)
        selected_rows: list[tuple[float, ...]] = []
        selected_labels: list[str] = []
        for label, row in zip(self.node_labels, self._values, strict=True):
            selected = tuple(row[index] for index in positions)
            if any(selected):
                selected_rows.append(selected)
                selected_labels.append(label)
        return DenseSummingMatrix(
            tuple(selected_rows),
            tuple(self.bottom_ids[index] for index in positions),
            tuple(selected_labels),
        )

    def matvec(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate dense rows against one validated bottom vector."""
        vector = self._bottom_vector(values)
        matrix = np.asarray(self._values, dtype=np.float64).reshape(self.shape)
        return matrix @ vector

    def to_dense(self) -> np.ndarray:
        """Materialize an isolated dense float64 matrix."""
        return np.asarray(self._values, dtype=np.float64).reshape(self.shape).copy()


@dataclass(frozen=True, slots=True)
class SparseSummingMatrix(_SummingMatrixOps):
    """Store an immutable CSR-style matrix without a SciPy dependency."""

    data: tuple[float, ...]
    indices: tuple[int, ...]
    indptr: tuple[int, ...]
    bottom_ids: tuple[str, ...]
    node_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_labels(self.bottom_ids, self.node_labels)
        if len(self.indptr) != self.n_nodes + 1 or not self.indptr or self.indptr[0] != 0:
            raise SummingMatrixError("sparse row pointer must index every matrix row")
        if tuple(sorted(self.indptr)) != self.indptr or self.indptr[-1] != len(self.indices):
            raise SummingMatrixError("sparse row pointer must be monotone and cover indices")
        if len(self.data) != len(self.indices) or any(value != 1.0 for value in self.data):
            raise SummingMatrixError("sparse data must contain one exact value per index")
        if any(index < 0 or index >= self.n_bottom for index in self.indices):
            raise SummingMatrixError("sparse column index is outside the matrix")
        for row in range(self.n_nodes):
            start, stop = self.indptr[row : row + 2]
            row_indices = self.indices[start:stop]
            if tuple(sorted(set(row_indices))) != row_indices:
                raise SummingMatrixError("sparse row indices must be unique and ordered")
        _validate_bottom_identity(self, n_bottom=self.n_bottom)

    @property
    def reduction_width(self) -> int:
        """Return the largest sparse row membership count."""
        return max(
            (self.indptr[row + 1] - self.indptr[row] for row in range(self.n_nodes)),
            default=0,
        )

    def subset(self, present_ids: Iterable[str]) -> SparseSummingMatrix:
        """Keep selected columns and canonically ordered nonempty rows."""
        positions = self._selected_positions(present_ids)
        remapped = {old: new for new, old in enumerate(positions)}
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        labels: list[str] = []
        for row, label in enumerate(self.node_labels):
            start, stop = self.indptr[row : row + 2]
            row_indices = [
                remapped[index] for index in self.indices[start:stop] if index in remapped
            ]
            if not row_indices:
                continue
            indices.extend(row_indices)
            data.extend(1.0 for _ in row_indices)
            indptr.append(len(indices))
            labels.append(label)
        return SparseSummingMatrix(
            tuple(data),
            tuple(indices),
            tuple(indptr),
            tuple(self.bottom_ids[index] for index in positions),
            tuple(labels),
        )

    def matvec(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate CSR rows against one validated bottom vector."""
        vector = self._bottom_vector(values)
        result = np.empty(self.n_nodes, dtype=np.float64)
        for row in range(self.n_nodes):
            start, stop = self.indptr[row : row + 2]
            result[row] = np.sum(vector[list(self.indices[start:stop])], dtype=np.float64)
        return result

    def to_csr(self) -> sparse.csr_array:
        """Materialize isolated SciPy CSR buffers without a dense intermediate."""
        return sparse.csr_array(
            (
                np.asarray(self.data, dtype=np.float64),
                np.asarray(self.indices, dtype=np.int32),
                np.asarray(self.indptr, dtype=np.int32),
            ),
            shape=self.shape,
        )

    def to_dense(self) -> np.ndarray:
        """Materialize an isolated dense float64 matrix."""
        matrix = np.zeros(self.shape, dtype=np.float64)
        for row in range(self.n_nodes):
            start, stop = self.indptr[row : row + 2]
            matrix[row, list(self.indices[start:stop])] = self.data[start:stop]
        return matrix


def build_dense_summing_matrix(
    hierarchy: HierarchyIndex,
    *,
    bottom_ids: Iterable[str] | None = None,
) -> DenseSummingMatrix:
    """Build an exact identity-first dense matrix, optionally for one bottom subset."""
    selected_bottoms, selected_nodes = _selected_hierarchy(hierarchy, bottom_ids=bottom_ids)
    rows = tuple(
        tuple(1.0 if bottom in node.members else 0.0 for bottom in selected_bottoms)
        for node in selected_nodes
    )
    return DenseSummingMatrix(
        rows,
        selected_bottoms,
        tuple(node.label for node in selected_nodes),
    )


def build_sparse_summing_matrix(
    hierarchy: HierarchyIndex,
    *,
    bottom_ids: Iterable[str] | None = None,
) -> SparseSummingMatrix:
    """Build immutable CSR buffers directly, optionally for one bottom subset."""
    selected_bottoms, selected_nodes = _selected_hierarchy(hierarchy, bottom_ids=bottom_ids)
    bottom_positions = {label: index for index, label in enumerate(selected_bottoms)}
    indices: list[int] = []
    indptr = [0]
    for node in selected_nodes:
        indices.extend(
            bottom_positions[member] for member in node.members if member in bottom_positions
        )
        indptr.append(len(indices))
    return SparseSummingMatrix(
        data=(1.0,) * len(indices),
        indices=tuple(indices),
        indptr=tuple(indptr),
        bottom_ids=selected_bottoms,
        node_labels=tuple(node.label for node in selected_nodes),
    )


def _selected_hierarchy(
    hierarchy: HierarchyIndex,
    *,
    bottom_ids: Iterable[str] | None,
) -> tuple[tuple[str, ...], tuple[HierarchyNode, ...]]:
    _require_hierarchy(hierarchy)
    if bottom_ids is None:
        return hierarchy.bottom_series, hierarchy.nodes
    if isinstance(bottom_ids, (str, bytes)):
        raise SummingMatrixError("present bottom ids must be an iterable of labels")
    try:
        supplied = tuple(bottom_ids)
    except TypeError as error:
        raise SummingMatrixError("present bottom ids must be iterable") from error
    if any(not isinstance(label, str) or not label for label in supplied):
        raise SummingMatrixError("present bottom ids must be non-empty strings")
    if len(set(supplied)) != len(supplied):
        raise SummingMatrixError("present bottom ids must be unique")
    known = set(hierarchy.bottom_series)
    unknown = sorted(set(supplied) - known, key=str.encode)
    if unknown:
        raise SummingMatrixError(f"unknown bottom ids: {unknown}")
    supplied_set = set(supplied)
    selected_bottoms = tuple(label for label in hierarchy.bottom_series if label in supplied_set)
    selected_nodes = tuple(
        node for node in hierarchy.nodes if any(member in supplied_set for member in node.members)
    )
    return selected_bottoms, selected_nodes


def _require_hierarchy(hierarchy: object) -> None:
    if not isinstance(hierarchy, HierarchyIndex):
        raise SummingMatrixError("summing matrix requires a HierarchyIndex")


def _validate_labels(bottom_ids: tuple[str, ...], node_labels: tuple[str, ...]) -> None:
    if len(set(bottom_ids)) != len(bottom_ids) or len(set(node_labels)) != len(node_labels):
        raise SummingMatrixError("summing matrix labels must be unique")
    if tuple(node_labels[: len(bottom_ids)]) != bottom_ids:
        raise SummingMatrixError("summing matrix must use identity-first node labels")


def _validate_bottom_identity(matrix: object, *, n_bottom: int) -> None:
    if isinstance(matrix, SparseSummingMatrix):
        for row in range(n_bottom):
            start, stop = matrix.indptr[row : row + 2]
            if matrix.indices[start:stop] != (row,):
                raise SummingMatrixError("summing matrix must begin with the bottom identity")
        return
    values = np.asarray(matrix, dtype=np.float64)
    if not np.array_equal(values[:n_bottom], np.eye(n_bottom, dtype=np.float64)):
        raise SummingMatrixError("summing matrix must begin with the bottom identity")
