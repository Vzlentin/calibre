"""Exercise deterministic dense and sparse reconciliation summing matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newcalibre.domain import HierarchyIndex
from newcalibre.reconcile import (
    DenseSummingMatrix,
    SparseSummingMatrix,
    SummingMatrixError,
    build_dense_summing_matrix,
    build_sparse_summing_matrix,
    coherence_tolerance,
)


def _hierarchy(*, permuted: bool = False) -> HierarchyIndex:
    records = [
        {"series_key": "c", "department": "outer", "store": 2},
        {"series_key": "a", "department": "inner", "store": 1},
        {"series_key": "b", "department": "inner", "store": 2},
    ]
    if permuted:
        records = list(reversed(records))
    facts = pd.DataFrame.from_records(records)
    columns = ["store", "series_key", "department"] if permuted else list(facts.columns)
    return HierarchyIndex.from_facts(facts.loc[:, columns], bottom_series=("c", "b", "a"))


def test_summing_matrix_node_and_membership_identities() -> None:
    hierarchy = _hierarchy()
    dense = build_dense_summing_matrix(hierarchy)

    expected_memberships = sum(node.expected_member_count for node in hierarchy.nodes)
    assert isinstance(dense, DenseSummingMatrix)
    assert dense.bottom_ids == hierarchy.bottom_series
    assert dense.node_labels == hierarchy.node_labels
    assert dense.shape == (len(hierarchy.nodes), len(hierarchy.bottom_series))
    assert dense.n_nodes == len(hierarchy.nodes)
    assert dense.n_bottom == len(hierarchy.bottom_series)
    assert np.array_equal(dense.bottom_identity, np.eye(dense.n_bottom))
    assert np.count_nonzero(dense.to_dense()) == expected_memberships
    assert set(np.unique(dense.to_dense())) == {0.0, 1.0}


def test_dense_and_sparse_builds_are_exact_and_deterministic() -> None:
    first = _hierarchy()
    rebuilt = _hierarchy(permuted=True)

    dense = build_dense_summing_matrix(first)
    sparse = build_sparse_summing_matrix(first)
    rebuilt_dense = build_dense_summing_matrix(rebuilt)
    rebuilt_sparse = build_sparse_summing_matrix(rebuilt)

    assert isinstance(sparse, SparseSummingMatrix)
    assert dense.bottom_ids == sparse.bottom_ids == rebuilt_dense.bottom_ids
    assert dense.node_labels == sparse.node_labels == rebuilt_sparse.node_labels
    assert np.array_equal(dense.to_dense(), sparse.to_dense())
    assert np.array_equal(dense.to_dense(), rebuilt_dense.to_dense())
    assert np.array_equal(sparse.to_dense(), rebuilt_sparse.to_dense())
    vector = np.array([2.0, 3.0, 5.0])
    assert np.array_equal(dense.matvec(vector), sparse.matvec(vector))


def test_sparse_builder_never_calls_the_dense_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newcalibre.reconcile.summing as summing

    def forbidden(_hierarchy: HierarchyIndex) -> DenseSummingMatrix:
        raise AssertionError("sparse construction called the dense builder")

    monkeypatch.setattr(summing, "build_dense_summing_matrix", forbidden)

    sparse = summing.build_sparse_summing_matrix(_hierarchy())

    assert isinstance(sparse, SparseSummingMatrix)


@pytest.mark.parametrize("builder", [build_dense_summing_matrix, build_sparse_summing_matrix])
def test_subset_keeps_canonical_bottoms_and_nonempty_rows(builder) -> None:
    matrix = builder(_hierarchy())

    subset = matrix.subset(("c", "a"))
    selected_columns = [matrix.bottom_ids.index(label) for label in ("a", "c")]
    expected = matrix.to_dense()[:, selected_columns]
    keep = expected.sum(axis=1) > 0

    assert subset.bottom_ids == ("a", "c")
    assert subset.node_labels == tuple(
        label for label, present in zip(matrix.node_labels, keep, strict=True) if present
    )
    assert np.array_equal(subset.to_dense(), expected[keep])
    assert np.array_equal(subset.bottom_identity, np.eye(2))

    with pytest.raises(SummingMatrixError, match="unknown bottom ids.*foreign"):
        matrix.subset(("foreign",))


def test_matrix_vector_validation_rejects_wrong_dimensions() -> None:
    matrix = build_sparse_summing_matrix(_hierarchy())

    with pytest.raises(SummingMatrixError, match="one-dimensional"):
        matrix.matvec(np.ones((matrix.n_bottom, 1)))
    with pytest.raises(SummingMatrixError, match="length"):
        matrix.matvec(np.ones(matrix.n_bottom + 1))


def test_coherence_tolerance_grows_with_width_magnitude_and_solver_tolerance() -> None:
    baseline = coherence_tolerance(reduction_width=2, vector_magnitude=1.0)
    wider = coherence_tolerance(reduction_width=8, vector_magnitude=1.0)
    larger = coherence_tolerance(reduction_width=8, vector_magnitude=100.0)
    solver_aware = coherence_tolerance(
        reduction_width=8,
        vector_magnitude=100.0,
        solver_tolerance=1e-8,
    )

    assert 0.0 < baseline < wider < larger < solver_aware
