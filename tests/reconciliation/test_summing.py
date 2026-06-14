"""Tests for the hierarchy summing matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    SparseSummingMatrix,
    build_hierarchy_index,
    build_summing_matrix,
    sparse_summing_matrix_from_index,
    summing_matrix_from_index,
)

M5_COLUMNS = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _two_attr_frame() -> pd.DataFrame:
    # Two grouping dimensions: category and store.
    return pd.DataFrame(
        {
            "unique_id": ["a", "b", "c"],
            "cat": ["X", "X", "Y"],
            "store": ["S1", "S2", "S1"],
        }
    )


def test_happy_path_two_attribute_shape_and_labels() -> None:
    summing = build_summing_matrix(_two_attr_frame())

    assert summing.bottom_ids == ("a", "b", "c")
    # 3 bottom + (cat: X, Y) + (store: S1, S2) + total = 3 + 2 + 2 + 1 = 8 nodes.
    assert summing.node_labels == (
        "a",
        "b",
        "c",
        "cat=X",
        "cat=Y",
        "store=S1",
        "store=S2",
        TOTAL_LABEL,
    )
    assert summing.S.shape == (8, 3)
    assert summing.S.dtype == np.float64
    # Bottom identity block leads the matrix.
    np.testing.assert_array_equal(summing.S[:3], np.eye(3))


def test_coherence_property_for_random_bottom_vector() -> None:
    summing = build_summing_matrix(_two_attr_frame())
    rng = np.random.default_rng(0)
    b = rng.normal(size=summing.n_bottom)
    agg = summing.S @ b

    labels = summing.node_labels
    assert agg[labels.index("cat=X")] == pytest.approx(b[0] + b[1])
    assert agg[labels.index("cat=Y")] == pytest.approx(b[2])
    assert agg[labels.index("store=S1")] == pytest.approx(b[0] + b[2])
    assert agg[labels.index("store=S2")] == pytest.approx(b[1])
    assert agg[summing.total_index] == pytest.approx(b.sum())


def test_m5_shape_validation() -> None:
    frame = pd.DataFrame(
        [
            ["HOBBIES_1_001_CA_1", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"],
            ["HOBBIES_1_001_CA_2", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_2", "CA"],
            ["HOBBIES_2_001_TX_1", "HOBBIES_2_001", "HOBBIES_2", "HOBBIES", "TX_1", "TX"],
            ["HOBBIES_2_001_TX_2", "HOBBIES_2_001", "HOBBIES_2", "HOBBIES", "TX_2", "TX"],
        ],
        columns=M5_COLUMNS,
    )
    summing = build_summing_matrix(frame)

    assert summing.n_bottom == 4
    # 5 attribute columns; distinct values: item_id 2, dept_id 2, cat_id 1,
    # store_id 4, state_id 2 => 11 aggregate rows, + 4 bottom + 1 total = 16.
    assert summing.n_nodes == 16
    assert summing.S.shape == (16, 4)
    assert "cat_id=HOBBIES" in summing.node_labels
    assert "state_id=CA" in summing.node_labels
    assert "store_id=CA_1" in summing.node_labels
    # cat_id=HOBBIES covers all four bottom series.
    cat_row = summing.S[summing.node_labels.index("cat_id=HOBBIES")]
    np.testing.assert_array_equal(cat_row, np.ones(4))
    # Grand total equals the bottom sum.
    np.testing.assert_array_equal(summing.S[summing.total_index], np.ones(4))


def test_single_bottom_series() -> None:
    summing = build_summing_matrix(pd.DataFrame({"unique_id": ["only"], "store": ["S1"]}))
    assert summing.bottom_ids == ("only",)
    # bottom (1) + store=S1 (1) + total (1) = 3 rows, all [1].
    assert summing.S.shape == (3, 1)
    np.testing.assert_array_equal(summing.S, np.ones((3, 1)))
    b = np.array([7.0])
    np.testing.assert_array_equal(summing.S @ b, np.array([7.0, 7.0, 7.0]))


def test_subset_to_strict_subset_of_bottom_ids() -> None:
    summing = build_summing_matrix(_two_attr_frame())
    # Keep only "a" and "c": store S2 (only b) and cat X-as-{a} ...
    sub = summing.subset(["a", "c"])

    assert sub.bottom_ids == ("a", "c")
    np.testing.assert_array_equal(sub.S[:2], np.eye(2))
    # "store=S2" had only member "b" (now absent) => row dropped.
    assert "store=S2" not in sub.node_labels
    # Aggregates that retain a member survive.
    assert "store=S1" in sub.node_labels  # members a, c
    assert "cat=X" in sub.node_labels  # member a
    assert "cat=Y" in sub.node_labels  # member c
    assert TOTAL_LABEL in sub.node_labels
    # Coherence holds on the subset.
    b = np.array([2.0, 5.0])
    agg = sub.S @ b
    assert agg[sub.node_labels.index("store=S1")] == pytest.approx(7.0)
    assert agg[sub.total_index] == pytest.approx(7.0)


def test_subset_rejects_unknown_ids() -> None:
    summing = build_summing_matrix(_two_attr_frame())
    with pytest.raises(ValueError, match="not in summing matrix bottom ids"):
        summing.subset(["a", "z"])


def test_duplicate_unique_id_raises() -> None:
    frame = pd.DataFrame({"unique_id": ["a", "a"], "store": ["S1", "S2"]})
    with pytest.raises(ValueError, match="duplicate unique_id"):
        build_summing_matrix(frame)


@pytest.mark.parametrize("colliding_id", ["cat=X", TOTAL_LABEL])
def test_generated_node_label_collision_raises(colliding_id: str) -> None:
    frame = pd.DataFrame(
        {
            "unique_id": [colliding_id, "b"],
            "cat": ["X", "Y"],
        }
    )
    with pytest.raises(ValueError, match="node labels must be unique"):
        build_summing_matrix(frame)


def test_null_attribute_value_raises() -> None:
    frame = pd.DataFrame({"unique_id": ["a", "b"], "store": ["S1", None]})
    with pytest.raises(ValueError, match="has null values"):
        build_summing_matrix(frame)


# ---------------------------------------------------------------------------
# expected_members predicate — the single stringified counting authority (#148).
# ---------------------------------------------------------------------------


def test_expected_members_matches_raw_counts_on_clean_string_hierarchy() -> None:
    index = build_hierarchy_index(_two_attr_frame())
    expected = index.expected_members()

    assert set(expected) == {"cat", "store"}
    # Counts are identical to a raw groupby on legal string data.
    assert expected["cat"].to_dict() == {"X": 2, "Y": 1}
    assert expected["store"].to_dict() == {"S1": 2, "S2": 1}


def test_expected_members_merges_str_colliding_values() -> None:
    # int 1 and str "1" on different bottom ids collide under str(): the
    # predicate counts both members in one coherent group (#148), matching what
    # node labels and the dense summing matrix already do.
    frame = pd.DataFrame({"unique_id": ["a", "b", "c"], "grp": [1, "1", 2]})
    index = build_hierarchy_index(frame)
    expected = index.expected_members()["grp"]

    assert expected["1"] == 2
    assert expected["2"] == 1
    # The dense summing matrix merges the same two values into one node row.
    summing = build_summing_matrix(frame)
    assert "grp=1" in summing.node_labels
    grp_row = summing.S[summing.node_labels.index("grp=1")]
    np.testing.assert_array_equal(grp_row, np.array([1.0, 1.0, 0.0]))


def test_expected_members_drops_categorical_phantom_groups() -> None:
    # A category-dtype column with an unobserved category would emit a phantom
    # zero-count group under raw groupby(observed=False); the stringified
    # predicate drops it by construction, matching node labels exactly.
    grp = pd.Categorical(["A", "A", "B"], categories=["A", "B", "C"])
    frame = pd.DataFrame({"unique_id": ["a", "b", "c"], "grp": grp})
    index = build_hierarchy_index(frame)
    expected = index.expected_members()["grp"]

    assert set(expected.index) == {"A", "B"}  # no phantom "C"
    summing = build_summing_matrix(frame)
    assert "grp=C" not in summing.node_labels
    assert {"grp=A", "grp=B"} <= set(summing.node_labels)


def test_missing_unique_id_column_raises() -> None:
    frame = pd.DataFrame({"store": ["S1", "S2"]})
    with pytest.raises(ValueError, match="missing required column: unique_id"):
        build_summing_matrix(frame)


# ---------------------------------------------------------------------------
# Sparse producer — csr built directly from index facts (#168).
# ---------------------------------------------------------------------------


def test_sparse_producer_matches_dense_matrix_exactly() -> None:
    index = build_hierarchy_index(_two_attr_frame())
    sparse_summing = sparse_summing_matrix_from_index(index)
    dense = summing_matrix_from_index(index)

    assert sparse_summing.bottom_ids == dense.bottom_ids
    assert sparse_summing.node_labels == dense.node_labels
    assert sparse_summing.n_bottom == dense.n_bottom
    assert sparse_summing.n_nodes == dense.n_nodes
    assert sparse_summing.total_index == dense.total_index
    np.testing.assert_array_equal(sparse_summing.S.toarray(), dense.S)


def test_sparse_producer_matches_dense_on_m5_shaped_frame() -> None:
    frame = pd.DataFrame(
        [
            ["HOBBIES_1_001_CA_1", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"],
            ["HOBBIES_1_001_CA_2", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_2", "CA"],
            ["HOBBIES_2_001_TX_1", "HOBBIES_2_001", "HOBBIES_2", "HOBBIES", "TX_1", "TX"],
            ["HOBBIES_2_001_TX_2", "HOBBIES_2_001", "HOBBIES_2", "HOBBIES", "TX_2", "TX"],
        ],
        columns=M5_COLUMNS,
    )
    index = build_hierarchy_index(frame)
    sparse_summing = sparse_summing_matrix_from_index(index)
    dense = summing_matrix_from_index(index)

    assert sparse_summing.node_labels == dense.node_labels
    np.testing.assert_array_equal(sparse_summing.S.toarray(), dense.S)


def test_sparse_producer_nnz_matches_analytic_formula_with_unit_values() -> None:
    index = build_hierarchy_index(_two_attr_frame())
    summing = sparse_summing_matrix_from_index(index)

    # Identity block + one membership per attribute column + total row.
    expected_nnz = summing.n_bottom * (2 + len(index.attr_cols))
    analytic_via_members = (
        summing.n_bottom
        + sum(int(counts.sum()) for counts in index.expected_members().values())
        + summing.n_bottom
    )
    assert summing.S.nnz == expected_nnz == analytic_via_members
    assert np.all(summing.S.data == 1.0)
    assert summing.S.dtype == np.float64


def test_sparse_producer_emits_csr_array_not_csr_matrix() -> None:
    summing = sparse_summing_matrix_from_index(build_hierarchy_index(_two_attr_frame()))

    assert isinstance(summing.S, sparse.csr_array)
    # csr_array row sums are 1-D ndarrays; csr_matrix would return np.matrix
    # and break the boolean row mask inside subset().
    row_sums = summing.S.sum(axis=1)
    assert isinstance(row_sums, np.ndarray)
    assert row_sums.ndim == 1


def test_sparse_subset_parity_with_dense() -> None:
    index = build_hierarchy_index(_two_attr_frame())
    sparse_sub = sparse_summing_matrix_from_index(index).subset(["a", "c"])
    dense_sub = summing_matrix_from_index(index).subset(["a", "c"])

    assert isinstance(sparse_sub, SparseSummingMatrix)
    assert isinstance(sparse_sub.S, sparse.csr_array)
    assert sparse_sub.bottom_ids == dense_sub.bottom_ids
    assert sparse_sub.node_labels == dense_sub.node_labels
    np.testing.assert_array_equal(sparse_sub.S.toarray(), dense_sub.S)
    # Coherence holds on the sparse subset and matvec stays a plain ndarray.
    b = np.array([2.0, 5.0])
    agg = sparse_sub.S @ b
    assert isinstance(agg, np.ndarray)
    assert agg[sparse_sub.node_labels.index("store=S1")] == pytest.approx(7.0)
    assert agg[sparse_sub.total_index] == pytest.approx(7.0)


def test_sparse_subset_to_single_bottom_id_mirrors_dense() -> None:
    index = build_hierarchy_index(_two_attr_frame())
    sparse_sub = sparse_summing_matrix_from_index(index).subset(["b"])
    dense_sub = summing_matrix_from_index(index).subset(["b"])

    assert sparse_sub.bottom_ids == dense_sub.bottom_ids == ("b",)
    assert sparse_sub.node_labels == dense_sub.node_labels
    np.testing.assert_array_equal(sparse_sub.S.toarray(), dense_sub.S)


def test_sparse_subset_rejects_unknown_ids() -> None:
    summing = sparse_summing_matrix_from_index(build_hierarchy_index(_two_attr_frame()))
    with pytest.raises(ValueError, match="not in summing matrix bottom ids"):
        summing.subset(["a", "z"])


def test_sparse_producer_single_attribute_single_bottom_matches_dense() -> None:
    index = build_hierarchy_index(pd.DataFrame({"unique_id": ["only"], "store": ["S1"]}))
    sparse_summing = sparse_summing_matrix_from_index(index)
    dense = summing_matrix_from_index(index)

    assert sparse_summing.node_labels == dense.node_labels
    np.testing.assert_array_equal(sparse_summing.S.toarray(), dense.S)
    np.testing.assert_array_equal(sparse_summing.S @ np.array([7.0]), np.array([7.0, 7.0, 7.0]))
