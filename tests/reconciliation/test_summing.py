from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    build_summing_matrix,
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


def test_null_attribute_value_raises() -> None:
    frame = pd.DataFrame({"unique_id": ["a", "b"], "store": ["S1", None]})
    with pytest.raises(ValueError, match="has null values"):
        build_summing_matrix(frame)


def test_missing_unique_id_column_raises() -> None:
    frame = pd.DataFrame({"store": ["S1", "S2"]})
    with pytest.raises(ValueError, match="missing required column: unique_id"):
        build_summing_matrix(frame)
