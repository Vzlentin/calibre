import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from calibre.data import Hierarchy
from tests.synthetic import make_panel


def test_hierarchy_from_attributes_shape_and_sums():
    panel, hierarchy = make_panel(n_series=20, n_periods=14)
    n_dept, n_store = 7, 10
    assert len(hierarchy.nodes) == 20 + n_dept + n_store + 1
    assert list(hierarchy.nodes[:20]) == list(panel.series)
    assert hierarchy.nodes[-1] == "total"
    assert hierarchy.n_bottom == 20
    row_sums = hierarchy.summing.sum(axis=1)
    assert row_sums[:20].tolist() == [1.0] * 20
    assert row_sums[-1] == 20.0
    assert row_sums[20 : 20 + n_dept].sum() == 20.0
    assert hierarchy.level.dtype == np.int32
    assert hierarchy.level.tolist() == [0] * 20 + [1] * n_dept + [2] * n_store + [3]


def test_hierarchy_flat_is_identity():
    series = np.array(["a", "b", "c"])
    hierarchy = Hierarchy.flat(series)
    assert list(hierarchy.nodes) == ["a", "b", "c"]
    np.testing.assert_array_equal(hierarchy.summing.toarray(), np.eye(3))
    assert hierarchy.level.tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("nodes", "summing", "level", "match"),
    [
        (["a", "b"], np.eye(2), [0], "one entry per node"),
        (["a", "a"], np.eye(2), [0, 0], "not unique"),
        (["a", "b"], [[2.0, 0.0], [1.0, 1.0]], [0, 0], "bottom identity"),
        (["a"], [[1.0, 1.0]], [0], "fewer rows"),
    ],
)
def test_hierarchy_rejects_invalid_inputs(nodes, summing, level, match):
    with pytest.raises(ValueError, match=match):
        Hierarchy(np.array(nodes), sp.csr_array(np.array(summing)), np.array(level))


def test_hierarchy_is_read_only_and_copies_its_matrix():
    source = sp.csr_array(np.eye(2))
    hierarchy = Hierarchy(np.array(["a", "b"]), source, np.zeros(2))
    source.data[0] = 5
    assert hierarchy.summing.toarray().tolist() == [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        hierarchy.summing.data[0] = 5


def crossed_attributes() -> tuple[np.ndarray, pd.DataFrame]:
    series = np.array(["a", "b", "c", "d", "e"])
    attributes = pd.DataFrame(
        {"state": ["CA", "CA", "TX", "TX", "TX"], "cat": ["F", "H", "F", "F", "H"]},
        index=series,
    )
    return series, attributes


def test_crossed_levels_keep_only_new_groups_of_bottoms():
    series, attributes = crossed_attributes()
    hierarchy = Hierarchy.from_attributes(series, attributes, ["state", "cat", ["state", "cat"]])
    # CA/F is a, CA/H is b, and TX/H is e: each one equals a bottom. TX/F is c and d.
    assert hierarchy.nodes[5:].tolist() == [
        "state=CA",
        "state=TX",
        "cat=F",
        "cat=H",
        "state=TX/cat=F",
        "total",
    ]
    assert hierarchy.level.tolist() == [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4]
    assert hierarchy.summing[9].toarray().tolist() == [0, 0, 1, 1, 0]


def test_an_aggregate_equal_to_the_total_or_an_earlier_aggregate_is_dropped():
    series, attributes = crossed_attributes()
    attributes["country"] = "US"
    attributes["region"] = attributes["state"].map({"CA": "west", "TX": "south"})
    hierarchy = Hierarchy.from_attributes(series, attributes, ["country", "state", "region"])
    assert hierarchy.nodes[5:].tolist() == ["state=CA", "state=TX", "total"]
    assert hierarchy.level.tolist() == [0] * 5 + [2, 2, 4]


def test_nodes_sum_bottom_values_and_flag_when_any_bottom_is_flagged():
    series, attributes = crossed_attributes()
    hierarchy = Hierarchy.from_attributes(series, attributes, ["state"])
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    flags = np.zeros((5, 2), dtype=bool)
    flags[3, 1] = True  # d, in TX
    nodes = list(hierarchy.nodes)
    totals = hierarchy.aggregate(values)
    assert totals[nodes.index("state=CA")].tolist() == [2, 4]
    assert totals[nodes.index("total")].tolist() == [20, 25]
    flagged = hierarchy.any_bottom(flags)
    assert flagged[nodes.index("state=TX")].tolist() == [False, True]
    assert flagged[nodes.index("state=CA")].tolist() == [False, False]
    assert flagged[nodes.index("total")].tolist() == [False, True]
