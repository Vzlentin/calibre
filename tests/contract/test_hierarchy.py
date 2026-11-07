"""H1, H2: the nodes of `Hierarchy.from_attributes` and `Hierarchy.flat`, and `aggregate`."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from calibre import Hierarchy, Panel

from ._panels import from_panel, make_dense, to_panel

SERIES = np.array(["s3", "s1", "s2", "s4", "s5", "s6"])


def attributes() -> pd.DataFrame:
    """Rows in another order than SERIES, with an extra row that is ignored."""
    frame = pd.DataFrame(
        {
            "state": ["CA", "CA", "TX", "TX", "WA", "CA", "NV"],
            "cat": ["F", "H", "F", "F", "H", "H", "F"],
            "dept": [10, 9, 9, 10, 9, 9, 8],
            "one": ["x"] * 7,
            "store": ["S1", "S1", "S2", "S2", "S3", "S4", "S9"],
        },
        index=pd.Index(["s3", "s1", "s2", "s4", "s5", "s6", "zz"]),
    )
    return frame.iloc[[4, 6, 0, 3, 1, 5, 2]]


def expected_matrix(groups: list[list[int]]) -> np.ndarray:
    rows = [np.eye(6)[i] for i in range(6)]
    for members in groups:
        row = np.zeros(6)
        row[members] = 1.0
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def check(hierarchy: Hierarchy, nodes, level, groups) -> None:
    assert list(hierarchy.nodes) == nodes
    assert all(isinstance(node, str) for node in hierarchy.nodes)
    assert isinstance(hierarchy.summing, sp.csr_array)
    assert hierarchy.summing.dtype == np.float32
    assert hierarchy.summing.shape == (len(nodes), 6)
    np.testing.assert_array_equal(hierarchy.summing.toarray(), expected_matrix(groups))
    assert hierarchy.level.dtype == np.int32
    np.testing.assert_array_equal(hierarchy.level, level)
    assert hierarchy.n_bottom == 6


def test_from_attributes_levels():
    """H1: bottoms first in the order of `series`, then one aggregate per distinct value of
    each level in groupby order, crossed labels joined by '/', duplicates dropped, total last."""
    hierarchy = Hierarchy.from_attributes(
        SERIES, attributes(), ["state", ["state", "cat"], "one", "dept", "store"]
    )
    nodes = [
        *SERIES,
        "state=CA",
        "state=TX",
        "state=CA/cat=H",
        "dept=9",
        "dept=10",
        "store=S1",
        "total",
    ]
    level = [0] * 6 + [1, 1, 2, 4, 4, 5, 6]
    groups = [[0, 1, 5], [2, 3], [1, 5], [1, 2, 4, 5], [0, 3], [0, 1], list(range(6))]
    check(hierarchy, nodes, level, groups)


def test_crossed_level_order():
    """H1: the aggregates of a crossed level follow groupby(sort=True) on its columns, not the
    order in which the values first appear."""
    series = np.array(["a", "b", "c", "d", "e", "f"])
    frame = pd.DataFrame(
        {"x": ["q", "q", "p", "p", "q", "p"], "y": [2, 1, 2, 1, 1, 2]}, index=pd.Index(series)
    )
    hierarchy = Hierarchy.from_attributes(series, frame, [["x", "y"]])
    assert list(hierarchy.nodes) == [*series, "x=p/y=2", "x=q/y=1", "total"]
    np.testing.assert_array_equal(hierarchy.level, [0] * 6 + [1, 1, 2])
    rows = hierarchy.summing.toarray()[6:8]
    np.testing.assert_array_equal(rows, [[0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0]])


def test_from_attributes_default_levels():
    """H1: levels=None makes one level per column, in the order of the columns."""
    hierarchy = Hierarchy.from_attributes(SERIES, attributes())
    nodes = [
        *SERIES,
        "state=CA",
        "state=TX",
        "cat=F",
        "cat=H",
        "dept=9",
        "dept=10",
        "store=S1",
        "total",
    ]
    level = [0] * 6 + [1, 1, 2, 2, 3, 3, 5, 6]
    groups = [[0, 1, 5], [2, 3], [0, 2, 3], [1, 4, 5], [1, 2, 4, 5], [0, 3], [0, 1], list(range(6))]
    check(hierarchy, nodes, level, groups)


def test_flat():
    """H1: flat gives the bottoms only, with the identity as summing matrix."""
    hierarchy = Hierarchy.flat(SERIES)
    assert list(hierarchy.nodes) == list(SERIES)
    assert isinstance(hierarchy.summing, sp.csr_array)
    assert hierarchy.summing.dtype == np.float32
    np.testing.assert_array_equal(hierarchy.summing.toarray(), np.eye(6, dtype=np.float32))
    np.testing.assert_array_equal(hierarchy.level, np.zeros(6, np.int32))
    assert hierarchy.level.dtype == np.int32
    assert hierarchy.n_bottom == 6


def test_from_attributes_raises():
    """H1: from_attributes raises for fewer than two series, a missing series, a missing
    value, a level that is not a column, a naming collision, and attributes that are not a
    DataFrame."""
    frame = attributes()
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(SERIES[:1], frame)
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(SERIES, frame, ["state", ["state", "region"]])
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(SERIES, frame.drop(index="s4"))
    missing = frame.copy()
    missing.loc["s2", "cat"] = None
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(SERIES, missing)
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(
            np.array(["total", "x"]),
            pd.DataFrame({"one": ["a", "a"]}, index=pd.Index(["total", "x"])),
        )
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(
            np.array(["state=CA", "s1", "s2"]),
            pd.DataFrame({"state": ["CA", "CA", "TX"]}, index=pd.Index(["state=CA", "s1", "s2"])),
        )
    with pytest.raises(ValueError):
        Hierarchy.from_attributes(
            np.array(["s1", "s2", "s3", "s4"]),
            pd.DataFrame(
                {
                    "a": ["x", "x", "x/b=1", "x/b=1"],
                    "b": ["1", "1", "2", "3"],
                },
                index=pd.Index(["s1", "s2", "s3", "s4"]),
            ),
            [["a", "b"], "a"],
        )
    from_attributes: Any = Hierarchy.from_attributes  # a dict, not a DataFrame, on purpose
    with pytest.raises(TypeError):
        from_attributes(SERIES, frame.to_dict())


def test_constructor_raises_on_shapes():
    """H1: the constructor checks that nodes, summing and level agree in shape and that node
    labels are unique."""
    good = Hierarchy.from_attributes(SERIES, attributes())
    rows = good.summing.toarray()
    with pytest.raises(ValueError):
        Hierarchy(nodes=good.nodes[:-1], summing=good.summing, level=good.level)
    with pytest.raises(ValueError):
        Hierarchy(nodes=good.nodes, summing=sp.csr_array(rows[:-1]), level=good.level)
    with pytest.raises(ValueError):
        Hierarchy(nodes=good.nodes, summing=good.summing, level=good.level[:-1])
    dup = good.nodes.copy()
    dup[-1] = dup[0]
    with pytest.raises(ValueError):
        Hierarchy(nodes=dup, summing=good.summing, level=good.level)


def bottoms(seed: int):
    periods = pd.date_range("2024-03-04", periods=60, freq="D")
    spans = [(0, 60), (10, 60), (0, 25), (40, 60), (5, 50), (0, 60)]
    rng = np.random.default_rng(seed)
    return make_dense(rng, tuple(SERIES), periods, "D", spans, coarse={5})


def expected_nodes(values: np.ndarray, summing: np.ndarray) -> np.ndarray:
    """[N, T]: float64 sums of the members, NaN outside the span of each node."""
    finite = np.isfinite(values)
    out = np.full((summing.shape[0], values.shape[1]), np.nan)
    for n, row in enumerate(summing):
        members = row > 0
        covered = finite[members].any(axis=0)
        first, last = np.flatnonzero(covered)[[0, -1]]
        total = np.nan_to_num(values[members]).sum(axis=0)
        out[n, first : last + 1] = total[first : last + 1]
    return out.astype(np.float32).astype(np.float64)


def test_aggregate_values():
    """H2: aggregate gives the nodes on the same calendar. A node runs from the first period
    of its members to their last, and sums them in float64, with zero outside a member span."""
    dense = bottoms(seed=12)
    panel = to_panel(dense)
    hierarchy = Hierarchy.from_attributes(SERIES, attributes(), ["state", ["state", "cat"], "dept"])
    nodes = hierarchy.aggregate(panel)
    assert isinstance(nodes, Panel)
    assert list(nodes.series) == list(hierarchy.nodes)
    assert nodes.periods.equals(panel.periods)
    assert pd.tseries.frequencies.to_offset(nodes.freq) == pd.tseries.frequencies.to_offset("D")
    assert nodes.data.dtype == np.float32
    assert nodes.censored is None
    summing = hierarchy.summing.toarray()
    expected = expected_nodes(dense.values, summing)
    np.testing.assert_array_equal(from_panel(nodes), expected)
    tx = list(hierarchy.nodes).index("state=TX")
    assert np.all(expected[tx, 25:40] == 0.0)  # a gap between members sums to zero
    # The data must tell a float64 sum from a float32 one.
    float32_sum = np.zeros(expected.shape[1], np.float32)
    for row in np.nan_to_num(dense.values).astype(np.float32):
        float32_sum = float32_sum + row
    total = list(hierarchy.nodes).index("total")
    assert expected[total].max() > 2**18
    assert np.any(float32_sum != expected[total].astype(np.float32))


def test_aggregate_flat_is_the_panel():
    """H2: with the flat hierarchy, each node is its own bottom."""
    dense = bottoms(seed=13)
    panel = to_panel(dense)
    nodes = Hierarchy.flat(SERIES).aggregate(panel)
    np.testing.assert_array_equal(nodes.data, panel.data)
    np.testing.assert_array_equal(nodes.indptr, panel.indptr)
    np.testing.assert_array_equal(nodes.starts, panel.starts)


def test_aggregate_raises():
    """H2: aggregate takes a panel whose series are the bottoms, in order."""
    dense = bottoms(seed=14)
    panel = to_panel(dense)
    hierarchy = Hierarchy.from_attributes(SERIES, attributes())
    order = np.argsort(SERIES)
    sorted_panel = Panel.from_frame(
        pd.DataFrame(
            {
                "unique_id": np.repeat(SERIES[order], 3),
                "ds": np.tile(pd.date_range("2024-01-01", periods=3), 6),
                "y": np.ones(18, np.float32),
            }
        ),
        freq="D",
    )
    with pytest.raises(ValueError):
        hierarchy.aggregate(sorted_panel)
    with pytest.raises(ValueError):
        Hierarchy.flat(SERIES[:5]).aggregate(panel)
    with pytest.raises(TypeError):
        hierarchy.aggregate(dense.values)
