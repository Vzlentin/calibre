"""H2 for censored values, C7: the censored flag from bottoms to nodes and to cells."""

import numpy as np
import pandas as pd
import pytest

from calibre import Hierarchy

from . import _cases, _expected
from ._expected import assert_bits
from ._panels import make_dense, to_panel

SERIES = np.array(["c2", "c0", "c1", "c3", "c4"])
ATTRIBUTES = pd.DataFrame({"group": ["a", "a", "b", "b", "b"]}, index=SERIES)


def flags_of(panel) -> np.ndarray:
    """[S, T] bool: the censored flag of each value, False outside each span."""
    assert panel.censored is not None
    out = np.zeros((len(panel.series), len(panel.periods)), bool)
    for s in range(len(panel.series)):
        chunk = panel.censored[panel.indptr[s] : panel.indptr[s + 1]]
        out[s, panel.starts[s] : panel.starts[s] + len(chunk)] = chunk
    return out


def test_aggregate_flags():
    """H2: a node value is censored when one member value at that period is censored."""
    periods = pd.date_range("2024-01-01", periods=50, freq="D")
    spans = [(0, 50), (10, 50), (0, 30), (5, 45), (0, 50)]
    rng = np.random.default_rng(61)
    dense = make_dense(rng, tuple(SERIES), periods, "D", spans, censored_share=0.2)
    hierarchy = Hierarchy.from_attributes(SERIES, ATTRIBUTES)
    nodes = hierarchy.aggregate(to_panel(dense))
    assert nodes.censored is not None
    assert nodes.censored.dtype == np.bool_
    assert nodes.censored.shape == nodes.data.shape
    summing = hierarchy.summing.toarray() > 0
    flags = dense.censored
    assert flags is not None
    expected = np.array([flags[row].any(axis=0) for row in summing])
    assert np.array_equal(flags_of(nodes), expected)
    total = list(hierarchy.nodes).index("total")
    assert expected[total].sum() > expected[: len(SERIES)].sum(axis=1).max()


@pytest.mark.parametrize("name", _cases.NAMES)
def test_censored_cells(name):
    """C7: a cell is censored when one value of its truth is censored, and is not censored
    when its truth is NaN."""
    case = _cases.make(name, seed=3)
    exp = _cases.expected(case)
    rep = case.run()
    expected = _expected.censored_cells(case.dense.censored, case.origin_idx, case.cover, exp.truth)
    assert rep.censored.dtype == np.bool_
    assert np.array_equal(rep.censored, expected)
    assert expected.any()


def test_cases_hold_censored_values_under_nan_truth():
    """C7: some span columns hold a censored value and a NaN truth, so the rule that such a
    cell is not censored is checked."""
    found = False
    for name in _cases.NAMES:
        case = _cases.make(name, seed=3)
        exp = _cases.expected(case)
        flags = _expected.censored_cells(
            case.dense.censored, case.origin_idx, case.cover, np.zeros_like(exp.truth)
        )
        found |= bool((flags & np.isnan(exp.truth)).any())
    assert found


@pytest.mark.parametrize("name", _cases.NAMES)
def test_censored_scores_enter_as_observed(name):
    """C7: censored scores enter score windows as observed. Every output but the flag equals
    that of the same truth with no censored array, and with no censored array no cell is
    censored."""
    case = _cases.make(name, seed=3)
    flagged = case.run()
    plain = _cases.without_censored(case).run()
    for field in ("point", "truth", "score", "threshold", "lower", "upper"):
        assert_bits(getattr(flagged, field), getattr(plain, field))
    assert not plain.censored.any()
    assert flagged.censored.any()
