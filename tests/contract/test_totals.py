"""R5, C6: no function reconciles thresholds or bounds, and each threshold comes from the
scores of its own node and column."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from conformalforecast.nonconformity import SignedResidual

from calibre import BottomUp, Forecasts, Hierarchy, Span, Split, Step, reconcile, replay

from . import _expected
from ._expected import assert_bits
from ._panels import from_panel, grid, make_dense, to_panel

SERIES = np.array(["b3", "b1", "b0", "b4", "b2", "b5"])
ATTRIBUTES = pd.DataFrame(
    {"group": ["x", "x", "y", "y", "y", "z"], "kind": ["k", "l", "k", "l", "k", "l"]},
    index=SERIES,
)


def coherent_run(seed: int = 51):
    """Bottoms, their hierarchy, the node truth, and BottomUp forecasts of the nodes."""
    periods = pd.date_range("2024-01-01", periods=90, freq="D")
    spans = [(0, 90), (0, 90), (3, 90), (0, 75), (0, 90), (8, 90)]
    rng = np.random.default_rng(seed)
    bottoms = make_dense(rng, tuple(SERIES), periods, "D", spans, coarse={5})
    hierarchy = Hierarchy.from_attributes(SERIES, ATTRIBUTES)
    truth = hierarchy.aggregate(to_panel(bottoms))
    origin_idx = np.arange(10, 86)
    points = grid(rng, (1, len(SERIES), 5, len(origin_idx))).astype(np.float32)
    forecasts = Forecasts(points, np.array(["m"]), SERIES, periods[origin_idx])
    nodes = reconcile(forecasts, hierarchy, BottomUp())
    return hierarchy, truth, nodes, origin_idx


def test_reconcile_takes_only_forecasts():
    """R5: reconcile raises TypeError when its input is not a Forecasts, so a Replay, its
    thresholds or its bounds cannot be reconciled."""
    hierarchy, truth, nodes, _ = coherent_run()
    rep = replay(
        nodes,
        truth,
        target=Span(1, 3),
        method=Split(score=SignedResidual(), window=12),
        level=0.9,
    )
    call: Any = reconcile  # the inputs below are not Forecasts, on purpose
    for value in (rep, rep.upper, rep.threshold, {"points": rep.upper}, nodes.points):
        with pytest.raises(TypeError):
            call(value, hierarchy, BottomUp())


@pytest.mark.parametrize("kind", ["step", "span"])
def test_one_point_changes_only_its_own_thresholds(kind):
    """C6: a change of one point changes only thresholds of its node and column, at later
    origins."""
    hierarchy, truth, nodes, _ = coherent_run()
    n_nodes = len(hierarchy.nodes)
    target = Step() if kind == "step" else Span(np.resize([1, 2, 1, 3], n_nodes), 4)
    method = Split(score=SignedResidual(), window=12)
    base = replay(nodes, truth, target=target, method=method, level=0.9)
    node, step, origin = n_nodes - 2, 2, 5
    points = np.array(nodes.points)
    points[0, node, step - 1, origin] = -(2.0**22)  # its score becomes the largest
    changed = replay(
        Forecasts(points, nodes.models, nodes.series, nodes.origins),
        truth,
        target=target,
        method=method,
        level=0.9,
    )
    cover = np.asarray(target.cover(n_nodes, 5))
    allowed = np.zeros(base.threshold.shape, bool)
    for c in np.flatnonzero(cover[node, :, step - 1]):
        allowed[node, c, origin + 1 :] = True
    assert_bits(changed.threshold[~allowed], base.threshold[~allowed])
    assert not np.array_equal(changed.threshold[allowed], base.threshold[allowed])


def test_aggregate_bound_is_its_own():
    """C6: the bound of an aggregate comes from the scores of that aggregate. It is not the sum
    of the bounds of its members."""
    hierarchy, truth, nodes, origin_idx = coherent_run(seed=52)
    target = Span(1, 3)
    rep = replay(
        nodes, truth, target=target, method=Split(score=SignedResidual(), window=12), level=0.9
    )
    n_nodes = len(hierarchy.nodes)
    cover = _expected.cover_span(n_nodes, 5, 1, 3)
    point, observed = _expected.columns(nodes.points[0], from_panel(truth), origin_idx, cover)
    score = _expected.scores(point, observed, signed=True)
    windows = _expected.members(origin_idx, _expected.last_steps(cover), 12)
    expected = _expected.rank_thresholds(score, windows, np.full(n_nodes, 1.0 - 0.9))
    assert_bits(rep.threshold, expected)
    total = n_nodes - 1
    members = hierarchy.summing.toarray()[total] > 0
    summed = rep.upper[: hierarchy.n_bottom][members].astype(np.float64).sum(axis=0)
    finite = np.isfinite(summed) & np.isfinite(rep.upper[total])
    assert finite.any()
    assert not np.array_equal(summed[finite], rep.upper[total][finite].astype(np.float64))
