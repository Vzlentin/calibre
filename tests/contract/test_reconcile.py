"""R1, R2, R3, R4: `BottomUp`, `Identity` and `WlsStruct` through `reconcile`."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pydantic
import pytest

from calibre import BottomUp, Forecasts, Hierarchy, Identity, Reconciler, WlsStruct, reconcile

from . import _reference
from ._expected import assert_bits
from ._panels import FINE

SERIES = np.array(["i5", "i1", "i3", "i0", "i7", "i2", "i6", "i4"])
ATTRIBUTES = pd.DataFrame(
    {
        "state": ["CA", "CA", "TX", "TX", "TX", "WI", "WI", "CA"],
        "cat": ["F", "H", "F", "F", "H", "F", "H", "H"],
        "store": ["s1", "s1", "s2", "s3", "s3", "s4", "s4", "s5"],
    },
    index=SERIES,
)
LEVELS = ["state", ["state", "cat"], "store"]
ORIGINS = pd.date_range("2024-01-07", periods=4, freq="W-SUN")


def hierarchy() -> Hierarchy:
    return Hierarchy.from_attributes(SERIES, ATTRIBUTES, LEVELS)


def large(rng, shape) -> np.ndarray:
    """Multiples of 1/64 in [2**16, 2**17): sums of a few of them exceed 2**18."""
    return (rng.integers(FINE // 2, FINE, shape) / 64.0).astype(np.float32)


def test_bottom_up_sums_in_float64():
    """R1: BottomUp gives each node as the sum of its bottoms, in float64, stored as float32.
    A NaN bottom adds zero, and a node whose bottoms are all NaN is NaN."""
    tree = hierarchy()
    rng = np.random.default_rng(71)
    points = large(rng, (2, len(SERIES), 3, 4))
    points[0, 2, 1, 3] = np.nan
    points[1, 0, 0, 0] = np.nan
    points[1, 7, 0, 0] = np.nan
    points[1, 1, 0, 0] = np.nan  # with 0 and 7, every bottom of state=CA
    forecasts = Forecasts(points, np.array(["m1", "m2"]), SERIES, ORIGINS)
    got = reconcile(forecasts, tree, BottomUp())
    summing = tree.summing.toarray().astype(np.float64)
    expected = np.einsum("nb,mbho->mnho", summing, np.nan_to_num(points.astype(np.float64)))
    present = np.einsum("nb,mbho->mnho", summing, (~np.isnan(points)).astype(np.float64))
    expected[present == 0] = np.nan
    assert_bits(got.points, expected.astype(np.float32))
    assert list(got.series) == list(tree.nodes)
    assert list(got.models) == ["m1", "m2"]
    assert got.origins.equals(ORIGINS)
    ca = list(tree.nodes).index("state=CA")
    assert np.isnan(got.points[1, ca, 0, 0])
    float32_sum = np.einsum("nb,mbho->mnho", summing.astype(np.float32), np.nan_to_num(points))
    assert np.any(float32_sum[present > 0] != expected.astype(np.float32)[present > 0])


def test_identity_returns_its_input():
    """R2: Identity returns its input, bit for bit, NaN included."""
    tree = hierarchy()
    rng = np.random.default_rng(72)
    points = large(rng, (1, len(tree.nodes), 3, 4))
    points[0, 3, 2, 1] = np.nan
    forecasts = Forecasts(points, np.array(["m"]), tree.nodes, ORIGINS)
    got = reconcile(forecasts, tree, Identity())
    assert_bits(got.points, points)
    assert list(got.series) == list(tree.nodes)


def projection(summing: np.ndarray) -> np.ndarray:
    """[B, N]: (S' W^-1 S)^-1 S' W^-1, with W the number of bottoms under each node."""
    inverse = np.diag(1.0 / summing.sum(axis=1))
    return np.linalg.solve(summing.T @ inverse @ summing, summing.T @ inverse)


def test_wls_struct_projection():
    """R3: WlsStruct gives the bottoms of MinT with structural scaling, within 1e-5 times the
    scale, and its aggregates are BottomUp of its bottoms. It raises on a NaN point."""
    tree = hierarchy()
    rng = np.random.default_rng(73)
    points = large(rng, (2, len(tree.nodes), 3, 4))
    forecasts = Forecasts(points, np.array(["m1", "m2"]), tree.nodes, ORIGINS)
    got = reconcile(forecasts, tree, WlsStruct())
    summing = tree.summing.toarray().astype(np.float64)
    bottoms = np.einsum("bn,mnho->mbho", projection(summing), points.astype(np.float64))
    scale = np.abs(points).max()
    n_bottom = tree.n_bottom
    assert np.max(np.abs(got.points[:, :n_bottom] - bottoms)) <= 1e-5 * scale
    sums = np.einsum("nb,mbho->mnho", summing, got.points[:, :n_bottom].astype(np.float64))
    assert_bits(got.points, sums.astype(np.float32))
    assert not np.allclose(points, got.points)  # the input is not coherent
    broken = points.copy()
    broken[1, 4, 0, 2] = np.nan
    with pytest.raises(ValueError):
        reconcile(Forecasts(broken, forecasts.models, tree.nodes, ORIGINS), tree, WlsStruct())


@pytest.fixture(scope="module")
def nixtla(tmp_path_factory):
    """BottomUpSparse and MinTrace(wls_struct) of hierarchicalforecast, bottoms last."""
    tree = hierarchy()
    rng = np.random.default_rng(74)
    points = large(rng, (2, len(tree.nodes), 3, 4))
    n_bottom, n_nodes = tree.n_bottom, len(tree.nodes)
    order = np.r_[n_bottom:n_nodes, 0:n_bottom]
    summing = tree.summing.toarray().astype(np.float64)[order]
    y_hat = np.moveaxis(points, 1, 0).reshape(n_nodes, -1)[order].astype(np.float64)
    out = _reference.run(
        "reconcile", {"summing": summing, "y_hat": y_hat}, tmp_path_factory.mktemp("rec")
    )
    back = np.argsort(order)
    shape = (n_nodes, 2, 3, 4)
    as_points = {
        key: np.moveaxis(out[key][back].reshape(shape), 0, 1) for key in ("bottom_up", "wls_struct")
    }
    return tree, points, as_points


def test_bottom_up_equals_hierarchicalforecast(nixtla):
    """R4: on points with no NaN, BottomUp equals BottomUpSparse within 1e-5 times the scale."""
    tree, points, reference = nixtla
    bottoms = Forecasts(points[:, : tree.n_bottom].copy(), np.array(["m1", "m2"]), SERIES, ORIGINS)
    got = reconcile(bottoms, tree, BottomUp())
    scale = np.abs(points).max()
    assert np.max(np.abs(got.points - reference["bottom_up"])) <= 1e-5 * scale


def test_wls_struct_equals_hierarchicalforecast(nixtla):
    """R4: on points with no NaN, WlsStruct equals MinTrace(method="wls_struct") within 1e-5
    times the scale."""
    tree, points, reference = nixtla
    nodes = Forecasts(points, np.array(["m1", "m2"]), tree.nodes, ORIGINS)
    got = reconcile(nodes, tree, WlsStruct())
    scale = np.abs(points).max()
    assert np.max(np.abs(got.points - reference["wls_struct"])) <= 1e-5 * scale
    assert np.max(np.abs(reference["wls_struct"] - points)) > 1e-3 * scale


def test_reconcile_input_conditions():
    """R1, R2, R3: reconcile raises TypeError for a hierarchy or a method of the wrong type,
    and ValueError for series that the method does not take. Reconcilers have no fields."""
    tree = hierarchy()
    rng = np.random.default_rng(75)
    bottoms = Forecasts(large(rng, (1, 8, 2, 4)), np.array(["m"]), SERIES, ORIGINS)
    nodes = Forecasts(large(rng, (1, len(tree.nodes), 2, 4)), np.array(["m"]), tree.nodes, ORIGINS)
    shuffled = Forecasts(
        bottoms.points[:, ::-1].copy(), np.array(["m"]), SERIES[::-1].copy(), ORIGINS
    )
    assert issubclass(BottomUp, Reconciler)
    call: Any = reconcile  # the arguments below have wrong types on purpose
    for hierarchy_or_not, method_or_not in [
        (ATTRIBUTES, BottomUp()),
        (tree, BottomUp),
        (tree, "bottom_up"),
    ]:
        with pytest.raises(TypeError):
            call(bottoms, hierarchy_or_not, method_or_not)
    for forecasts, method in [
        (nodes, BottomUp()),
        (shuffled, BottomUp()),
        (bottoms, Identity()),
        (bottoms, WlsStruct()),
    ]:
        with pytest.raises(ValueError):
            reconcile(forecasts, tree, method)
    for kind in (BottomUp, Identity, WlsStruct):
        with pytest.raises(pydantic.ValidationError):
            cast(Any, kind)(weights=1)
