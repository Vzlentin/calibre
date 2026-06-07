from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
)
from calibre.reconciliation.registry import resolve_reconciler
from calibre.reconciliation.strategies import BottomUpReconciler, TopDownReconciler
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix


def _two_group_summing() -> SummingMatrix:
    # a, b in G1; c in G2. Nodes: a, b, c, grp=G1, grp=G2, __total__.
    hierarchy = pd.DataFrame({UNIQUE_ID: ["a", "b", "c"], "grp": ["G1", "G1", "G2"]})
    return build_summing_matrix(hierarchy)


def _node_vector(summing: SummingMatrix, bottom: list[float]) -> np.ndarray:
    """Build a node-level base vector with the given bottom entries (aggregates 0)."""
    vec = np.zeros(summing.n_nodes, dtype=np.float64)
    vec[: summing.n_bottom] = bottom
    return vec


def _assert_coherent(summing: SummingMatrix, vector: np.ndarray) -> None:
    bottom = vector[: summing.n_bottom]
    np.testing.assert_allclose(summing.S @ bottom, vector, rtol=1e-12, atol=1e-12)


def test_bottom_up_aggregates_equal_member_sums() -> None:
    summing = _two_group_summing()
    # Incoherent base: aggregates/root deliberately wrong; bottom-up ignores them.
    base = _node_vector(summing, [2.0, 3.0, 5.0])
    base[summing.node_labels.index("grp=G1")] = 999.0
    base[summing.total_index] = -1.0

    out = BottomUpReconciler().reconcile_vector(base, summing)

    labels = summing.node_labels
    assert out[labels.index("grp=G1")] == pytest.approx(2.0 + 3.0)
    assert out[labels.index("grp=G2")] == pytest.approx(5.0)
    assert out[summing.total_index] == pytest.approx(10.0)
    _assert_coherent(summing, out)


def test_bottom_up_leaves_bottom_block_unchanged() -> None:
    summing = _two_group_summing()
    base = _node_vector(summing, [2.0, 3.0, 5.0])
    out = BottomUpReconciler().reconcile_vector(base, summing)
    np.testing.assert_array_equal(out[: summing.n_bottom], [2.0, 3.0, 5.0])


def test_top_down_distributes_root_by_forecast_proportions() -> None:
    summing = _two_group_summing()
    # Bottom base sums to 10, but the root forecast is 20 (differs from bottom sum).
    base = _node_vector(summing, [2.0, 3.0, 5.0])
    base[summing.total_index] = 20.0

    out = TopDownReconciler().reconcile_vector(base, summing)

    reconciled_bottom = out[: summing.n_bottom]
    # shares = [0.2, 0.3, 0.5] of root 20 -> [4, 6, 10].
    np.testing.assert_allclose(reconciled_bottom, [4.0, 6.0, 10.0])
    assert reconciled_bottom.sum() == pytest.approx(20.0)
    assert out[summing.total_index] == pytest.approx(20.0)
    _assert_coherent(summing, out)


def test_top_down_proportions_sum_to_one() -> None:
    summing = _two_group_summing()
    base = _node_vector(summing, [1.0, 4.0, 5.0])
    base[summing.total_index] = 100.0
    out = TopDownReconciler().reconcile_vector(base, summing)
    reconciled_bottom = out[: summing.n_bottom]
    assert reconciled_bottom.sum() == pytest.approx(100.0)
    # Reconciled bottom is root * shares; shares implied by reconciled/root sum to 1.
    np.testing.assert_allclose((reconciled_bottom / 100.0).sum(), 1.0)


def test_top_down_zero_total_degrades_to_equal_split() -> None:
    summing = _two_group_summing()
    base = _node_vector(summing, [0.0, 0.0, 0.0])
    base[summing.total_index] = 30.0
    out = TopDownReconciler().reconcile_vector(base, summing)
    reconciled_bottom = out[: summing.n_bottom]
    np.testing.assert_allclose(reconciled_bottom, [10.0, 10.0, 10.0])
    assert reconciled_bottom.sum() == pytest.approx(30.0)
    _assert_coherent(summing, out)


def _node_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["a", "b", "c", "grp=G1", "grp=G2", "__total__"], dtype="object"),
            "ds": pd.to_datetime(["2024-01-14"] * 6),
            "y": np.array([np.nan] * 6, dtype="float64"),
            Y_HAT: np.array([2.0, 3.0, 5.0, 999.0, -1.0, 20.0], dtype="float64"),
            H: np.array([1] * 6, dtype="int64"),
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-07"] * 6),
            MODEL_NAME: pd.Series(["m"] * 6, dtype="object"),
        }
    )
    return frame


def test_bottom_up_frame_keeps_bottom_and_overwrites_aggregates() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["a", "b", "c"], "grp": ["G1", "G1", "G2"]})
    out = BottomUpReconciler()(_node_frame(), hierarchy)
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), [2.0, 3.0, 5.0, 5.0, 5.0, 10.0])


def test_top_down_frame_uses_supplied_total_to_move_bottom() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["a", "b", "c"], "grp": ["G1", "G1", "G2"]})
    out = TopDownReconciler()(_node_frame(), hierarchy)
    np.testing.assert_allclose(out[Y_HAT].to_numpy(), [4.0, 6.0, 10.0, 10.0, 10.0, 20.0])


def test_registry_resolves_bottom_up_and_top_down() -> None:
    assert isinstance(resolve_reconciler("bottom_up"), BottomUpReconciler)
    assert isinstance(resolve_reconciler("top_down"), TopDownReconciler)
