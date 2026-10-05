import numpy as np

from calibre.reconcile import BottomUp, Identity, WlsStruct
from tests.synthetic import make_panel


def test_wls_struct_matches_dense_mint_and_is_coherent():
    _, hierarchy = make_panel(n_series=12, n_periods=20)
    n_nodes, n_bottom = hierarchy.summing.shape
    base = np.random.default_rng(1).normal(size=(n_nodes, 4)).astype(np.float32)

    reconcile = WlsStruct(hierarchy)
    assert reconcile.shape == (n_nodes, n_nodes)
    points = reconcile(base)
    assert points.shape == (n_nodes, 4) and points.dtype == np.float32
    # Coherent: aggregates equal the sum of their bottoms.
    np.testing.assert_allclose(points, hierarchy.summing @ points[:n_bottom], rtol=1e-5, atol=1e-5)

    # Same estimate as the textbook S (S^T W^-1 S)^-1 S^T W^-1 b with W = diag(S 1).
    summing = hierarchy.summing.toarray().astype(np.float64)
    weights_inv = np.diag(1.0 / summing.sum(axis=1))
    gain = np.linalg.solve(summing.T @ weights_inv @ summing, summing.T @ weights_inv)
    expected = summing @ gain @ base.astype(np.float64)
    np.testing.assert_allclose(points, expected, rtol=1e-5, atol=1e-5)


def test_wls_struct_leaves_coherent_points_unchanged():
    panel, hierarchy = make_panel(n_series=8, n_periods=20)
    coherent = hierarchy.summing @ panel.values[:, :3]
    np.testing.assert_allclose(WlsStruct(hierarchy)(coherent), coherent, rtol=1e-6, atol=1e-4)


def test_bottom_up_sums_and_identity_keeps_base_forecasts():
    panel, hierarchy = make_panel(n_series=8, n_periods=20)
    n_nodes, n_bottom = hierarchy.summing.shape
    bottom_up = BottomUp(hierarchy)
    assert bottom_up.shape == (n_nodes, n_bottom)
    points = bottom_up(panel.values[:, :2].astype(np.float64))
    assert points.dtype == np.float32
    assert points[-1].tolist() == panel.values[:, :2].sum(axis=0).tolist()
    base = np.arange(n_nodes * 2, dtype=np.float64).reshape(n_nodes, 2)
    identity = Identity(hierarchy)
    assert identity.shape == (n_nodes, n_nodes)
    np.testing.assert_array_equal(identity(base), base)
