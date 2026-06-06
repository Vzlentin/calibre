from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import UNIQUE_ID
from calibre.reconciliation.mint import (
    MinTReconciler,
    build_mint_reconciler,
    mint_projection,
    ols_projection,
    schafer_strimmer_shrinkage,
)
from calibre.reconciliation.registry import resolve_reconciler
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix


def _two_group_summing() -> SummingMatrix:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["a", "b", "c"], "grp": ["G1", "G1", "G2"]})
    return build_summing_matrix(hierarchy)


def _dense_full_rank_S(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(10, 4))


def _candidate_summing_matrices() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [
        _dense_full_rank_S(rng),
        _two_group_summing().S,  # lattice S with a redundant (row-rank-deficient) row
        build_summing_matrix(pd.DataFrame({UNIQUE_ID: ["only"], "g": ["G"]})).S,  # [[1],[1],[1]]
    ]


def test_ols_case_reproduces_literature_projection_exactly() -> None:
    """R7/KTD7: mint_projection(S, I) == S(SᵀS)⁻¹Sᵀ within tight tolerance."""
    for S in _candidate_summing_matrices():
        eye = np.eye(S.shape[0])
        np.testing.assert_allclose(
            mint_projection(S, eye), ols_projection(S), rtol=1e-10, atol=1e-12
        )


def test_projection_is_idempotent_for_ols_and_shrinkage() -> None:
    summing = _two_group_summing()
    rng = np.random.default_rng(1)
    residuals = rng.normal(size=(50, summing.n_nodes))
    weights = {
        "ols": np.eye(summing.n_nodes),
        "shrinkage": schafer_strimmer_shrinkage(residuals),
    }
    for weight in weights.values():
        projection = mint_projection(summing.S, weight)
        np.testing.assert_allclose(projection @ projection, projection, rtol=1e-9, atol=1e-10)


def test_mint_output_is_coherent() -> None:
    summing = _two_group_summing()
    base = np.array([2.0, 7.0, 5.0, 100.0, 3.0, -4.0])  # intentionally incoherent
    out = MinTReconciler(weighting="ols").reconcile_vector(base, summing)
    bottom = out[: summing.n_bottom]
    np.testing.assert_allclose(summing.S @ bottom, out, rtol=1e-10, atol=1e-12)


def test_diagonal_weight_reproduces_wls_and_differs_from_ols() -> None:
    summing = _two_group_summing()
    diag_weight = np.diag([1.0, 5.0, 0.25, 2.0, 3.0, 0.5])
    wls = mint_projection(summing.S, diag_weight)
    ols = ols_projection(summing.S)
    assert not np.allclose(wls, ols)
    # WLS is still a valid projection onto the coherent subspace.
    np.testing.assert_allclose(wls @ wls, wls, rtol=1e-9, atol=1e-10)


def test_ols_reconciler_matches_ols_projection() -> None:
    summing = _two_group_summing()
    base = np.array([1.0, 2.0, 3.0, 9.0, 9.0, 9.0])
    out = MinTReconciler(weighting="ols").reconcile_vector(base, summing)
    np.testing.assert_allclose(out, ols_projection(summing.S) @ base, rtol=1e-10, atol=1e-12)


def test_shrinkage_on_near_singular_covariance_is_invertible() -> None:
    rng = np.random.default_rng(2)
    # Highly collinear residuals -> near-singular sample covariance.
    base_signal = rng.normal(size=(40, 1))
    residuals = base_signal @ np.ones((1, 4)) + 1e-6 * rng.normal(size=(40, 4))
    weight = schafer_strimmer_shrinkage(residuals)
    assert weight.shape == (4, 4)
    np.testing.assert_allclose(weight, weight.T, rtol=1e-12, atol=1e-12)
    # Shrunk toward the diagonal -> invertible despite collinear sample covariance.
    assert abs(float(np.linalg.det(weight))) > 0.0
    np.linalg.inv(weight)  # does not raise


def test_shrinkage_lambda_is_one_when_residuals_are_uncorrelated() -> None:
    rng = np.random.default_rng(3)
    # Independent columns with distinct variances -> off-diagonal correlations ~0.
    residuals = rng.normal(size=(2000, 3)) * np.array([1.0, 3.0, 7.0])
    weight = schafer_strimmer_shrinkage(residuals)
    # As lambda -> 1, W approaches the diagonal of the sample covariance.
    off_diag = weight - np.diag(np.diag(weight))
    assert np.abs(off_diag).max() < 1e-1
    # Diagonal still carries the (distinct) per-node variances, not collapsed to one.
    assert np.diag(weight)[2] > np.diag(weight)[0]


def test_shrinkage_requires_at_least_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2 residual observations"):
        schafer_strimmer_shrinkage(np.ones((1, 3)))


def test_registry_resolves_mint_with_weighting_knob() -> None:
    default = resolve_reconciler("mint")
    assert isinstance(default, MinTReconciler)
    assert default.weighting == "ols"

    shrink = resolve_reconciler("mint", weighting="shrinkage")
    assert isinstance(shrink, MinTReconciler)
    assert shrink.weighting == "shrinkage"


def test_build_mint_reconciler_rejects_unknown_weighting() -> None:
    with pytest.raises(ValueError, match="weighting must be"):
        build_mint_reconciler(weighting="bogus")  # type: ignore[arg-type]
