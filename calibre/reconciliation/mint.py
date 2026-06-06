"""MinT (minimum-trace) reconciliation and its shrinkage covariance estimator.

MinT projects a node-level base forecast onto the coherent subspace using the
general projection ``P = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹`` (KTD4/KTD7). The OLS / identity
case (``W = I``) reduces to the literature projection ``S (SᵀS)⁻¹ Sᵀ`` exactly,
which is the DoD acceptance for R7. A lattice ``S`` can carry redundant rows, so
``numpy.linalg.pinv`` is used for the inner inverse for numerical stability.

The residual-covariance weighting for ``shrinkage`` mode is the
Wickramasuriya MinT(Shrink) / Schäfer–Strimmer estimator: the sample residual
covariance shrunk toward its diagonal. The residual matrix is instance state
built by the registry from config (fixtures now; sourcing it from backtest
residuals is a documented follow-up — until then ``shrinkage`` with no residuals
falls back to the OLS weighting).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.summing import SummingMatrix

Weighting = Literal["ols", "shrinkage"]


def mint_projection(summing_matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """General MinT projection ``P = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹`` (pinv for stability)."""
    S = np.asarray(summing_matrix, dtype=np.float64)
    W = np.asarray(weight, dtype=np.float64)
    w_inv = np.linalg.pinv(W)
    inner = np.linalg.pinv(S.T @ w_inv @ S)
    bottom_map = inner @ S.T @ w_inv  # G = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹
    return S @ bottom_map


def schafer_strimmer_shrinkage(residuals: np.ndarray) -> np.ndarray:
    """Schäfer–Strimmer shrinkage of a residual covariance toward its diagonal.

    ``residuals`` is a ``(T, n)`` matrix of node-level forecast errors. Returns an
    ``(n, n)`` covariance ``W = (1 - λ) Ŝ_sample + λ diag(Ŝ_sample)`` with the
    shrinkage intensity ``λ`` estimated from the data (Wickramasuriya MinT(Shrink)).
    """
    res = np.asarray(residuals, dtype=np.float64)
    if res.ndim != 2:
        raise ValueError("residuals must be a 2D (T, n) array")
    n_obs, n_nodes = res.shape
    if n_obs < 2:
        raise ValueError("shrinkage needs at least 2 residual observations")

    covm = (res.T @ res) / n_obs
    var_diag = np.diag(covm).copy()
    sd = np.sqrt(var_diag)
    safe_sd = np.where(sd > 0.0, sd, 1.0)
    corm = covm / np.outer(safe_sd, safe_sd)

    standardized = res / safe_sd
    sq = standardized**2
    var_corr = (1.0 / (n_obs * (n_obs - 1))) * (
        sq.T @ sq - (1.0 / n_obs) * (standardized.T @ standardized) ** 2
    )
    np.fill_diagonal(var_corr, 0.0)

    off_diag_sq = (corm - np.eye(n_nodes)) ** 2
    np.fill_diagonal(off_diag_sq, 0.0)
    denom = float(off_diag_sq.sum())
    lam = 1.0 if denom == 0.0 else float(np.clip(var_corr.sum() / denom, 0.0, 1.0))

    target = np.diag(var_diag)
    return lam * target + (1.0 - lam) * covm


class MinTReconciler(VectorReconciler):
    """MinT reconciliation with selectable residual-covariance weighting.

    ``weighting='ols'`` uses ``W = I`` (and reduces to the literature projection).
    ``weighting='shrinkage'`` uses the Schäfer–Strimmer shrunk residual covariance
    when a residual matrix is supplied; absent residuals it falls back to ``W = I``
    (documented follow-up: source residuals from backtest errors).
    """

    def __init__(self, weighting: Weighting = "ols", residuals: np.ndarray | None = None) -> None:
        if weighting not in ("ols", "shrinkage"):
            raise ValueError("weighting must be 'ols' or 'shrinkage'")
        self.weighting = weighting
        self.residuals = None if residuals is None else np.asarray(residuals, dtype=np.float64)

    def _weight_matrix(self, n_nodes: int) -> np.ndarray:
        if self.weighting == "shrinkage" and self.residuals is not None:
            if self.residuals.shape[1] != n_nodes:
                raise ValueError(
                    "residuals second axis must match the number of nodes "
                    f"({self.residuals.shape[1]} != {n_nodes})"
                )
            return schafer_strimmer_shrinkage(self.residuals)
        return np.eye(n_nodes, dtype=np.float64)

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        weight = self._weight_matrix(summing.n_nodes)
        projection = mint_projection(summing.S, weight)
        return projection @ base


def build_mint_reconciler(
    weighting: Weighting = "ols", residuals: np.ndarray | None = None
) -> MinTReconciler:
    """Registry builder for the MinT strategy."""
    return MinTReconciler(weighting=weighting, residuals=residuals)
