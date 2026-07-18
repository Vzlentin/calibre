"""Candidate projection formulations and the dense/sparse solve paths.

Every candidate is the same projection with a different weight matrix `W`::

    beta = argmin_b (y_hat - S b)' W^-1 (y_hat - S b)
         = (S' W^-1 S)^-1 S' W^-1 y_hat
    reconciled = S @ beta

- ``wls_struct``: `W = diag(S S')` — structural weights, no residuals.
- ``wls_var``: `W = diag(v)` with `v` the per-node in-sample residual
  variances — the MinT trace minimizer under a diagonal covariance model,
  i.e. shrinkage to the diagonal target at full intensity.
- ``mint_shrink``: `W = D ((1-lam) R + lam I) D` with `R` the sample
  correlation of the residuals, `D = diag(sd)`, `lam` the Schafer-Strimmer
  shrinkage intensity — dense-only; the rejected loser.
- ``mint_cov``: `lam = 0` (raw sample covariance) — rank-deficient whenever
  `T < n_nodes` and ill-conditioned on retail-sized lattices; rejected by
  name per `[REC-10]`. Implemented nowhere; a test pins why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from gate_b_proto.lattice import Lattice, structural_weights


class SolverConvergenceError(RuntimeError):
    """Report a non-converged iterative projection solve (`[REC-21]`)."""


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Carry one reconciled vector plus the facts the derived tolerance needs."""

    reconciled: np.ndarray
    condition: float
    solver_rtol: float


@dataclass(frozen=True, slots=True)
class VarianceWeights:
    """Carry floored residual-variance weights and the floor that was applied."""

    values: np.ndarray
    floor: float


@dataclass(frozen=True, slots=True)
class ShrunkCovariance:
    """Carry the shrunk covariance estimate and its shrinkage intensity."""

    matrix: np.ndarray
    intensity: float


def validate_residuals(residuals: np.ndarray, n_nodes: int) -> np.ndarray:
    """Enforce the fitted-values contract a residual-requiring strategy reads.

    The production sidecar contract (`[REC-5]`): a complete, timestamp-aligned
    ``(n_nodes, T)`` residual matrix per (origin, model); `T < 2` makes the
    variance estimator undefined and must fail loudly before anything
    reconciles, exactly like a missing or misaligned sidecar.
    """
    array = np.asarray(residuals, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != n_nodes:
        raise ValueError(f"residual matrix must have shape ({n_nodes}, T); got {array.shape}")
    if array.shape[1] < 2:
        raise ValueError("residual-variance estimation requires at least 2 periods")
    if not np.all(np.isfinite(array)):
        raise ValueError("residual matrix must be finite")
    return array


def wls_struct_weights(lattice: Lattice) -> np.ndarray:
    """Return the structural weight vector `diag(S S')`; no residuals required."""
    return structural_weights(lattice)


def wls_var_weights(residuals: np.ndarray, n_nodes: int) -> VarianceWeights:
    """Estimate diagonal MinT weights from in-sample residuals, with a floor.

    `v_i = var(e_i, ddof=1)` over the `T` aligned periods. Singular behavior:
    a node with constant residuals (a degenerate perfect fit) has zero
    variance and would make `W^-1` infinite, so variances are floored at
    ``max(v) * T * eps`` — a few rounding units of the largest variance, which
    leaves every genuinely nonzero variance untouched. The floor derivation is
    part of the formulation, not a tunable knob.
    """
    array = validate_residuals(residuals, n_nodes)
    periods = array.shape[1]
    variances = np.var(array, axis=1, ddof=1)
    floor = float(variances.max()) * periods * np.finfo(np.float64).eps
    return VarianceWeights(values=np.maximum(variances, floor), floor=floor)


def mint_shrink_covariance(residuals: np.ndarray, n_nodes: int) -> ShrunkCovariance:
    """Estimate the Schafer-Strimmer shrunk covariance of the residuals.

    Estimator, stated exactly: standardize the centered residual rows (ddof=1)
    into `Z`; sample correlation `R = Z Z' / (T - 1)`; with cross-product
    terms `w_ijt = z_it * z_jt` and their time means `w_bar_ij`, the
    shrinkage intensity toward the identity (diagonal) target is

    ``lam = sum_{i != j} (T / (T - 1)^3) * sum_t (w_ijt - w_bar_ij)^2
            / sum_{i != j} r_ij^2``, clipped to [0, 1];

    then `W = D ((1 - lam) R + lam I) D` with `D = diag(sd)` of the residuals.
    `lam = 1` degenerates to the diagonal model (`wls_var` with sample
    variances); `lam = 0` is `mint_cov`. Every intermediate is `n_nodes^2`
    dense — that is the feasibility problem, not S.
    """
    array = validate_residuals(residuals, n_nodes)
    periods = array.shape[1]
    centered = array - array.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.einsum("ij,ij->i", centered, centered) / (periods - 1))
    if np.any(sd == 0.0):
        raise ValueError("shrunk covariance is undefined for a zero-variance node")
    standardized = centered / sd[:, None]
    correlation = (standardized @ standardized.T) / (periods - 1)
    products = standardized[:, None, :] * standardized[None, :, :]
    product_means = products.mean(axis=2)
    var_r = (periods / (periods - 1) ** 3) * ((products - product_means[..., None]) ** 2).sum(
        axis=2
    )
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    denominator = float((correlation[off_diagonal] ** 2).sum())
    intensity = (
        1.0
        if denominator == 0.0
        else float(np.clip(var_r[off_diagonal].sum() / denominator, 0.0, 1.0))
    )
    shrunk = (1.0 - intensity) * correlation
    np.fill_diagonal(shrunk, 1.0)
    matrix = (sd[:, None] * shrunk) * sd[None, :]
    return ShrunkCovariance(matrix=matrix, intensity=intensity)


def project_dense(S: np.ndarray, weight_diag: np.ndarray, y_hat: np.ndarray) -> ProjectionResult:
    """Solve the projection with a dense S and a diagonal weight matrix.

    Dense workspace: the ``(n_bottom, n_bottom)`` normal matrix plus one
    factorization temporary — the arrays the preflight charges for.
    """
    normal = S.T @ (S / weight_diag[:, None])
    rhs = S.T @ (y_hat / weight_diag)
    beta = linalg.solve(normal, rhs, assume_a="pos")
    return ProjectionResult(
        reconciled=S @ beta, condition=float(np.linalg.cond(normal, 2)), solver_rtol=0.0
    )


def project_sparse(
    S_csr: sparse.csr_array,
    weight_diag: np.ndarray,
    y_hat: np.ndarray,
    *,
    rtol: float = 1e-10,
    maxiter: int | None = None,
) -> ProjectionResult:
    """Solve the same projection through a matrix-free sparse operator.

    The normal matrix ``S' W^-1 S`` is never formed: each operator action is
    ``v -> S' ((S v) / w)``, costing ``O(nnz(S))``. The solver is conjugate
    gradient (the operator is symmetric positive definite because S contains
    the identity block); any iterative solver fits the same guard shape —
    convergence status is surfaced and non-convergence raises
    :class:`SolverConvergenceError`, because the output is coherent by
    construction and a bad solve is otherwise undetectable (`[REC-21]`).
    ``maxiter`` exists so tests can starve the solver and prove the guard.
    """
    n_bottom = S_csr.shape[1]

    def matvec(v: np.ndarray) -> np.ndarray:
        return S_csr.T @ ((S_csr @ v) / weight_diag)

    operator = sparse_linalg.LinearOperator((n_bottom, n_bottom), matvec=matvec)
    rhs = S_csr.T @ (y_hat / weight_diag)
    beta, info = sparse_linalg.cg(operator, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
    if info != 0:
        reason = "no convergence within maxiter" if info > 0 else "breakdown or illegal input"
        raise SolverConvergenceError(
            f"sparse projection solve failed: exit_code={info} ({reason}). The reconciled "
            "output is coherent by construction, so this guard is the only convergence "
            "signal; production errors carry the cross-section identity (model name, "
            "origin, horizon step)."
        )
    return ProjectionResult(reconciled=S_csr @ beta, condition=np.nan, solver_rtol=rtol)


def project_dense_covariance(
    S: np.ndarray, covariance: np.ndarray, y_hat: np.ndarray
) -> ProjectionResult:
    """Solve the projection with a dense S and a full covariance matrix.

    Dense workspace: `W` itself (``n_nodes^2``), the ``(n_bottom, n_bottom)``
    normal matrix, and factorization temporaries. Exists to define the
    fixture-scale closed-form reference for `mint_shrink` — and to make the
    full-scale cost undeniable in the recorded estimate.
    """
    covariance_factor = linalg.cho_factor(covariance)
    normal = S.T @ linalg.cho_solve(covariance_factor, S)
    rhs = S.T @ linalg.cho_solve(covariance_factor, y_hat)
    beta = linalg.solve(normal, rhs, assume_a="pos")
    return ProjectionResult(
        reconciled=S @ beta, condition=float(np.linalg.cond(normal, 2)), solver_rtol=0.0
    )


def normal_equation_condition(S: np.ndarray, weight_diag: np.ndarray) -> float:
    """Return the exact 2-norm condition number of ``S' W^-1 S``.

    Affordable at fixture scale and used to evaluate the derived tolerance;
    the production engine substitutes an estimate, per the tolerance
    function's own documentation.
    """
    return float(np.linalg.cond(S.T @ (S / weight_diag[:, None]), 2))
