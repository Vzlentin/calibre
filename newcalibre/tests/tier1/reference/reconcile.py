"""Provide fixture-scale closed-form reconciliation references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass(frozen=True, slots=True)
class ReferenceProjection:
    """Carry a reconciled vector and its normal-system condition number."""

    reconciled: np.ndarray
    condition_number: float


def structural_weights(matrix: np.ndarray) -> np.ndarray:
    """Return structural diagonal weights from exact memberships."""
    return np.sum(matrix, axis=1, dtype=np.float64)


def variance_weights(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Return sample variances floored by the selected derived floor."""
    periods = residuals.shape[1]
    variances = np.var(residuals, axis=1, ddof=1)
    floor = float(variances.max()) * periods * np.finfo(np.float64).eps
    return np.maximum(variances, floor), floor


def diagonal_projection(
    matrix: np.ndarray,
    base_forecast: np.ndarray,
    weights: np.ndarray,
) -> ReferenceProjection:
    """Solve one diagonal-weight projection through dense normal equations."""
    normal = matrix.T @ (matrix / weights[:, None])
    right_hand = matrix.T @ (base_forecast / weights)
    bottom = linalg.solve(normal, right_hand, assume_a="pos")
    return ReferenceProjection(
        reconciled=matrix @ bottom,
        condition_number=float(np.linalg.cond(normal, 2)),
    )


def shrink_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the Schäfer-Strimmer covariance with no literal ridge."""
    periods = residuals.shape[1]
    centered = residuals - residuals.mean(axis=1, keepdims=True)
    standard_deviation = np.sqrt(np.einsum("ij,ij->i", centered, centered) / (periods - 1))
    standardized = centered / standard_deviation[:, None]
    correlation = (standardized @ standardized.T) / (periods - 1)
    products = standardized[:, None, :] * standardized[None, :, :]
    product_means = products.mean(axis=2)
    variance_correlation = (
        periods / (periods - 1) ** 3 * ((products - product_means[..., None]) ** 2).sum(axis=2)
    )
    off_diagonal = ~np.eye(len(residuals), dtype=bool)
    denominator = float((correlation[off_diagonal] ** 2).sum())
    intensity = (
        1.0
        if denominator == 0.0
        else float(
            np.clip(
                variance_correlation[off_diagonal].sum() / denominator,
                0.0,
                1.0,
            )
        )
    )
    shrunk_correlation = (1.0 - intensity) * correlation
    np.fill_diagonal(shrunk_correlation, 1.0)
    covariance = standard_deviation[:, None] * shrunk_correlation * standard_deviation[None, :]
    return covariance, intensity


def covariance_projection(
    matrix: np.ndarray,
    base_forecast: np.ndarray,
    covariance: np.ndarray,
) -> ReferenceProjection:
    """Solve one full-covariance projection through Cholesky systems."""
    factor = linalg.cho_factor(covariance)
    precision_matrix = linalg.cho_solve(factor, matrix)
    precision_forecast = linalg.cho_solve(factor, base_forecast)
    normal = matrix.T @ precision_matrix
    right_hand = matrix.T @ precision_forecast
    bottom = linalg.solve(normal, right_hand, assume_a="pos")
    return ReferenceProjection(
        reconciled=matrix @ bottom,
        condition_number=float(np.linalg.cond(normal, 2)),
    )
