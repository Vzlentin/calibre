"""Private numeric helpers shared across conformal modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias, overload

import numpy as np


def _as_scalar_score(score) -> float:
    arr = np.asarray(score, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError("Expected Score to return a scalar score")
    return float(arr[0])


def _as_1d_array(values, name: str, length: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        if length is None:
            return arr.reshape(1)
        return np.full(length, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")
    return arr.astype(float, copy=True)


def _validate_bounds(bounds) -> tuple[float, float] | None:
    if bounds is None:
        return None
    lower, upper = bounds
    if lower > upper:
        raise ValueError("alpha_bounds must satisfy lower <= upper")
    return float(lower), float(upper)


AlphaInput: TypeAlias = float | np.ndarray
AlphaBounds: TypeAlias = tuple[float, float] | None


@overload
def _clip_alpha(alpha: float, bounds: AlphaBounds) -> float: ...


@overload
def _clip_alpha(alpha: np.ndarray, bounds: AlphaBounds) -> np.ndarray: ...


def _clip_alpha(alpha: AlphaInput, bounds: AlphaBounds) -> np.ndarray | float:
    if bounds is None:
        arr = np.asarray(alpha, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return arr.astype(float, copy=True)
    lower, upper = bounds
    clipped = np.clip(np.asarray(alpha, dtype=float), lower, upper)
    if clipped.ndim == 0:
        return float(clipped)
    return clipped.astype(float, copy=True)


def _validate_quantile_rule(quantile_rule: str) -> Literal["conformal", "higher"]:
    if quantile_rule == "conformal":
        return "conformal"
    if quantile_rule == "higher":
        return "higher"
    raise ValueError("quantile_rule must be 'conformal' or 'higher'")


def _finite_sample_radius(
    scores: Iterable[float],
    alpha: float,
    default_radius: float,
    quantile_rule: Literal["conformal", "higher"] = "conformal",
) -> float:
    """Compute the (1-alpha) quantile of scores under the chosen rule."""
    scores_arr = np.asarray(list(scores), dtype=float)
    if scores_arr.size == 0:
        return float(default_radius)
    ordered = np.sort(scores_arr)
    quantile_rule = _validate_quantile_rule(quantile_rule)
    alpha = float(np.asarray(alpha, dtype=float))

    if quantile_rule == "higher":
        if alpha <= 1.0 / (ordered.size + 1):
            return float(np.inf)
        clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(np.quantile(ordered, 1.0 - clipped_alpha, method="higher"))

    clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
    rank = int(np.ceil((ordered.size + 1) * (1.0 - clipped_alpha))) - 1
    rank = min(max(rank, 0), ordered.size - 1)
    return float(ordered[rank])
