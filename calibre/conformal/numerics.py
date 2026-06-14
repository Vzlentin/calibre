"""Numeric helpers shared across conformal modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast, overload

import numpy as np

ArrayLike = float | Iterable[float] | np.ndarray


def as_scalar_score(score: ArrayLike) -> float:
    """Coerce a score to a single float, rejecting non-scalar inputs."""
    arr = np.asarray(score, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError("Expected Score to return a scalar score")
    return float(arr[0])


def as_1d_array(values: ArrayLike, name: str, length: int | None = None) -> np.ndarray:
    """Coerce ``values`` to a 1D float array, broadcasting scalars to ``length``.

    Args:
        values: A scalar or 1D array-like of floats.
        name: Label used in the raised error messages.
        length: Required length; a scalar is repeated to fill it, and a 1D
            input of the wrong length is rejected.

    Raises:
        ValueError: If ``values`` is not 1D, or its length does not match
            ``length``.
    """
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


def validate_bounds(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    """Validate and float-cast an ``(lower, upper)`` alpha-bounds pair.

    Returns ``None`` unchanged; raises if ``lower > upper``.
    """
    if bounds is None:
        return None
    lower, upper = bounds
    if lower > upper:
        raise ValueError("alpha_bounds must satisfy lower <= upper")
    return float(lower), float(upper)


@overload
def clip_alpha(alpha: float, bounds: tuple[float, float] | None) -> float: ...
@overload
def clip_alpha(alpha: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray: ...
def clip_alpha(alpha: float | np.ndarray, bounds: tuple[float, float] | None) -> float | np.ndarray:
    """Clip ``alpha`` into ``bounds``, preserving scalar-vs-array shape.

    With ``bounds=None`` the value passes through unchanged (still float-cast).
    """
    if bounds is None:
        arr = np.asarray(alpha, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return arr.astype(float, copy=True)
    lower, upper = bounds
    clipped = np.clip(alpha, lower, upper)
    if np.ndim(clipped) == 0:
        return float(clipped)
    return clipped


def validate_quantile_rule(quantile_rule: str) -> Literal["conformal", "higher"]:
    """Validate that ``quantile_rule`` is ``"conformal"`` or ``"higher"``."""
    if quantile_rule not in {"conformal", "higher"}:
        raise ValueError("quantile_rule must be 'conformal' or 'higher'")
    return cast(Literal["conformal", "higher"], quantile_rule)


def finite_sample_radius(
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
    quantile_rule = validate_quantile_rule(quantile_rule)
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
