"""Nonconformity score functions used to calibrate conformal radii."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AbsoluteErrorScore:
    """Absolute residual ``|y_true - y_pred|`` nonconformity score."""

    def __call__(self, y_true, y_pred, *, mask=None, weights=None) -> np.ndarray:
        score = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        if mask is not None:
            score = score[np.asarray(mask, dtype=bool)]
        if weights is not None:
            score = score * np.asarray(weights, dtype=float)
        return score


absolute_error_score = AbsoluteErrorScore()


def absolute_error(y_true, y_pred, *, mask=None, weights=None):
    """Absolute residual nonconformity score."""
    return absolute_error_score(y_true, y_pred, mask=mask, weights=weights)


def scaled_absolute_error(y_true, y_pred, scale, eps: float = 1e-8, *, mask=None, weights=None):
    """Scale-invariant absolute residual score (``scale`` floored at ``eps``).

    Takes three positional arguments and cannot be used directly as a two-arg
    ``score(y_true, y_pred)`` callable.  Wrap with ``functools.partial`` to fix
    the scale::

        import functools
        score = functools.partial(scaled_absolute_error, scale=demand_mean)
    """
    denom = np.maximum(np.asarray(scale, dtype=float), eps)
    score = absolute_error_score(y_true, y_pred, mask=mask, weights=weights) / denom
    return np.asarray(score, dtype=float)
