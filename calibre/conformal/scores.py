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
