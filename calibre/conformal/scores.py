from __future__ import annotations

import numpy as np


def absolute_error(y_true, y_pred):
    """Absolute residual nonconformity score."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


def scaled_absolute_error(y_true, y_pred, scale, eps: float = 1e-8):
    """Scale-invariant absolute residual score."""
    denom = np.maximum(np.asarray(scale, dtype=float), eps)
    return absolute_error(y_true, y_pred) / denom
