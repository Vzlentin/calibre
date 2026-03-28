from __future__ import annotations

import numpy as np


def absolute_error(y_true, y_pred):
    """Absolute residual nonconformity score."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


def scaled_absolute_error(y_true, y_pred, scale, eps: float = 1e-8):
    """Scale-invariant absolute residual score.

    Takes three positional arguments and cannot be used directly as the
    ``score_fn`` parameter of ACI controllers (which call
    ``score_fn(y_true, y_pred)`` with two arguments).  Wrap with
    ``functools.partial`` to fix the scale::

        import functools
        score_fn = functools.partial(scaled_absolute_error, scale=demand_mean)
        aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05, score_fn=score_fn)
    """
    denom = np.maximum(np.asarray(scale, dtype=float), eps)
    return absolute_error(y_true, y_pred) / denom
