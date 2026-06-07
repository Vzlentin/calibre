"""Proportion-based reconciliation strategies: bottom-up and top-down.

Both implement the :class:`~calibre.reconciliation.apply.VectorReconciler` hook
``reconcile_vector`` and are registered with the strategy registry. When a
hierarchy is supplied, the shared harness expects a node-level forecast frame:
bottom rows plus the aggregate rows required by the active summing-matrix
subset.
"""

from __future__ import annotations

import numpy as np

from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.summing import SummingMatrix


class BottomUpReconciler(VectorReconciler):
    """Bottom-up: keep the bottom forecasts, set every aggregate to its sum.

    The reconciled vector is ``S @ b`` over the bottom base entries, so each
    aggregate node equals the sum of its member bottom series and the grand total
    equals the bottom sum.
    """

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        bottom = base[: summing.n_bottom]
        return summing.S @ bottom


class TopDownReconciler(VectorReconciler):
    """Top-down: disaggregate the root forecast by bottom forecast proportions.

    The grand-total (root) base forecast is split across bottom series by each
    series' share of the summed bottom base forecasts (forecast proportions);
    aggregates are then re-summed, so the result is coherent and the reconciled
    bottom sums to the root. A zero-total cross-section degrades gracefully to an
    equal split.
    """

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        bottom = base[: summing.n_bottom]
        root = float(base[summing.total_index])
        bottom_total = float(bottom.sum())
        if bottom_total == 0.0:
            shares = np.full(summing.n_bottom, 1.0 / summing.n_bottom, dtype=np.float64)
        else:
            shares = bottom / bottom_total
        return summing.S @ (root * shares)
