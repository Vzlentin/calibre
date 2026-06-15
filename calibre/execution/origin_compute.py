"""Picklable Ray worker target for the fused hierarchical-interval phase.

The fused ``HierarchicalIntervals`` phase is RNG-free and a pure function of its
inputs, so it can be fanned out to a Ray worker one origin at a time. This module
holds the single module-level free function that worker runs — a free function
(not a bound method) so Ray can pickle it without dragging the whole engine
across the wire.
"""

from __future__ import annotations

import pandas as pd
from threadpoolctl import threadpool_limits

from calibre.execution.threading import thread_budget
from calibre.reconciliation.hierarchical_intervals import (
    HierarchicalIntervalContext,
    HierarchicalIntervalOptions,
    NixtlaHierarchicalIntervalPhase,
)
from calibre.reconciliation.summing import HierarchyIndex


def compute_origin_intervals(
    origin_preds: pd.DataFrame,
    hierarchy_index: HierarchyIndex,
    context: HierarchicalIntervalContext,
    options: HierarchicalIntervalOptions,
    cpu_per_task: float | None,
) -> pd.DataFrame:
    """Apply the fused hierarchical-interval phase in a Ray worker.

    Rebuilds the phase in-worker (its summing-matrix memo starts empty — a
    correct per-worker cache miss) and runs ``apply`` under the same
    ``threadpool_limits`` budget the driver's serial path uses, so the BLAS
    thread count is symmetric serial-vs-parallel and the output stays
    byte-identical (a thread-count asymmetry breaks dense-BLAS reductions).

    Args:
        origin_preds: This origin's point forecasts across the hierarchy.
        hierarchy_index: The run-constant hierarchy index (one ``ray.put`` ref
            reused by every task).
        context: Per-origin sidecar carrying the in-sample fitted values.
        options: The frozen interval options the driver phase was built with.
        cpu_per_task: Per-worker CPU budget mapped to the BLAS thread budget.
    """
    phase = NixtlaHierarchicalIntervalPhase(options)
    with threadpool_limits(limits=thread_budget(cpu_per_task)):
        return phase.apply(origin_preds, hierarchy_index, context)
