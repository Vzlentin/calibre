"""Picklable Ray worker targets for the across-origin parallel harness.

Each per-origin compute target is a module-level free function (not a bound
method) so Ray can pickle it without dragging the whole engine across the wire,
and a pure function of by-value inputs so it can be fanned out one origin at a
time. Two targets live here:

* :func:`compute_origin_intervals` — the fused ``HierarchicalIntervals`` phase
  (RNG-free; rebuilds the phase in-worker).
* :func:`compute_origin_coherent` — the coherent draw+reconcile slab
  (RNG-*seeded* via :class:`~calibre.conformal.coherent_draws.CoherentDraws`'
  per-section seed, NOT RNG-free; the seed is the reproducibility guarantee, so a
  later reader must not "fix" it to be RNG-free).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calibre.conformal.coherent_draws import CoherentDraws
from calibre.conformal.protocols import SpreadContext
from calibre.execution.threading import thread_budget
from calibre.reconciliation.hierarchical_intervals import (
    HierarchicalIntervalContext,
    HierarchicalIntervalOptions,
    NixtlaHierarchicalIntervalPhase,
)
from calibre.reconciliation.summing import HierarchyIndex, SparseSummingMatrix


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


@dataclass(frozen=True, slots=True)
class CoherentOriginInputs:
    """Driver-snapshotted, by-value payload for the coherent draw+reconcile slab.

    Everything the worker needs to reproduce one origin's coherent bounds, read
    once on the driver against the live calibrator/controller/joint store and
    frozen so only data — never the stateful
    :class:`~calibre.conformal.runtime.SymmetricIntervalRuntime` — crosses the
    wire. ``centers``/``radii``/``issue`` align positionally with ``context.frame``
    rows; ``draw_count`` reconstructs the :class:`CoherentDraws` spread in-worker.
    The held-out half-width map, kappa map, base seed, and fitted-value residual
    sidecar all ride inside ``context`` (a frozen
    :class:`~calibre.conformal.protocols.SpreadContext`). Disjoint from
    :class:`~calibre.reconciliation.hierarchical_intervals.HierarchicalIntervalContext`.
    The run-constant summing matrix ``S`` is NOT carried here — it is ``ray.put``
    once and passed to the worker out-of-band so it is not re-serialized per origin.

    Attributes:
        centers: Per-row point forecasts (the ``y_hat`` column).
        radii: Per-row marginal radii (interface parity; the coherent slab reads
            width from ``context.held_out_half_width``, not these).
        issue: Per-row emission gate — rows the calibrator has not made ready
            stay ``NaN``.
        context: Frozen per-origin spread context (frame slice, alpha, fitted
            residual sidecar, base seed, held-out half-width + kappa maps).
        draw_count: ``B``, the bootstrap draw count, to rebuild the spread.
    """

    centers: np.ndarray
    radii: np.ndarray
    issue: np.ndarray
    context: SpreadContext
    draw_count: int


def compute_origin_coherent(
    inputs: CoherentOriginInputs,
    summing: SparseSummingMatrix,
    cpu_per_task: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the coherent draw+reconcile slab for one origin in a Ray worker.

    Rebuilds :class:`~calibre.conformal.coherent_draws.CoherentDraws` from the
    by-value snapshot (plus the run-constant ``summing`` matrix ``S`` passed
    out-of-band, Ray-dereferenced from a one-time ``ray.put``) and runs the pure
    slab — residual bootstrap →
    ``centers + draws`` → PRE-``S`` held-out/kappa-aware deviations-only rescale →
    ``S @ bottom_draws`` reconcile → per-node ``np.quantile`` — under the **same**
    ``threadpool_limits`` budget the serial coherent apply uses, so the dense
    ``S @ draws`` / ``np.quantile`` BLAS path is thread-symmetric serial-vs-parallel
    (a thread-count asymmetry breaks byte-identity, the same class as the
    documented cross-arch LightGBM divergence).

    RNG-*seeded* (not RNG-free): the per-``(model, origin, horizon)`` seed rides
    in ``inputs.context.seed`` and is derived inside ``CoherentDraws`` via
    ``_section_seed`` (blake2b, call-order-invariant), so bounds reproduce
    byte-for-byte regardless of worker completion order.

    Returns the per-node ``(lower, upper)`` arrays positionally aligned to
    ``inputs.context.frame`` rows; the driver writes them into the issued band.
    """
    spread = CoherentDraws(summing=summing, draw_count=inputs.draw_count)
    with threadpool_limits(limits=thread_budget(cpu_per_task)):
        return spread.to_interval(
            inputs.centers, inputs.radii, inputs.issue, context=inputs.context
        )
