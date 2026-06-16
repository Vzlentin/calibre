"""The :class:`CoherentDraws` spread: hierarchically-coherent interval bounds.

The second :class:`~calibre.conformal.protocols.Spread` (after
:class:`~calibre.conformal.spread.AnalyticRadius`) and the one that makes the
seam earn its keep. Rather than a scalar ``center +/- radius``, it draws a joint
predictive sample for each cross-section, reconciles every draw through the
sparse summing matrix ``S`` (so ``y_draw = S @ b_draw`` holds exactly), and
reads per-node empirical quantiles off the shared coherent draw set. Any
functional read — a per-node interval, a window-sum interval — is therefore
aggregate-consistent by construction.

Draws are an intra-producer intermediate: they never cross the
``ForecastFrame`` seam, only the reduced ``lo_*``/``hi_*`` columns do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from calibre.conformal.protocols import SpreadContext
from calibre.core.forecast_frame import (
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    H,
    Y,
    validate_fitted_values_frame,
)
from calibre.reconciliation.summing import SparseSummingMatrix

DEFAULT_DRAW_COUNT = 200

# Cap the held-out width re-anchoring: a target/raw-half-width ratio above this
# means the raw inner-quantile window is degenerate relative to the held-out
# radius (a heavy-tailed node), so scaling would explode outlier draws instead
# of faithfully re-anchoring the width. Such nodes fall back to in-sample
# geometry rather than distort the reconciled aggregate.
MAX_HELD_OUT_FACTOR = 1.0e3


@dataclass(frozen=True, slots=True)
class CoherentDraws:
    """Residual-bootstrap draws reconciled through ``S`` into coherent bounds.

    Constructed with the run-constant sparse summing matrix ``S``, the draw
    count ``B``, and a base seed. ``S`` arrives at construction (it is
    run-invariant) so the per-origin :meth:`to_interval` signature stays the
    stable :class:`~calibre.conformal.protocols.Spread` shape. Each per-origin
    call derives its own RNG from the base seed and the cross-section keys, so
    bounds reproduce byte-for-byte under resume regardless of call order.

    Attributes:
        summing: the sparse summing matrix whose ``subset``/``node_labels`` align
            reconciled draws back to frame rows; never densified.
        draw_count: ``B``, the number of bootstrap draws per cross-section.
    """

    summing: SparseSummingMatrix
    draw_count: int = DEFAULT_DRAW_COUNT

    def __post_init__(self) -> None:
        if int(self.draw_count) < 1:
            raise ValueError("draw_count must be at least 1")

    def to_interval(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        issue: np.ndarray,
        *,
        context: SpreadContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return coherent ``(lower, upper)`` bounds for the per-origin frame.

        ``centers``/``radii``/``issue`` align positionally with ``context.frame``
        rows. The marginal calibrator's ``issue`` gate is honoured unchanged —
        rows it has not made ready stay ``NaN`` — and the bounds come from the
        reconciled draw quantiles, with each bottom node's draw spread re-anchored
        to the held-out half-width in ``context.held_out_half_width`` (see
        :meth:`_fill_section`). ``radii`` is accepted for interface parity and is
        otherwise unused on this path; the held-out width threads via ``context``.
        """
        if context is None:
            raise ValueError(
                "CoherentDraws requires a per-origin SpreadContext (fitted-value "
                "sidecar + frame); none was supplied"
            )
        frame = context.frame
        n_rows = len(frame)
        lower = np.full(n_rows, np.nan, dtype=float)
        upper = np.full(n_rows, np.nan, dtype=float)
        if n_rows == 0:
            return lower, upper

        residuals = self._bottom_residuals(context)
        held_out = context.held_out_half_width
        centers = np.asarray(centers, dtype=float)
        issue = np.asarray(issue, dtype=bool)

        uids = frame[UNIQUE_ID].astype(str).to_numpy()
        models = frame[MODEL_NAME].astype(str).to_numpy()
        origins = frame[FORECAST_ORIGIN].to_numpy()
        horizons = frame[H].to_numpy()
        positions = np.arange(n_rows)

        # One coherent reconciliation per (model, origin, horizon): the bottom
        # nodes at a single forecast date are the vector S reconciles. Grouping
        # on a stable factorization keeps cross-section iteration deterministic.
        section_keys = pd.MultiIndex.from_arrays([models, origins, horizons])
        section_codes, _ = pd.factorize(section_keys, sort=False)
        for code in np.unique(section_codes):
            sel = positions[section_codes == code]
            self._fill_section(
                sel,
                uids=uids,
                models=models,
                origins=origins,
                horizons=horizons,
                centers=centers,
                issue=issue,
                residuals=residuals,
                held_out=held_out,
                base_seed=int(context.seed),
                alpha=float(context.alpha),
                lower=lower,
                upper=upper,
            )
        return lower, upper

    def _fill_section(
        self,
        sel: np.ndarray,
        *,
        uids: np.ndarray,
        models: np.ndarray,
        origins: np.ndarray,
        horizons: np.ndarray,
        centers: np.ndarray,
        issue: np.ndarray,
        residuals: dict[tuple[str, str], np.ndarray],
        held_out: dict[str, float] | None,
        base_seed: int,
        alpha: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """Reconcile one cross-section's draws and write its per-node bounds.

        The ``issue`` gate and the width come from the *same* per-partition
        marginal calibrator but answer different questions: ``issue`` is its
        readiness (rows stay ``NaN`` until it has held-out history), while the
        width is its ``(1-alpha)`` radius re-anchoring the in-sample draw spread.
        Each bottom node's deviations-from-center are scaled by
        ``held_out_half_width / raw_draw_half_width`` — both measured at the same
        ``[alpha/2, 1-alpha/2]`` endpoints :meth:`_reconcile_quantiles` reads — so
        held-out calibration owns the width while the bootstrap carries the draw
        *shape* (asymmetry, intermittency, zero-collapse). The single symmetric
        scalar scales both tails: it is a deliberate simplification, not a unit
        mismatch — the numerator is the held-out absolute-error radius and the
        denominator the in-sample bootstrap spread, different populations by
        design. Scaling is applied **before** ``S @ bottom_draws`` and quantiles
        are read **after**, so ``y_draw = S @ b_draw`` stays exact; held-out
        ownership is therefore **bottom-level** only — aggregate nodes are the
        coherent sum-of-member quantile, not an independently calibrated band.
        Degenerate or thin-history nodes fall back to factor 1.0 (unscaled).
        """
        row_by_label = {uids[i]: i for i in sel}
        present_bottom = [b for b in self.summing.bottom_ids if b in row_by_label]
        if not present_bottom:
            return
        subset = self.summing.subset(present_bottom)

        section_model = str(models[sel[0]])
        section_horizon = int(horizons[sel[0]])
        centers_b = np.array([centers[row_by_label[b]] for b in subset.bottom_ids], dtype=float)
        rng = np.random.default_rng(
            _section_seed(base_seed, section_model, origins[sel[0]], section_horizon)
        )
        draws = centers_b[:, None] + _bootstrap(
            [
                residuals.get((section_model, b), np.zeros(1, dtype=float))
                for b in subset.bottom_ids
            ],
            self.draw_count,
            rng,
        )
        keys = [f"{section_model}:h{section_horizon}:{b}" for b in subset.bottom_ids]
        draws = _rescale_to_held_out(draws, centers_b, held_out, keys, alpha)
        lo, hi = self._reconcile_quantiles(subset, draws, alpha)

        label_to_index = {label: idx for idx, label in enumerate(subset.node_labels)}
        for i in sel:
            node = uids[i]
            node_idx = label_to_index.get(node)
            if node_idx is None or not issue[i]:
                continue
            lower[i] = lo[node_idx]
            upper[i] = hi[node_idx]

    def _reconcile_quantiles(
        self, subset: SparseSummingMatrix, bottom_draws: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconcile bottom draws through ``S`` and read per-node quantiles.

        One sparse matmul: ``subset.S @ bottom_draws`` makes every aggregate row
        the exact sum of its member bottom draws, so each per-node quantile read
        off the result is aggregate-consistent. Returns ``(lower, upper)`` aligned
        positionally to ``subset.node_labels``.
        """
        coherent = subset.S @ bottom_draws
        lo, hi = np.quantile(coherent, [alpha / 2.0, 1.0 - alpha / 2.0], axis=1)
        return lo, hi

    def cumulative_interval(
        self,
        context: SpreadContext,
        *,
        present_bottom: list[str],
        centers_by_node: dict[str, float],
        alpha: float,
    ) -> dict[str, tuple[float, float]]:
        """Return per-node window-sum bounds for the cumulative protection window.

        Draws are summed over ``h <= protection_period`` per bottom node, then
        reconciled once through ``S`` so the window-sum interval is coherent at
        every node. Returns ``{node_label: (lower, upper)}`` for the present
        cross-section; the runtime keys these onto the terminal-``h`` rows. The
        window-sum deviations are re-anchored to the cumulative partition's
        held-out radius (the complete-window residual distribution) the same way
        the per-horizon path is — pre-``S``, deviations-only, with the
        degenerate/thin-history fallback — so window-sum coherence stays exact.
        """
        if context.protection_period is None:
            raise ValueError("cumulative_interval requires context.protection_period")
        residuals = self._bottom_residuals(context)
        subset = self.summing.subset(present_bottom)
        protection = int(context.protection_period)
        model = str(context.frame[MODEL_NAME].iloc[0])
        origin = context.frame[FORECAST_ORIGIN].iloc[0]

        # ``centers_by_node`` is already the window-sum point forecast
        # (sum of ``y_hat`` over ``h <= protection_period``), so the center is
        # added once; only the residual draws accumulate per horizon.
        centers_b = np.array([centers_by_node[b] for b in subset.bottom_ids], dtype=float)
        rng = np.random.default_rng(
            _cumulative_seed(int(context.seed), model, origin, subset.bottom_ids)
        )
        window = np.zeros((len(subset.bottom_ids), self.draw_count), dtype=float)
        for _ in range(protection):
            window += _bootstrap(
                [residuals.get((model, b), np.zeros(1, dtype=float)) for b in subset.bottom_ids],
                self.draw_count,
                rng,
            )
        window += centers_b[:, None]
        keys = [f"{model}:cumulative:{b}" for b in subset.bottom_ids]
        window = _rescale_to_held_out(window, centers_b, context.held_out_half_width, keys, alpha)
        lo, hi = self._reconcile_quantiles(subset, window, alpha)
        return {
            label: (float(lo[idx]), float(hi[idx])) for idx, label in enumerate(subset.node_labels)
        }

    def _bottom_residuals(self, context: SpreadContext) -> dict[tuple[str, str], np.ndarray]:
        """Per-``(model, node)`` in-sample residuals ``y - fitted_y_hat``.

        Pooling is deliberately avoided on two axes: each bottom node keeps its
        own residual vector (so per-series scale and intermittency survive into
        the reconciled aggregates), and residuals are partitioned by model so a
        multi-model run never bootstraps one model's draws from another model's
        error distribution.
        """
        fitted = context.fitted_values
        if fitted is None or fitted.empty:
            raise ValueError(
                "CoherentDraws requires the in-sample fitted-value sidecar, but "
                "none was staged for this origin."
            )
        validate_fitted_values_frame(fitted)
        keyed = pd.DataFrame(
            {
                "__model__": fitted[MODEL_NAME].astype(str).to_numpy(),
                "__uid__": fitted[UNIQUE_ID].astype(str).to_numpy(),
                "__resid__": fitted[Y].to_numpy(dtype=float)
                - fitted[FITTED_Y_HAT].to_numpy(dtype=float),
            }
        )
        residuals: dict[tuple[str, str], np.ndarray] = {}
        for (model, uid), group in keyed.groupby(["__model__", "__uid__"], sort=False):
            residuals[(str(model), str(uid))] = group["__resid__"].to_numpy()
        return residuals


def _rescale_to_held_out(
    draws: np.ndarray,
    centers_b: np.ndarray,
    held_out: dict[str, float] | None,
    keys: list[str],
    alpha: float,
) -> np.ndarray:
    """Scale each bottom node's deviations-from-center to its held-out half-width.

    ``draws`` is ``(n_bottom, B)`` of ``center + bootstrap`` samples and ``keys``
    holds the marginal partition key per bottom row. The per-node factor is
    ``held_out_half_width / raw_draw_half_width``, both at ``[alpha/2,
    1-alpha/2]``; the deviations ``draws - center`` are scaled and the center is
    added back, so the draw *shape* is preserved and only the width is re-anchored
    to the held-out radius. Returns a new array, leaving ``draws`` untouched.

    Falls back to factor 1.0 (unscaled, never ``NaN``) whenever the raw spread is
    degenerate (non-positive half-width) or the held-out radius is missing,
    ``NaN``, or non-finite — the marginal calibrator returns ``+inf`` under the
    ``higher`` rule on thin history, which degrades to today's in-sample geometry
    rather than blowing the width to infinity. It also falls back when the implied
    factor exceeds :data:`MAX_HELD_OUT_FACTOR`: a raw inner-window that small
    relative to the target means a heavy-tailed node the ``[alpha/2, 1-alpha/2]``
    half-width misrepresents, so re-anchoring would explode the node's outlier
    draws into the reconciled aggregate quantiles rather than re-scale faithfully.
    """
    if held_out is None:
        return draws
    deviations = draws - centers_b[:, None]
    lo_raw, hi_raw = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0], axis=1)
    raw_half_width = (hi_raw - lo_raw) / 2.0
    factors = np.ones(len(keys), dtype=float)
    for i, key in enumerate(keys):
        target = held_out.get(key)
        if target is None or not np.isfinite(target) or raw_half_width[i] <= 0.0:
            continue
        factor = float(target) / raw_half_width[i]
        if factor > MAX_HELD_OUT_FACTOR:
            continue
        factors[i] = factor
    return centers_b[:, None] + deviations * factors[:, None]


def _bootstrap(
    residual_vectors: list[np.ndarray],
    draw_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample each node's residual vector with replacement into ``(n, B)``.

    A node with a degenerate (all-equal, e.g. all-zero) residual vector yields a
    zero-spread row, so its interval collapses to a point — the intended
    behaviour, not an error.
    """
    rows = np.empty((len(residual_vectors), draw_count), dtype=float)
    for i, residuals in enumerate(residual_vectors):
        if residuals.size == 0:
            rows[i] = 0.0
            continue
        rows[i] = rng.choice(residuals, size=draw_count, replace=True)
    return rows


def _section_seed(base_seed: int, model: Any, origin: Any, horizon: int) -> int:
    """Derive a deterministic per-cross-section seed from the base seed + keys."""
    origin_ns = pd.Timestamp(origin).value
    digest = hashlib.blake2b(
        f"{base_seed}|{model}|{origin_ns}|{horizon}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def _cumulative_seed(base_seed: int, model: Any, origin: Any, bottom_ids: tuple[str, ...]) -> int:
    """Derive a deterministic per-``(model, origin, window)`` cumulative seed."""
    origin_ns = pd.Timestamp(origin).value
    digest = hashlib.blake2b(
        f"{base_seed}|cumulative|{model}|{origin_ns}|{'|'.join(bottom_ids)}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")
