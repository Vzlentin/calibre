"""Tests for the CoherentDraws spread: coherence, determinism, and reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.conformal.coherent_draws import (
    CoherentDraws,
    _bootstrap,
    _cumulative_seed,
    _section_seed,
)
from calibre.conformal.protocols import SpreadContext
from calibre.core.forecast_frame import (
    DS,
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.reconciliation.summing import (
    SparseSummingMatrix,
    build_hierarchy_index,
    sparse_summing_matrix_from_index,
)

ORIGIN = pd.Timestamp("2024-01-07")
MODEL = "SeasonalNaive"


def _hierarchy() -> pd.DataFrame:
    # Two bottom series in one shared group => nodes: A, B, group=g, total.
    return pd.DataFrame({UNIQUE_ID: ["A", "B"], "group": ["g", "g"]})


def _summing() -> SparseSummingMatrix:
    return sparse_summing_matrix_from_index(build_hierarchy_index(_hierarchy()))


def _fitted(resid_a: list[float], resid_b: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(resid_a), freq="W")
    rows = []
    for uid, residuals in (("A", resid_a), ("B", resid_b)):
        for ds, r in zip(dates, residuals, strict=True):
            # residual = y - fitted_y_hat; fix fitted at 0 so y == residual.
            rows.append({UNIQUE_ID: uid, DS: ds, Y: float(r), MODEL_NAME: MODEL, FITTED_Y_HAT: 0.0})
    return pd.DataFrame(rows)


def _frame(nodes: list[str], centers: list[float], h: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: nodes,
            DS: [ORIGIN + pd.Timedelta(weeks=h)] * len(nodes),
            Y: [np.nan] * len(nodes),
            Y_HAT: centers,
            H: [h] * len(nodes),
            FORECAST_ORIGIN: [ORIGIN] * len(nodes),
            MODEL_NAME: [MODEL] * len(nodes),
        }
    )


def _context(frame: pd.DataFrame, fitted: pd.DataFrame, *, alpha: float = 0.1, seed: int = 7):
    return SpreadContext(frame=frame, alpha=alpha, fitted_values=fitted, seed=seed)


def _reconstruct_draws(
    summing: SparseSummingMatrix,
    present_bottom: list[str],
    centers_b: dict[str, float],
    residuals: dict[str, list[float]],
    *,
    draw_count: int,
    seed: int,
    h: int = 1,
) -> np.ndarray:
    """Replay the adapter's seeded draw generation for a cross-section."""
    subset = summing.subset(present_bottom)
    rng = np.random.default_rng(_section_seed(seed, MODEL, ORIGIN, h))
    centers = np.array([centers_b[b] for b in subset.bottom_ids], dtype=float)
    return centers[:, None] + _bootstrap(
        [np.asarray(residuals[b], dtype=float) for b in subset.bottom_ids],
        draw_count,
        rng,
    )


def _reconstruct_window_draws(
    summing: SparseSummingMatrix,
    present_bottom: list[str],
    centers_b: dict[str, float],
    residuals: dict[str, list[float]],
    *,
    draw_count: int,
    seed: int,
    protection: int,
    model: str = MODEL,
    origin: pd.Timestamp = ORIGIN,
) -> np.ndarray:
    """Replay the adapter's cumulative window-sum draw generation.

    Mirrors :meth:`CoherentDraws.cumulative_interval`: ``protection`` independent
    residual draws accumulate, then the window-sum center (``centers_b``) is added
    exactly once.
    """
    subset = summing.subset(present_bottom)
    rng = np.random.default_rng(_cumulative_seed(seed, model, origin, subset.bottom_ids))
    centers = np.array([centers_b[b] for b in subset.bottom_ids], dtype=float)
    window = np.zeros((len(subset.bottom_ids), draw_count), dtype=float)
    for _ in range(protection):
        window += _bootstrap(
            [np.asarray(residuals[b], dtype=float) for b in subset.bottom_ids],
            draw_count,
            rng,
        )
    return window + centers[:, None]


def _cumulative_frame(
    origin: pd.Timestamp, *, model: str = MODEL, protection: int = 3
) -> pd.DataFrame:
    """Build a terminal-``h`` per-(model, origin) section frame for cumulative tests."""
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: [origin + pd.Timedelta(weeks=protection)] * 2,
            Y: [np.nan, np.nan],
            Y_HAT: [10.0, 20.0],
            H: [protection, protection],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: [model, model],
        }
    )


def test_every_reconciled_draw_satisfies_y_equals_s_b_exactly():
    # R3 coherence: coherent[n_bottom:] == S_agg @ draws, i.e. the aggregate rows
    # are exactly the summed member bottom draws — checked for every draw.
    summing = _summing()
    fitted = _fitted([1.0, -2.0, 3.0, -1.0], [0.5, -0.5, 2.0, -2.0])
    centers = {"A": 10.0, "B": 20.0}

    draws = _reconstruct_draws(
        summing,
        ["A", "B"],
        centers,
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=64,
        seed=7,
    )
    subset = summing.subset(["A", "B"])
    coherent = subset.S @ draws

    # Bottom block is identity: coherent bottom rows == draws.
    np.testing.assert_array_equal(coherent[: subset.n_bottom], draws)
    # Every aggregate row is the exact sum of its member bottom draws, per draw.
    for agg_idx in range(subset.n_bottom, subset.n_nodes):
        members = subset.S[[agg_idx]] @ draws
        np.testing.assert_array_equal(coherent[agg_idx], members[0])
        np.testing.assert_array_equal(coherent[agg_idx], draws.sum(axis=0))


def test_aggregate_hi_equals_quantile_of_summed_member_draws():
    # R3: an aggregate node's hi_* equals the quantile of the summed member draws
    # from the SAME draw set, not an independently-computed interval.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=128)
    fitted = _fitted([2.0, -3.0, 5.0, -4.0, 1.0], [1.0, -1.0, 4.0, -2.0, 0.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])

    lower, upper = spread.to_interval(centers, np.zeros(4), issue, context=_context(frame, fitted))

    draws = _reconstruct_draws(
        summing,
        ["A", "B"],
        {"A": 10.0, "B": 20.0},
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=128,
        seed=7,
    )
    total_draws = draws.sum(axis=0)
    expected_lo, expected_hi = np.quantile(total_draws, [0.05, 0.95])

    # Row order: A, B, group=g, __total__. The total and group share members here.
    np.testing.assert_allclose(upper[3], expected_hi)
    np.testing.assert_allclose(lower[3], expected_lo)
    np.testing.assert_allclose(upper[2], expected_hi)  # group=g has the same members


def test_happy_path_produces_finite_per_node_bounds():
    spread = CoherentDraws(summing=_summing(), draw_count=64)
    fitted = _fitted([1.0, -1.0, 2.0], [0.5, -0.5, 1.5])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    lower, upper = spread.to_interval(
        np.array([10.0, 20.0, 30.0, 30.0]),
        np.zeros(4),
        np.array([True, True, True, True]),
        context=_context(frame, fitted),
    )
    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()
    assert (upper >= lower).all()


def test_not_issued_rows_stay_nan():
    spread = CoherentDraws(summing=_summing(), draw_count=32)
    fitted = _fitted([1.0, -1.0], [0.5, -0.5])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, False, True, True])
    lower, upper = spread.to_interval(
        np.array([10.0, 20.0, 30.0, 30.0]), np.zeros(4), issue, context=_context(frame, fitted)
    )
    assert np.isnan(lower[1]) and np.isnan(upper[1])
    assert np.isfinite(lower[0]) and np.isfinite(lower[2]) and np.isfinite(lower[3])


def test_same_seed_is_byte_identical():
    # R7 determinism: same seed => byte-identical lo/hi across two runs.
    fitted = _fitted([2.0, -3.0, 5.0, -1.0], [1.0, -2.0, 3.0, -4.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])

    spread1 = CoherentDraws(summing=_summing(), draw_count=128)
    spread2 = CoherentDraws(summing=_summing(), draw_count=128)
    lo1, hi1 = spread1.to_interval(
        centers, np.zeros(4), issue, context=_context(frame, fitted, seed=42)
    )
    lo2, hi2 = spread2.to_interval(
        centers, np.zeros(4), issue, context=_context(frame, fitted, seed=42)
    )

    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)


def test_different_seed_changes_bounds():
    # A richer residual vector so the empirical 5/95 quantiles resolve to
    # different resampled values under different seeds (a tiny support saturates
    # the tails and would hide the seed dependence).
    resid_a = list(np.linspace(-10.0, 10.0, 40))
    resid_b = list(np.linspace(-6.0, 6.0, 40))
    fitted = _fitted(resid_a, resid_b)
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])
    spread = CoherentDraws(summing=_summing(), draw_count=128)
    hi_a = spread.to_interval(centers, np.zeros(4), issue, context=_context(frame, fitted, seed=1))[
        1
    ]
    hi_b = spread.to_interval(centers, np.zeros(4), issue, context=_context(frame, fitted, seed=2))[
        1
    ]
    assert not np.array_equal(hi_a, hi_b)


def test_degenerate_residuals_collapse_to_zero_width():
    # A node whose residuals are all equal => zero-spread => point interval.
    spread = CoherentDraws(summing=_summing(), draw_count=64)
    fitted = _fitted([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    lower, upper = spread.to_interval(
        np.array([10.0, 20.0, 30.0, 30.0]),
        np.zeros(4),
        np.array([True, True, True, True]),
        context=_context(frame, fitted),
    )
    np.testing.assert_allclose(lower, [10.0, 20.0, 30.0, 30.0])
    np.testing.assert_allclose(upper, [10.0, 20.0, 30.0, 30.0])


def test_partial_bottom_subset_handled():
    # Only A present at this cross-section: subset drops B, group=g collapses to A.
    spread = CoherentDraws(summing=_summing(), draw_count=64)
    fitted = _fitted([1.0, -1.0, 2.0], [0.5, -0.5, 1.5])
    frame = _frame(["A", "group=g", "__total__"], [10.0, 10.0, 10.0])
    lower, upper = spread.to_interval(
        np.array([10.0, 10.0, 10.0]),
        np.zeros(3),
        np.array([True, True, True]),
        context=_context(frame, fitted),
    )
    assert np.isfinite(lower).all()
    # With only A present, the total and group equal A's own interval.
    np.testing.assert_allclose(lower, lower[0])
    np.testing.assert_allclose(upper, upper[0])


def test_uses_sparse_csr_matrix_never_densified():
    # No densification: the construction-time S is a scipy csr_array.
    from scipy import sparse

    summing = _summing()
    assert isinstance(summing.S, sparse.csr_array)
    spread = CoherentDraws(summing=summing, draw_count=16)
    assert isinstance(spread.summing.S, sparse.csr_array)
    subset = spread.summing.subset(["A", "B"])
    assert isinstance(subset.S, sparse.csr_array)


def test_missing_fitted_sidecar_fails_fast():
    spread = CoherentDraws(summing=_summing(), draw_count=16)
    frame = _frame(["A", "B"], [10.0, 20.0])
    context = SpreadContext(frame=frame, alpha=0.1, fitted_values=None)
    with pytest.raises(ValueError, match="fitted-value sidecar"):
        spread.to_interval(
            np.array([10.0, 20.0]), np.zeros(2), np.array([True, True]), context=context
        )


def test_none_context_fails_fast():
    spread = CoherentDraws(summing=_summing(), draw_count=16)
    with pytest.raises(ValueError, match="SpreadContext"):
        spread.to_interval(np.array([1.0]), np.zeros(1), np.array([True]), context=None)


def test_empty_frame_returns_empty_bounds():
    spread = CoherentDraws(summing=_summing(), draw_count=16)
    empty = _frame([], []).iloc[0:0]
    context = SpreadContext(frame=empty, alpha=0.1, fitted_values=_fitted([1.0], [1.0]))
    lower, upper = spread.to_interval(np.array([]), np.array([]), np.array([]), context=context)
    assert lower.size == 0 and upper.size == 0


def test_draw_count_must_be_positive():
    with pytest.raises(ValueError, match="draw_count"):
        CoherentDraws(summing=_summing(), draw_count=0)


def test_cumulative_interval_is_coherent_and_centers_once():
    # The cumulative window-sum path: (1) coherence — an aggregate node's window
    # bounds equal the quantile of the summed member window-draws; (2) the
    # window-sum center (already summed over the window) is added ONCE, not once
    # per horizon (the protection-fold inflation bug).
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=256)
    fitted = _fitted([2.0, -3.0, 5.0, -4.0, 1.0], [1.0, -1.0, 4.0, -2.0, 0.0])
    protection = 3
    centers_by_node = {"A": 30.0, "B": 60.0}  # window-sum point forecasts (3 * per-h)
    context = SpreadContext(
        frame=_cumulative_frame(ORIGIN, protection=protection),
        alpha=0.1,
        fitted_values=fitted,
        seed=7,
        protection_period=protection,
    )

    bounds = spread.cumulative_interval(
        context, present_bottom=["A", "B"], centers_by_node=centers_by_node, alpha=0.1
    )

    window = _reconstruct_window_draws(
        summing,
        ["A", "B"],
        centers_by_node,
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=256,
        seed=7,
        protection=protection,
    )
    total = window.sum(axis=0)
    exp_lo, exp_hi = np.quantile(total, [0.05, 0.95])
    # Coherence: __total__ (and group=g, same members) equal the quantile of the
    # summed member window-draws.
    np.testing.assert_allclose(bounds["__total__"], (exp_lo, exp_hi))
    np.testing.assert_allclose(bounds["group=g"], (exp_lo, exp_hi))
    # Center added once: the window center is 30 + 60 = 90, so the interval
    # brackets ~90 — NOT 270 (= protection * 90, the per-horizon double-count).
    lo_total, hi_total = bounds["__total__"]
    assert lo_total <= 90.0 <= hi_total
    assert hi_total < 150.0


def test_cumulative_draws_vary_across_origins():
    # The cumulative seed must key on origin (like the per-horizon path): two
    # origins sharing the same present-bottom set must NOT draw identical windows.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=128)
    fitted = _fitted(list(np.linspace(-10.0, 10.0, 40)), list(np.linspace(-6.0, 6.0, 40)))
    centers_by_node = {"A": 30.0, "B": 60.0}

    def bounds_for(origin: pd.Timestamp) -> tuple[float, float]:
        context = SpreadContext(
            frame=_cumulative_frame(origin),
            alpha=0.1,
            fitted_values=fitted,
            seed=7,
            protection_period=3,
        )
        return spread.cumulative_interval(
            context, present_bottom=["A", "B"], centers_by_node=centers_by_node, alpha=0.1
        )["__total__"]

    assert bounds_for(pd.Timestamp("2024-01-07")) != bounds_for(pd.Timestamp("2024-02-04"))


def test_residuals_partitioned_by_model():
    # A multi-model sidecar must NOT pool residuals across models: a tight-error
    # model and a wide-error model on the same nodes get tight vs wide widths.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=256)
    dates = pd.date_range("2023-01-01", periods=4, freq="W")
    rows = []
    for model, scale in (("tight", 0.1), ("wide", 50.0)):
        for uid in ("A", "B"):
            for ds, sign in zip(dates, [1.0, -1.0, 1.0, -1.0], strict=True):
                rows.append(
                    {UNIQUE_ID: uid, DS: ds, Y: sign * scale, MODEL_NAME: model, FITTED_Y_HAT: 0.0}
                )
    fitted = pd.DataFrame(rows)

    def width_for(model: str) -> float:
        frame = pd.DataFrame(
            {
                UNIQUE_ID: ["A", "B", "group=g", "__total__"],
                DS: [ORIGIN + pd.Timedelta(weeks=1)] * 4,
                Y: [np.nan] * 4,
                Y_HAT: [10.0, 20.0, 30.0, 30.0],
                H: [1] * 4,
                FORECAST_ORIGIN: [ORIGIN] * 4,
                MODEL_NAME: [model] * 4,
            }
        )
        lo, hi = spread.to_interval(
            np.array([10.0, 20.0, 30.0, 30.0]),
            np.zeros(4),
            np.array([True, True, True, True]),
            context=_context(frame, fitted),
        )
        return float(hi[0] - lo[0])  # node A's width

    assert width_for("tight") < 1.0
    assert width_for("wide") > 50.0


def test_bounds_independent_of_row_order():
    # Per-cross-section seeds key on (model, origin, horizon), not row position,
    # so permuting the frame rows must not change any per-node bound.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=128)
    fitted = _fitted([2.0, -3.0, 5.0, -1.0], [1.0, -2.0, 3.0, -4.0])
    nodes = ["A", "B", "group=g", "__total__"]
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])

    lo1, hi1 = spread.to_interval(
        centers, np.zeros(4), issue, context=_context(_frame(nodes, list(centers)), fitted, seed=5)
    )
    order = [3, 1, 0, 2]
    pframe = _frame([nodes[i] for i in order], list(centers[order]))
    lo2, hi2 = spread.to_interval(
        centers[order], np.zeros(4), issue, context=_context(pframe, fitted, seed=5)
    )

    inv = np.argsort(order)
    np.testing.assert_array_equal(lo1, lo2[inv])
    np.testing.assert_array_equal(hi1, hi2[inv])


def test_per_node_bound_equals_quantile_of_own_draws():
    # Node->bound alignment: a bottom node's emitted [lo, hi] equals the quantile
    # of its OWN draws, and ~(1 - alpha) of those draws fall inside it.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=4000)
    resid = list(np.linspace(-20.0, 20.0, 81))
    fitted = _fitted(resid, resid)
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])

    lo, hi = spread.to_interval(
        np.array([10.0, 20.0, 30.0, 30.0]),
        np.zeros(4),
        np.array([True, True, True, True]),
        context=_context(frame, fitted, seed=11),
    )

    draws = _reconstruct_draws(
        summing,
        ["A", "B"],
        {"A": 10.0, "B": 20.0},
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=4000,
        seed=11,
    )
    a_draws = draws[0]
    exp_lo, exp_hi = np.quantile(a_draws, [0.05, 0.95])
    np.testing.assert_allclose((lo[0], hi[0]), (exp_lo, exp_hi))
    inside = float(np.mean((a_draws >= lo[0]) & (a_draws <= hi[0])))
    assert 0.85 <= inside <= 0.95


def fitted_resid(fitted: pd.DataFrame, uid: str) -> list[float]:
    """Extract a node's residual vector (y - fitted_y_hat) from the sidecar."""
    sub = fitted[fitted[UNIQUE_ID] == uid]
    return list(sub[Y].to_numpy() - sub[FITTED_Y_HAT].to_numpy())


def _held_out_context(
    frame: pd.DataFrame,
    fitted: pd.DataFrame,
    held_out: dict[str, float],
    *,
    alpha: float = 0.1,
    seed: int = 7,
) -> SpreadContext:
    return SpreadContext(
        frame=frame,
        alpha=alpha,
        fitted_values=fitted,
        seed=seed,
        held_out_half_width=held_out,
    )


def _raw_half_width(node_draws: np.ndarray, alpha: float) -> float:
    lo, hi = np.quantile(node_draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float((hi - lo) / 2.0)


def test_held_out_scaled_draws_stay_coherent():
    # Coherence under held-out scaling: each aggregate node's (lower, upper) is the
    # exact sum of its members' bounds, because scaling is applied pre-S (to bottom
    # deviations) and quantiles are read after the single S @ bottom_draws matmul.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=512)
    fitted = _fitted([2.0, -3.0, 5.0, -4.0, 1.0], [1.0, -1.0, 4.0, -2.0, 0.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])
    held_out = {"SeasonalNaive:h1:A": 9.0, "SeasonalNaive:h1:B": 2.5}

    lower, upper = spread.to_interval(
        centers, np.zeros(4), issue, context=_held_out_context(frame, fitted, held_out)
    )

    # Replay the scaled bottom draws, reconcile, and assert the emitted aggregate
    # endpoints equal the quantiles of the summed scaled member draws.
    raw = _reconstruct_draws(
        summing,
        ["A", "B"],
        {"A": 10.0, "B": 20.0},
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=512,
        seed=7,
    )
    centers_b = np.array([10.0, 20.0])
    factor_a = held_out["SeasonalNaive:h1:A"] / _raw_half_width(raw[0], 0.1)
    factor_b = held_out["SeasonalNaive:h1:B"] / _raw_half_width(raw[1], 0.1)
    scaled = centers_b[:, None] + (raw - centers_b[:, None]) * np.array([[factor_a], [factor_b]])
    total = scaled.sum(axis=0)
    exp_lo, exp_hi = np.quantile(total, [0.05, 0.95])
    # Aggregate == exact sum-of-scaled-member quantile (group=g and __total__ share
    # members here) and bottom rows track their own scaled draws.
    np.testing.assert_allclose((lower[3], upper[3]), (exp_lo, exp_hi))
    np.testing.assert_allclose((lower[2], upper[2]), (exp_lo, exp_hi))
    np.testing.assert_allclose(np.quantile(scaled[0], [0.05, 0.95]), (lower[0], upper[0]))


def test_bottom_width_tracks_held_out_not_in_sample():
    # The bottom-node width is owned by the held-out half-width, not the in-sample
    # bootstrap spread — verified both when held-out is materially wider and tighter.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=4000)
    resid = list(np.linspace(-20.0, 20.0, 81))
    fitted = _fitted(resid, resid)
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])

    raw = _reconstruct_draws(
        summing,
        ["A", "B"],
        {"A": 10.0, "B": 20.0},
        {"A": resid, "B": resid},
        draw_count=4000,
        seed=7,
    )
    raw_hw_a = _raw_half_width(raw[0], 0.1)
    raw_hw_b = _raw_half_width(raw[1], 0.1)
    # A widened well above its in-sample spread; B tightened well below it.
    held_out = {"SeasonalNaive:h1:A": raw_hw_a * 3.0, "SeasonalNaive:h1:B": raw_hw_b * 0.25}

    lower, upper = spread.to_interval(
        centers, np.zeros(4), issue, context=_held_out_context(frame, fitted, held_out)
    )

    # Each bottom node's emitted half-width equals its held-out target (the scale
    # makes the [alpha/2, 1-alpha/2] half-width match the held-out radius exactly).
    np.testing.assert_allclose((upper[0] - lower[0]) / 2.0, raw_hw_a * 3.0)
    np.testing.assert_allclose((upper[1] - lower[1]) / 2.0, raw_hw_b * 0.25)
    # And these are not the in-sample widths.
    assert (upper[0] - lower[0]) / 2.0 > raw_hw_a
    assert (upper[1] - lower[1]) / 2.0 < raw_hw_b


def test_aggregate_width_is_exact_member_sum_not_calibrated():
    # An aggregate node's bounds equal the quantile of the summed SCALED member
    # draws (coherence) — deliberately NOT a 1-alpha calibrated band on the
    # aggregate, whose independent held-out calibration is deferred (out of scope here).
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=1024)
    fitted = _fitted([2.0, -3.0, 5.0, -4.0, 1.0], [1.0, -1.0, 4.0, -2.0, 0.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])
    held_out = {"SeasonalNaive:h1:A": 7.0, "SeasonalNaive:h1:B": 3.0}

    lower, upper = spread.to_interval(
        centers, np.zeros(4), issue, context=_held_out_context(frame, fitted, held_out)
    )

    raw = _reconstruct_draws(
        summing,
        ["A", "B"],
        {"A": 10.0, "B": 20.0},
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=1024,
        seed=7,
    )
    centers_b = np.array([10.0, 20.0])
    factor_a = held_out["SeasonalNaive:h1:A"] / _raw_half_width(raw[0], 0.1)
    factor_b = held_out["SeasonalNaive:h1:B"] / _raw_half_width(raw[1], 0.1)
    scaled = centers_b[:, None] + (raw - centers_b[:, None]) * np.array([[factor_a], [factor_b]])
    exp_lo, exp_hi = np.quantile(scaled.sum(axis=0), [0.05, 0.95])
    np.testing.assert_allclose((lower[3], upper[3]), (exp_lo, exp_hi))
    # The aggregate width is strictly SMALLER than the sum of the two held-out
    # radii: a naive independent-calibration band would add them, but the coherent
    # aggregate reads the quantile of the summed (cross-node) scaled draws, which
    # is sub-additive under independent draw structure.
    naive_band = held_out["SeasonalNaive:h1:A"] + held_out["SeasonalNaive:h1:B"]
    assert (upper[3] - lower[3]) / 2.0 < naive_band


def test_cumulative_held_out_scaled_window_stays_coherent():
    # Same coherence-under-scaling property on the cumulative window-sum path,
    # keyed by the cumulative partition.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=512)
    fitted = _fitted([2.0, -3.0, 5.0, -4.0, 1.0], [1.0, -1.0, 4.0, -2.0, 0.0])
    protection = 3
    centers_by_node = {"A": 30.0, "B": 60.0}
    held_out = {"SeasonalNaive:cumulative:A": 12.0, "SeasonalNaive:cumulative:B": 4.0}
    context = SpreadContext(
        frame=_cumulative_frame(ORIGIN, protection=protection),
        alpha=0.1,
        fitted_values=fitted,
        seed=7,
        protection_period=protection,
        held_out_half_width=held_out,
    )

    bounds = spread.cumulative_interval(
        context, present_bottom=["A", "B"], centers_by_node=centers_by_node, alpha=0.1
    )

    window = _reconstruct_window_draws(
        summing,
        ["A", "B"],
        centers_by_node,
        {"A": fitted_resid(fitted, "A"), "B": fitted_resid(fitted, "B")},
        draw_count=512,
        seed=7,
        protection=protection,
    )
    centers_b = np.array([30.0, 60.0])
    factor_a = held_out["SeasonalNaive:cumulative:A"] / _raw_half_width(window[0], 0.1)
    factor_b = held_out["SeasonalNaive:cumulative:B"] / _raw_half_width(window[1], 0.1)
    scaled = centers_b[:, None] + (window - centers_b[:, None]) * np.array([[factor_a], [factor_b]])
    exp_lo, exp_hi = np.quantile(scaled.sum(axis=0), [0.05, 0.95])
    np.testing.assert_allclose(bounds["__total__"], (exp_lo, exp_hi))
    np.testing.assert_allclose(bounds["group=g"], (exp_lo, exp_hi))
    # Bottom node A tracks its own scaled window draws.
    np.testing.assert_allclose(bounds["A"], np.quantile(scaled[0], [0.05, 0.95]))


def test_degenerate_and_thin_history_fall_back_unscaled():
    # Two fallback paths to factor 1.0 (never NaN/inf): a node with zero raw spread
    # (degenerate residuals) and a node whose held-out radius is +inf (thin history)
    # or missing both keep today's in-sample geometry.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=256)
    # A has a real spread; B is degenerate (all-zero residuals).
    fitted = _fitted([3.0, -3.0, 6.0, -6.0], [0.0, 0.0, 0.0, 0.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])
    # A: thin-history +inf -> fall back unscaled. B: a finite target, but B is
    # degenerate (zero raw spread) -> also unscaled.
    held_out = {"SeasonalNaive:h1:A": np.inf, "SeasonalNaive:h1:B": 5.0}

    lower, upper = spread.to_interval(
        centers, np.zeros(4), issue, context=_held_out_context(frame, fitted, held_out)
    )

    # Compare against the unscaled (held_out=None) baseline: byte-identical.
    base_lo, base_hi = spread.to_interval(
        centers, np.zeros(4), issue, context=_context(frame, fitted)
    )
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    np.testing.assert_array_equal(lower, base_lo)
    np.testing.assert_array_equal(upper, base_hi)
    # B stayed a point (degenerate) and A kept its in-sample width.
    np.testing.assert_allclose((lower[1], upper[1]), (20.0, 20.0))


def test_missing_held_out_key_falls_back_unscaled():
    # A held-out map missing a node's key leaves that node unscaled (factor 1.0).
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=256)
    fitted = _fitted([3.0, -3.0, 6.0, -6.0], [1.0, -1.0, 2.0, -2.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])
    held_out = {"SeasonalNaive:h1:A": 9.0}  # B missing on purpose

    lower, upper = spread.to_interval(
        centers, np.zeros(4), issue, context=_held_out_context(frame, fitted, held_out)
    )
    base_lo, base_hi = spread.to_interval(
        centers, np.zeros(4), issue, context=_context(frame, fitted)
    )
    # B (missing key) matches the unscaled baseline; A (present key) does not.
    np.testing.assert_allclose((lower[1], upper[1]), (base_lo[1], base_hi[1]))
    assert not np.isclose(upper[0] - lower[0], base_hi[0] - base_lo[0])


def test_pathological_factor_falls_back_unscaled():
    # A node whose raw inner-quantile half-width is tiny-but-nonzero against a large
    # held-out target implies a factor above MAX_HELD_OUT_FACTOR; it falls back to
    # in-sample geometry rather than exploding its outlier draws into the aggregate.
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=256)
    # A: a tiny-but-nonzero spread (implied factor ~9e6). B: a normal spread.
    fitted = _fitted([1e-6, -1e-6, 1e-6, -1e-6], [3.0, -3.0, 6.0, -6.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])
    held_out = {"SeasonalNaive:h1:A": 9.0, "SeasonalNaive:h1:B": 9.0}

    lower, upper = spread.to_interval(
        centers, np.zeros(4), issue, context=_held_out_context(frame, fitted, held_out)
    )
    base_lo, base_hi = spread.to_interval(
        centers, np.zeros(4), issue, context=_context(frame, fitted)
    )
    # A (capped) keeps its tiny in-sample width — not exploded toward the target.
    np.testing.assert_allclose((lower[0], upper[0]), (base_lo[0], base_hi[0]))
    assert (upper[0] - lower[0]) < 1.0
    # B (factor 1.5, within the cap) DID engage — differs from the unscaled baseline.
    assert not np.isclose(upper[1] - lower[1], base_hi[1] - base_lo[1])


def test_held_out_none_is_byte_identical_to_unscaled():
    # held_out_half_width=None must leave the in-sample geometry byte-identical to
    # the pre-held-out path (the analytic/default-path guard at the spread level).
    summing = _summing()
    spread = CoherentDraws(summing=summing, draw_count=256)
    fitted = _fitted([2.0, -3.0, 5.0, -1.0], [1.0, -2.0, 3.0, -4.0])
    frame = _frame(["A", "B", "group=g", "__total__"], [10.0, 20.0, 30.0, 30.0])
    centers = np.array([10.0, 20.0, 30.0, 30.0])
    issue = np.array([True, True, True, True])

    none_lo, none_hi = spread.to_interval(
        centers,
        np.zeros(4),
        issue,
        context=SpreadContext(frame=frame, alpha=0.1, fitted_values=fitted, seed=7),
    )
    base_lo, base_hi = spread.to_interval(
        centers, np.zeros(4), issue, context=_context(frame, fitted)
    )
    np.testing.assert_array_equal(none_lo, base_lo)
    np.testing.assert_array_equal(none_hi, base_hi)
