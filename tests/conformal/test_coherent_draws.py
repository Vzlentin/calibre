"""Tests for the CoherentDraws spread: coherence, determinism, and reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.conformal.coherent_draws import (
    CoherentDraws,
    _bootstrap,
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


def fitted_resid(fitted: pd.DataFrame, uid: str) -> list[float]:
    """Extract a node's residual vector (y - fitted_y_hat) from the sidecar."""
    sub = fitted[fitted[UNIQUE_ID] == uid]
    return list(sub[Y].to_numpy() - sub[FITTED_Y_HAT].to_numpy())
