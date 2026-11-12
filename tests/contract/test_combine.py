"""E1 to E7: `combine`, the models weighted by the losses known at each origin."""

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest

from calibre import Forecasts, Panel

from . import _checks, _m2
from ._expected import assert_bits
from ._panels import Dense, grid, make_dense, to_panel

FREQ = "W-MON"
PERIODS = pd.date_range("2024-01-01", periods=40, freq=FREQ)
IDS = ("s0", "s1", "s2", "s3", "s4")
# Full, late launches, an early end, and a series that ends before the last origins. Every
# series starts by the second origin, where the first visibility probe runs.
SPANS = [(0, 40), (5, 40), (0, 30), (6, 40), (2, 24)]
# Irregular origins, so the lag of a step is not a fixed number of origins.
ORIGIN_IDX = np.array([4, 6, 7, 10, 11, 12, 15, 19, 20, 22, 25, 26, 30, 33, 34, 36])
HORIZON = 4
WINDOW = 3


def truth_dense(seed: int) -> Dense:
    return make_dense(np.random.default_rng(seed), IDS, PERIODS, FREQ, SPANS, coarse={4})


def model_points(rng: np.random.Generator, dense: Dense, n_models: int = 3) -> np.ndarray:
    """[M, S, H, O] float32: models around the truth, with holes, a model of zero loss on one
    series, ties, and NaN outside the spans."""
    shape = (len(IDS), HORIZON, len(ORIGIN_IDX))
    truth = step_truth(dense.values, ORIGIN_IDX, HORIZON)
    base = np.where(np.isnan(truth), grid(rng, shape), truth)
    points = np.stack(
        [base + grid(rng, shape, coarse=True) * (m + 1) / 16 - 8.0 * m for m in range(n_models)]
    )
    points[1, 1] = np.where(np.isnan(truth[1]), points[1, 1], truth[1])  # zero loss
    points[2, 0] = np.nan  # a model with no point on a series
    points[0, 3, 2, 5] = np.nan
    points[0, 0, 1, 10] = np.nan  # a hole after known losses
    points[:, 4, :, 3] = points[0, 4, :, 3]  # ties between the models
    inside = (dense.starts[:, None] <= ORIGIN_IDX) & (dense.ends[:, None] > ORIGIN_IDX)
    points[:, ~inside[:, None, :].repeat(HORIZON, 1)] = np.nan
    return points.astype(np.float32)


def step_truth(values: np.ndarray, origin_idx: np.ndarray, horizon: int) -> np.ndarray:
    """[S, H, O] float64: the value of each series at each step, NaN outside its span."""
    n_series, n_periods = values.shape
    out = np.full((n_series, horizon, len(origin_idx)), np.nan)
    for o, t in enumerate(origin_idx):
        for h in range(1, horizon + 1):
            if t + h < n_periods:
                out[:, h - 1, o] = values[:, t + h]
    return out


def forecasts_of(points: np.ndarray, origin_idx: np.ndarray = ORIGIN_IDX, ids=IDS) -> Forecasts:
    models = np.array([f"m{m}" for m in range(len(points))])
    return Forecasts(points, models, np.array(ids), PERIODS[origin_idx])


def reference(
    points: np.ndarray, dense: Dense, origin_idx: np.ndarray, window: int, best: bool
) -> np.ndarray:
    """[1, S, H, O] float64 by loops, from the rule: the truth of a cell is read by its date."""
    n_models, n_series, horizon, n_origins = points.shape
    dates = dense.periods
    by_date = {
        (s, dates[t]): dense.values[s, t]
        for s in range(n_series)
        for t in range(len(dates))
        if np.isfinite(dense.values[s, t])
    }
    origins = dates[origin_idx]
    out = np.full((1, n_series, horizon, n_origins), np.nan)
    for s in range(n_series):
        for h in range(1, horizon + 1):
            step = h * pd.tseries.frequencies.to_offset(dense.freq)
            loss = np.full((n_models, n_origins), np.nan)
            for k, origin in enumerate(origins):
                truth = by_date.get((s, origin + step), np.nan)
                loss[:, k] = np.abs(truth - points[:, s, h - 1, k].astype(np.float64))
            for o in range(n_origins):
                has_point = ~np.isnan(points[:, s, h - 1, o])
                if not has_point.any():
                    continue
                known_origins = [k for k in range(n_origins) if origins[k] + step <= origins[o]]
                known = np.full(n_models, np.nan)
                if known_origins:
                    last = known_origins[-1]
                    for m in np.flatnonzero(has_point):
                        losses = loss[m, max(0, last - window + 1) : last + 1]
                        losses = losses[~np.isnan(losses)]
                        if len(losses):
                            known[m] = losses.mean()
                weight = np.zeros(n_models)
                have = ~np.isnan(known)
                if have.any():
                    low = known[have].min()
                    if best or low == 0:
                        weight[have] = known[have] == low
                    else:
                        weight[have] = 1.0 / known[have]
                else:
                    weight[has_point] = 1.0
                value = points[has_point, s, h - 1, o].astype(np.float64)
                out[0, s, h - 1, o] = (weight[has_point] * value).sum() / weight.sum()
    return out


def assert_float32_rounding(actual: np.ndarray, expected: np.ndarray) -> None:
    """NaN at the same cells, and each value within one float32 step of the float64 value."""
    assert actual.dtype == np.float32
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    known = ~np.isnan(expected)
    gap = np.abs(actual[known].astype(np.float64) - expected[known])
    assert (gap <= np.spacing(np.abs(expected[known]).astype(np.float32))).all()


@pytest.mark.parametrize("window", [1, 3, 20])
@pytest.mark.parametrize("best", [False, True])
def test_combine_follows_the_rule(window, best):
    """E1, E2, E3, E4: one model named `combined`, with the series and the origins of the
    forecasts. Each cell is the mean of the models, weighted by the inverse of the mean
    absolute error at the last `window` origins whose truth is known at the origin, read
    from the dates. With `best`, or when the lowest known loss is 0, the models with the
    lowest known loss take all the weight. Before a loss is known, the weights are equal. A
    model with no point is out. Checked within float32 rounding of a float64 reference."""
    dense = truth_dense(seed=71)
    points = model_points(np.random.default_rng(72), dense)
    got = _m2.combine(forecasts_of(points), to_panel(dense), window=window, best=best)
    expected = reference(points, dense, ORIGIN_IDX, window, best)
    assert list(got.models) == ["combined"]
    assert list(got.series) == list(IDS)
    assert got.origins.equals(PERIODS[ORIGIN_IDX])
    assert got.points.shape == (1, len(IDS), HORIZON, len(ORIGIN_IDX))
    assert np.isnan(expected).any()
    assert_float32_rounding(got.points, expected)


def test_weights_written_by_hand():
    """E2, E3, E4: at the first origin no loss is known, so the weights are equal. At the
    next one, the losses of step 1 are 1 and 3: the weights are 3/4 and 1/4, or 1 and 0 with
    `best`."""
    panel = Panel(
        series=np.array(["s0"]),
        periods=PERIODS,
        freq=FREQ,
        data=np.full(40, 10.0, np.float32),
        indptr=np.array([0, 40]),
        starts=np.array([0]),
    )
    points = np.array([[[[11.0, 12.0]]], [[[7.0, 16.0]]]], np.float32)
    forecasts = Forecasts(points, np.array(["a", "b"]), np.array(["s0"]), PERIODS[[0, 1]])
    weighted = _m2.combine(forecasts, panel, window=WINDOW)
    best = _m2.combine(forecasts, panel, window=WINDOW, best=True)
    assert_bits(weighted.points[0, 0, 0], np.array([9.0, 0.75 * 12.0 + 0.25 * 16.0], np.float32))
    assert_bits(best.points[0, 0, 0], np.array([9.0, 12.0], np.float32))


def test_equal_weights_before_a_loss_is_known():
    """E3: at an origin whose earlier origins have no known truth for a step, the cell is the
    plain mean of the models that have a point."""
    dense = truth_dense(seed=73)
    rng = np.random.default_rng(74)
    origin_idx = np.array([24, 26])  # two periods apart: steps 3 and 4 have no known loss
    points = grid(rng, (3, 2, HORIZON, 2)).astype(np.float32)
    points[2, 1, :, :] = np.nan
    panel = to_panel(dense).take(np.array([0, 1]))
    got = _m2.combine(forecasts_of(points, origin_idx, IDS[:2]), panel, window=WINDOW)
    mean = np.nanmean(points.astype(np.float64), axis=0)
    assert_float32_rounding(got.points[0, :, :, 0], mean[:, :, 0])
    assert_float32_rounding(got.points[0, :, 2:, 1], mean[:, 2:, 1])


def test_a_model_with_no_point_is_out():
    """E3: a model with no point anywhere changes no cell."""
    dense = truth_dense(seed=75)
    points = model_points(np.random.default_rng(76), dense)
    points[2] = np.nan
    panel = to_panel(dense)
    three = _m2.combine(forecasts_of(points), panel, window=WINDOW)
    two = _m2.combine(forecasts_of(points[:2]), panel, window=WINDOW)
    assert_float32_rounding(three.points, two.points.astype(np.float64))


def test_one_model_gives_its_points_bit_for_bit():
    """E5: with one model, the result is its points, bit for bit, named `combined`."""
    dense = truth_dense(seed=77)
    rng = np.random.default_rng(78)
    points = rng.lognormal(3.0, 2.0, (1, len(IDS), HORIZON, len(ORIGIN_IDX))).astype(np.float32)
    points[0, 2, :, -3:] = np.nan
    for best in (False, True):
        got = _m2.combine(forecasts_of(points), to_panel(dense), window=WINDOW, best=best)
        assert list(got.models) == ["combined"]
        assert_bits(got.points, points)


@pytest.mark.parametrize("best", [False, True])
def test_a_cell_reads_no_truth_after_its_origin(best):
    """E6: no cell issued at origin o changes when a value of truth after o changes, or when
    the data after o is removed, bit for bit."""
    dense = truth_dense(seed=79)
    points = model_points(np.random.default_rng(80), dense)
    run = partial(combine_at, points=points, best=best)
    rng = np.random.default_rng(81)
    _checks.assert_ignores_later_values(run, dense, ORIGIN_IDX, rng)


def combine_at(dense: Dense, origin_idx: np.ndarray, *, points: np.ndarray, best: bool):
    """`combine` of the points [M, S, H, ORIGIN_IDX] at the origins `origin_idx`, with the
    truth `dense`."""
    columns = np.searchsorted(ORIGIN_IDX, origin_idx)
    forecasts = forecasts_of(points[..., columns], origin_idx)
    return _m2.combine(forecasts, to_panel(dense), window=WINDOW, best=best)


@pytest.mark.parametrize("best", [False, True])
def test_combine_splits_by_series(best):
    """E7: a call on the forecasts of some of the series, with `truth.take` of their
    positions, gives the rows of those series in the whole call, bit for bit."""
    dense = truth_dense(seed=82)
    points = model_points(np.random.default_rng(83), dense)
    panel = to_panel(dense)
    whole = _m2.combine(forecasts_of(points), panel, window=WINDOW, best=best)
    for rows in ([0, 2], [1, 3, 4], [4], [0, 1, 2, 3]):
        index = np.array(rows)
        part = _m2.combine(
            Forecasts(
                points[:, index],
                np.array(["m0", "m1", "m2"]),
                np.array(IDS)[index],
                PERIODS[ORIGIN_IDX],
            ),
            panel.take(index),
            window=WINDOW,
            best=best,
        )
        assert_bits(part.points, whole.points[:, index])
        assert list(part.series) == [IDS[i] for i in rows]


def test_combine_input_conditions():
    """E1: combine raises TypeError for an input of the wrong type, a bool window included,
    and ValueError for a window below 1, forecast series that are not the series of truth in
    order, and an origin that is not a period of truth."""
    dense = truth_dense(seed=84)
    points = model_points(np.random.default_rng(85), dense)
    panel = to_panel(dense)
    forecasts = forecasts_of(points)
    call: Any = _m2.combine  # the arguments below have wrong types on purpose
    wrong_types = [
        ((points, panel), {"window": WINDOW}),
        ((forecasts, dense), {"window": WINDOW}),
        ((forecasts, panel), {"window": True}),
        ((forecasts, panel), {"window": 1.5}),
        ((forecasts, panel), {"window": "3"}),
        ((forecasts, panel), {"window": WINDOW, "best": 1}),
    ]
    for args, kwargs in wrong_types:
        with pytest.raises(TypeError):
            call(*args, **kwargs)
    with pytest.raises(TypeError):
        call(forecasts, panel, WINDOW)  # window is keyword-only
    swapped = Forecasts(points, forecasts.models, np.array(IDS)[[1, 0, 2, 3, 4]], forecasts.origins)
    fewer = Forecasts(points[:, :4], forecasts.models, np.array(IDS[:4]), forecasts.origins)
    late = Forecasts(
        points[..., :1], forecasts.models, forecasts.series, pd.DatetimeIndex(["2030-01-07"])
    )
    off = Forecasts(
        points[..., :1],
        forecasts.models,
        forecasts.series,
        pd.DatetimeIndex(["2024-02-06"]),  # a Tuesday, off the W-MON calendar
    )
    wrong_values = [
        (forecasts, 0),
        (forecasts, -2),
        (swapped, WINDOW),
        (fewer, WINDOW),
        (late, WINDOW),
        (off, WINDOW),
    ]
    for value, window in wrong_values:
        with pytest.raises(ValueError):
            _m2.combine(value, panel, window=window)
    assert _m2.combine(forecasts, panel, window=2).points.shape[0] == 1
