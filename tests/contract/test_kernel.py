"""K1, K2, K3, K4: the kernel models `WindowAverage` and `SeasonalNaive`."""

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest

from calibre import Forecaster, Panel, rolling_forecasts

from . import _checks, _m2
from ._expected import assert_bits, window_bounds
from ._panels import make_dense, probe_dense, to_panel
from .test_forecasts import ragged_dense, window_points

HORIZON = 9
# Origins before, at and after the late launches and the early ends of `ragged_dense`, and
# windows shorter than 7 values at the first origins of each series.
ORIGIN_IDX = np.array([3, 5, 6, 10, 19, 20, 29, 32, 33, 36, 40, 46, 49, 54, 55, 60, 69])


def kernel_points(panel: Panel, engines, lookback: int | None, origin_idx=ORIGIN_IDX):
    return rolling_forecasts(
        panel, engines, panel.periods[origin_idx], HORIZON, lookback=lookback
    ).points


def window_sizes(dense, origin_idx: np.ndarray, lookback: int | None) -> np.ndarray:
    """[S, O] the number of values that each series reads at each origin, 0 outside its span."""
    inside, first = window_bounds(dense, origin_idx, lookback)
    return np.where(inside, origin_idx - first + 1, 0)


@pytest.mark.parametrize("lookback", [None, 9])
def test_seasonal_naive_gives_the_bits_of_statsforecast(tmp_path, lookback):
    """K1, K3: on a window of at least `season_length` values, SeasonalNaive gives the points
    of statsforecast's SeasonalNaive, bit for bit. A shorter window gives NaN."""
    dense = ragged_dense(seed=61)
    panel = to_panel(dense)
    got = np.concatenate(
        [
            kernel_points(panel, [_m2.SeasonalNaive(7)], lookback),
            kernel_points(panel, [_m2.SeasonalNaive(3)], lookback),
        ]
    )
    expected = window_points(
        dense,
        ORIGIN_IDX,
        HORIZON,
        lookback,
        [("SeasonalNaive", {"season_length": 7}), ("SeasonalNaive", {"season_length": 3})],
        None,
        tmp_path,
    )
    sizes = window_sizes(dense, ORIGIN_IDX, lookback)
    for m, season in enumerate((7, 3)):
        short = (sizes < season)[:, None, :]
        expected[m] = np.where(short, np.nan, expected[m])
    assert got.dtype == np.float32
    assert_bits(got, expected)
    assert ((sizes > 0) & (sizes < 7)).any()  # windows that are short for one season only
    assert np.isfinite(got[1][((sizes >= 3) & (sizes < 7))[:, None, :].repeat(HORIZON, 1)]).all()


@pytest.mark.parametrize("lookback", [None, 9])
def test_window_average_is_within_two_ulp_of_statsforecast(tmp_path, lookback):
    """K2, K3: WindowAverage is within 2 units in the last place of statsforecast's
    WindowAverage, with NaN at the same cells, NaN on a window shorter than `window_size`."""
    dense = ragged_dense(seed=62)
    panel = to_panel(dense)
    got = np.concatenate(
        [
            kernel_points(panel, [_m2.WindowAverage(5)], lookback),
            kernel_points(panel, [_m2.WindowAverage(1)], lookback),
        ]
    )
    expected = window_points(
        dense,
        ORIGIN_IDX,
        HORIZON,
        lookback,
        [("WindowAverage", {"window_size": 5}), ("WindowAverage", {"window_size": 1})],
        None,
        tmp_path,
    )
    assert got.dtype == np.float32
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    sizes = window_sizes(dense, ORIGIN_IDX, lookback)
    assert np.isnan(got[0][((sizes > 0) & (sizes < 5))[:, None, :].repeat(HORIZON, 1)]).all()
    known = np.isfinite(expected)
    assert known.any()
    ulp = np.abs(got[known] - expected[known]) / np.spacing(np.abs(expected[known]))
    assert ulp.max() <= 2


def test_window_average_reads_its_window_in_float64():
    """K2: the mean is the float64 mean of the window values, stored as float32, and a run of
    zeros gives 0."""
    periods = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    rng = np.random.default_rng(63)
    ramp = rng.lognormal(3.0, 2.0, 30)
    values = np.concatenate([ramp, np.zeros(10)]).astype(np.float32)
    panel = Panel(
        series=np.array(["x"]),
        periods=periods,
        freq="W-MON",
        data=values,
        indptr=np.array([0, 40]),
        starts=np.array([0]),
    )
    origin_idx = np.array([10, 29, 33, 39])
    got = rolling_forecasts(panel, [_m2.WindowAverage(5)], periods[origin_idx], 2).points
    for o, t in enumerate(origin_idx):
        mean = np.float32(values[t - 4 : t + 1].astype(np.float64).mean())
        assert_bits(got[0, 0, :, o], np.full(2, mean, np.float32))
    assert got[0, 0, 0, 3] == 0.0


def test_threads_give_the_same_bits():
    """K4: WindowAverage gives the same bits for any `num_threads`."""
    rng = np.random.default_rng(64)
    periods = pd.date_range("2024-01-01", periods=50, freq="D")
    ids = tuple(f"s{i:02d}" for i in range(64))
    spans = [(int(rng.integers(0, 20)), int(rng.integers(30, 51))) for _ in ids]
    panel = to_panel(make_dense(rng, ids, periods, "D", spans))
    origins = periods[[20, 25, 29, 35, 49]]
    runs = [
        rolling_forecasts(panel, [_m2.WindowAverage(6, num_threads=n)], origins, 3).points
        for n in (1, 2, 4)
    ]
    assert np.isfinite(runs[0]).any()
    for run in runs[1:]:
        assert_bits(run, runs[0])


@pytest.mark.parametrize("lookback", [None, 10])
def test_kernel_models_ignore_later_values(lookback):
    """K3, F3: no point issued at origin o changes when a value after o changes, or when the
    data after o is removed."""
    engines = [_m2.WindowAverage(4), _m2.SeasonalNaive(7)]
    run = partial(_checks.rolling, engines=engines, horizon=5, lookback=lookback)
    origin_idx = np.array([12, 13, 19, 26, 30, 31, 40, 47, 50, 58, 66, 70])
    rng = np.random.default_rng(66)
    _checks.assert_ignores_later_values(run, probe_dense(65, 80), origin_idx, rng)


@pytest.mark.parametrize("lookback", [None, 12])
def test_kernel_models_split_by_series_and_origins(lookback):
    """K3, F7: the kernel models are local. A call on `panel.take(index)` gives the rows
    `index`, and a call on some of the origins gives their columns, bit for bit."""
    panel = to_panel(ragged_dense(seed=67))
    origins = panel.periods[[8, 9, 15, 22, 23, 30, 41, 52, 60]]
    engines = [_m2.WindowAverage(5), _m2.SeasonalNaive(7), _m2.WindowAverage(3, num_threads=2)]
    with pytest.raises(ValueError):
        rolling_forecasts(panel, engines, origins, 6)  # two models named WindowAverage
    engines = engines[:2]
    assert all(engine.local is True for engine in engines)
    rows = ([0, 1, 2, 3], [4, 5, 6, 7], [1, 4, 6], [6])
    columns = ([0, 1, 2], [3, 4, 5, 6, 7, 8], [4], [8])
    whole = _checks.assert_splits(panel, engines, origins, 6, lookback, rows, columns)
    assert list(whole.models) == ["WindowAverage", "SeasonalNaive"]
    assert np.isnan(whole.points).any() and np.isfinite(whole.points).any()


def test_kernel_models_are_forecasters():
    """K1, K2: each kernel model is a local Forecaster, named after its statsforecast model."""
    for engine, name in (
        (_m2.WindowAverage(3), "WindowAverage"),
        (_m2.SeasonalNaive(7), "SeasonalNaive"),
    ):
        assert isinstance(engine, Forecaster)
        assert engine.local is True
        assert list(engine.models) == [name]


def test_kernel_input_conditions():
    """K1, K2: `season_length`, `window_size` and `num_threads` raise TypeError when they are
    not an int, a bool included, and ValueError below 1."""
    make: Any = {
        "season_length": _m2.SeasonalNaive,
        "window_size": _m2.WindowAverage,
        "num_threads": partial(_m2.WindowAverage, 5),
    }
    for build in make.values():
        for bad in (True, 2.0, "3", None):
            with pytest.raises(TypeError):
                build(bad)
        for bad in (0, -1):
            with pytest.raises(ValueError):
                build(bad)
