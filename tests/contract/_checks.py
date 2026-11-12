"""Checks that several guarantees share: the cross-validation check of F1, the visibility
probe of F3 and the split check of F7.

A run is a function `run(dense, origin_idx) -> Forecasts` that issues forecasts at the
periods `origin_idx` of `dense`.
"""

import numpy as np
import pandas as pd

from calibre import Forecasts, Panel, rolling_forecasts

from ._expected import RESERVED, assert_bits, layout
from ._panels import Dense, grid, to_panel, truncated, with_values_after


def assert_equals_cross_validation(
    got: Forecasts, cv: pd.DataFrame, panel: Panel, horizon: int
) -> list[str]:
    """`got` holds the points of the cross-validation frame `cv`, bit for bit, NaN included,
    with its models, the series of `panel` and the cutoffs as origins, and equals
    `Forecasts.from_cv` of the frame. Returns the models of the frame."""
    models = [column for column in cv.columns if column not in RESERVED]
    assert_bits(got.points, layout(cv, panel.series, panel.periods, models, horizon))
    assert list(got.models) == models
    assert list(got.series) == list(panel.series)
    assert np.array_equal(got.origins.values, np.unique(cv["cutoff"].to_numpy()))
    from_cv = Forecasts.from_cv(cv, series=panel.series, horizon=horizon)
    assert_bits(got.points, from_cv.points)
    assert list(from_cv.models) == models
    return models


def probe_positions(n_origins: int) -> list[int]:
    """The second, the middle and the one-before-last origin."""
    return [1, n_origins // 2, n_origins - 2]


def rolling(dense: Dense, origin_idx: np.ndarray, *, engines, horizon: int, lookback=None):
    """A run of `rolling_forecasts`. Bind its keywords with `functools.partial`."""
    origins = dense.periods[origin_idx]
    return rolling_forecasts(to_panel(dense), engines, origins, horizon, lookback=lookback)


def assert_ignores_later_values(run, dense: Dense, origin_idx, rng) -> Forecasts:
    """No output issued at origin o changes when the values after o change to seeded noise,
    or when the data after o is removed. The noise changes the values and some output of each
    probe, so the probe can see a leak. Returns the run on `dense`."""
    origin_idx = np.asarray(origin_idx)
    base = run(dense, origin_idx)
    for position in probe_positions(len(origin_idx)):
        t = int(origin_idx[position])
        keep = np.flatnonzero(origin_idx <= t)
        noisy = with_values_after(dense, t, grid(rng, dense.values.shape))
        assert not np.array_equal(noisy.values, dense.values, equal_nan=True)
        changed = run(noisy, origin_idx)
        assert_bits(changed.points[..., keep], base.points[..., keep])
        assert not np.array_equal(changed.points, base.points, equal_nan=True)
        removed = run(truncated(dense, t), origin_idx[keep])
        assert removed.origins.equals(base.origins[keep])
        assert_bits(removed.points, base.points[..., keep])
    return base


def assert_splits(panel, engines, origins, horizon: int, lookback, rows, columns) -> Forecasts:
    """A call on `panel.take(index)` for each index of `rows` gives the rows `index` of the
    whole call, and a call on `origins[index]` for each index of `columns` gives the columns
    `index`, bit for bit. Returns the whole call."""
    whole = rolling_forecasts(panel, engines, origins, horizon, lookback=lookback)
    for index in rows:
        part = rolling_forecasts(
            panel.take(np.array(index)), engines, origins, horizon, lookback=lookback
        )
        assert_bits(part.points, whole.points[:, index])
        assert list(part.series) == list(whole.series[index])
        assert list(part.models) == list(whole.models)
        assert part.origins.equals(whole.origins)
    for index in columns:
        part = rolling_forecasts(panel, engines, origins[index], horizon, lookback=lookback)
        assert_bits(part.points, whole.points[..., index])
        assert part.origins.equals(origins[index])
    return whole
