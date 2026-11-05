"""F3, C1: no output issued at origin o changes when the data after o changes or is removed.

The probes run at three origins: the second, the middle one and the one before the last. One
probe replaces every value after the origin, with seeded noise for points and with 1e6 for
truth. A second probe removes the data after the origin. Outputs are compared by origin label.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from calibre import rolling_forecasts

from . import _cases, _expected
from ._engines import engine
from ._expected import assert_bits
from ._panels import grid, make_dense, to_panel, truncated, with_values_after

MODELS = [
    ("Naive", {}),
    ("SeasonalNaive", {"season_length": 7}),
    ("WindowAverage", {"window_size": 4}),
    ("HistoricAverage", {}),
]
WITH_FALLBACK = [("AutoETS", {"model": "MNN", "alias": "ets_mnn"})]


def probe_positions(n_origins: int) -> list[int]:
    return [1, n_origins // 2, n_origins - 2]


def panel_dense():
    """Every series starts before the first probe origin. Some end early."""
    periods = pd.date_range("2024-01-01", periods=80, freq="D")
    ids = ("p2", "p0", "p1", "p3", "p5", "p4")
    spans = [(0, 80), (2, 80), (0, 60), (5, 80), (0, 35), (1, 80)]
    return make_dense(np.random.default_rng(41), ids, periods, "D", spans, coarse={2})


def at_or_before(origins: pd.DatetimeIndex, origin) -> np.ndarray:
    return np.flatnonzero(origins <= origin)


@pytest.mark.parametrize("lookback", [None, 10])
def test_points_ignore_later_values(lookback):
    """F3: no point issued at origin o changes when a value after o changes, or when the data
    after o is removed."""
    dense = panel_dense()
    origin_idx = np.array([12, 13, 19, 26, 30, 31, 40, 47, 50, 58, 66, 70])
    origins = dense.periods[origin_idx]
    engines = [engine(MODELS, "D"), engine(WITH_FALLBACK, "D", ("Naive", {}))]
    base = rolling_forecasts(to_panel(dense), engines, origins, 5, lookback=lookback)
    rng = np.random.default_rng(42)
    for position in probe_positions(len(origins)):
        origin, t = origins[position], int(origin_idx[position])
        keep = at_or_before(origins, origin)
        noisy = with_values_after(dense, t, grid(rng, dense.values.shape))
        changed = rolling_forecasts(to_panel(noisy), engines, origins, 5, lookback=lookback)
        assert not np.array_equal(noisy.values, dense.values, equal_nan=True)
        assert_bits(changed.points[..., keep], base.points[..., keep])
        cut = truncated(dense, t)
        removed = rolling_forecasts(to_panel(cut), engines, origins[keep], 5, lookback=lookback)
        assert removed.origins.equals(origins[keep])
        assert_bits(removed.points, base.points[..., keep])


def first_ready(case) -> int:
    """The first origin with a ready cell, from the expected score windows."""
    exp = _cases.expected(case)
    counts = _expected.window_counts(exp.score, exp.windows)
    need = np.array([_expected.ceil9(1.0 / alpha) for alpha in case.alphas])
    ready = (counts >= need[:, None, None]).any(axis=(0, 1))
    return int(np.flatnonzero(ready)[0])


@pytest.mark.parametrize("name", _cases.NAMES)
def test_calibration_ignores_later_truth(name):
    """C1: no point, threshold, lower or upper issued at origin o changes when a value of
    truth after o changes, or when the data after o is removed."""
    case = _cases.make(name, seed=2, probe=True)
    base = case.run()
    origins = base.origins
    assert (case.dense.starts < case.origin_idx[1]).all()
    positions = sorted({*probe_positions(len(origins)), first_ready(case)})
    assert np.isfinite(base.threshold[..., first_ready(case)]).any()
    for position in positions:
        origin, t = origins[position], int(case.origin_idx[position])
        keep = at_or_before(origins, origin)
        flooded = case.run(dense=with_values_after(case.dense, t, 1e6))
        cut = truncated(case.dense, t)
        short = replace(
            case,
            dense=cut,
            points=case.points[..., keep],
            origin_idx=case.origin_idx[keep],
        )
        removed = short.run()
        assert removed.origins.equals(origins[keep])
        for field in ("point", "threshold", "lower", "upper"):
            assert_bits(getattr(flooded, field)[..., keep], getattr(base, field)[..., keep])
            assert_bits(getattr(removed, field), getattr(base, field)[..., keep])
    assert np.isfinite(base.threshold[..., keep]).any()


def test_probe_changes_truth():
    """C1: the flood probe changes the truth of the cells after the origin, so the probe can
    see a leak."""
    case = _cases.make("daily-step-absolute", seed=2, probe=True)
    base = case.run()
    t = int(case.origin_idx[len(case.origin_idx) // 2])
    flooded = case.run(dense=with_values_after(case.dense, t, 1e6))
    assert not np.array_equal(flooded.truth, base.truth, equal_nan=True)
    assert not np.array_equal(flooded.threshold, base.threshold)
