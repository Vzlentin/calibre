"""Seeded inputs of `replay`, and their expected columns, scores and score windows.

Each case holds a truth panel with late launches, early ends, censored values and ties, points
with NaN cells, a target, a score, a window and levels. The origins are daily, every second
day, weekly on daily data, irregular, and weekly on weekly data.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from conformalforecast.nonconformity import AbsoluteResidual, SignedResidual

from calibre import Forecasts, Panel, Replay, Span, Split, Step, replay

from . import _expected
from ._panels import Dense, grid, make_dense, to_panel

NAMES = [
    "daily-step-absolute",
    "weekly-origins-on-daily-step-signed",
    "irregular-span-signed",
    "weekly-span-signed",
    "every-second-day-span-absolute",
]


@dataclass(frozen=True)
class Case:
    name: str
    dense: Dense  # truth values [N, T], NaN outside each span
    origin_idx: np.ndarray  # [O] calendar indexes of the origins
    points: np.ndarray  # [1, N, H, O] float32
    target: Step | Span
    cover: np.ndarray  # [N, C, H] bool, by a direct computation
    signed: bool
    window: int
    level: float | np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.dense.ids)

    @property
    def method(self) -> Split:
        score = SignedResidual() if self.signed else AbsoluteResidual()
        return Split(score=score, window=self.window)

    @property
    def alphas(self) -> np.ndarray:
        levels = np.asarray(self.level, dtype=np.float64)
        return 1.0 - np.broadcast_to(levels, (self.n_nodes,))

    def forecasts(self) -> Forecasts:
        origins = self.dense.periods[self.origin_idx]
        return Forecasts(self.points, np.array(["m"]), np.array(self.dense.ids), origins)

    def truth(self) -> Panel:
        return to_panel(self.dense)

    def run(self, **change) -> Replay:
        case = replace(self, **change)
        return replay(
            case.forecasts(),
            case.truth(),
            target=case.target,
            method=case.method,
            level=case.level,
        )


@dataclass(frozen=True)
class Expected:
    point: np.ndarray  # [N, C, O] float32
    truth: np.ndarray  # [N, C, O] float32
    score: np.ndarray  # [N, C, O] float32
    last: np.ndarray  # [N, C] the last step of each column
    windows: list  # windows[n][c][o]: the origins of the score window, oldest first


def expected(case: Case) -> Expected:
    point, truth = _expected.columns(case.points[0], case.dense.values, case.origin_idx, case.cover)
    score = _expected.scores(point, truth, signed=case.signed)
    last = _expected.last_steps(case.cover)
    windows = _expected.members(case.origin_idx, last, case.window)
    return Expected(point, truth, score, last, windows)


def _points(rng, dense: Dense, horizon: int, origin_idx, coarse) -> np.ndarray:
    n_nodes = len(dense.ids)
    points = grid(rng, (1, n_nodes, horizon, len(origin_idx))).astype(np.float32)
    for n in coarse:
        points[0, n] = grid(rng, (horizon, len(origin_idx)), coarse=True)
    points[rng.random(points.shape) < 0.08] = np.nan
    inside = (dense.starts[:, None] <= origin_idx) & (origin_idx < dense.ends[:, None])
    points[0][np.broadcast_to(~inside[:, None, :], points.shape[1:])] = np.nan
    return points


def make(name: str, seed: int = 0, probe: bool = False) -> Case:
    """The case `name`. With `probe`, every series starts at or before the first origin, as
    the visibility probes need."""
    rng = np.random.default_rng([NAMES.index(name), seed])
    ids = ("n3", "n1", "n0", "n5", "n2", "n4")
    if name == "daily-step-absolute":
        periods = pd.date_range("2024-01-01", periods=90, freq="D")
        spans = [(0, 90), (0, 90), (30, 90), (0, 70), (10, 90), (50, 61)]
        coarse = {1}
        origin_idx = np.arange(20, 88)
        horizon, target, signed, window = 4, Step(), False, 12
        level = np.array([0.9, 0.8, 0.75, 0.5, 0.9, 0.85])
    elif name == "weekly-origins-on-daily-step-signed":
        periods = pd.date_range("2024-01-01", periods=120, freq="D")
        spans = [(0, 120), (5, 120), (0, 100), (40, 120), (0, 120), (60, 120)]
        coarse = {4}
        origin_idx = np.arange(14, 119, 7)
        horizon, target, signed, window = 10, Step(), True, 6
        level = 0.8
    elif name == "irregular-span-signed":
        periods = pd.date_range("2024-01-01", periods=100, freq="D")
        spans = [(0, 100), (0, 100), (20, 100), (0, 80), (5, 95), (0, 100)]
        coarse = {0, 5}
        origin_idx = np.sort(rng.choice(np.arange(10, 96), 40, replace=False))
        horizon, signed, window = 7, True, 15
        target = Span(first=np.array([1, 1, 2, 3, 1, 4]), last=np.array([7, 3, 2, 6, 1, 7]))
        level = np.array([1 / 1.2, 0.9, 0.7, 0.6, 0.95, 0.8])
    elif name == "weekly-span-signed":
        periods = pd.date_range("2022-01-03", periods=80, freq="W-MON")
        spans = [(0, 80), (10, 80), (0, 80), (0, 60), (30, 80), (0, 80)]
        coarse = {1, 2}
        origin_idx = np.arange(10, 79)
        horizon, target, signed, window = 3, Span(1, 3), True, 52
        level = 1 / 1.2
    elif name == "every-second-day-span-absolute":
        periods = pd.date_range("2024-05-01", periods=60, freq="D")
        spans = [(0, 60), (0, 60), (8, 60), (0, 45), (3, 60), (0, 60)]
        coarse = {3}
        origin_idx = np.arange(5, 59, 2)
        horizon, signed, window = 5, False, 8
        target = Span(first=2, last=np.array([2, 5, 3, 4, 5, 2]))
        level = np.array(0.75)
    else:
        raise AssertionError(name)
    if probe:
        spans = [(min(start, int(origin_idx[0])), end) for start, end in spans]
    freq = "W-MON" if name == "weekly-span-signed" else "D"
    dense = make_dense(rng, ids, periods, freq, spans, coarse=coarse, censored_share=0.15)
    if isinstance(target, Step):
        cover = _expected.cover_step(len(ids), horizon)
    else:
        cover = _expected.cover_span(len(ids), horizon, target.first, target.last)
    points = _points(rng, dense, horizon, origin_idx, coarse)
    return Case(name, dense, origin_idx, points, target, cover, signed, window, level)


def without_censored(case: Case) -> Case:
    return replace(case, dense=replace(case.dense, censored=None))
