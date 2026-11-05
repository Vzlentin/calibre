"""C2, C3, C4, C5: the score windows, thresholds, readiness, bands and bounds of `replay`, and
the settings of `Split`."""

from typing import Any

import numpy as np
import pydantic
import pytest
from conformalforecast.nonconformity import AbsoluteResidual, QuantileScore, SignedResidual

from calibre import Forecasts, Span, Split, replay

from . import _cases, _expected, _reference
from ._expected import assert_bits


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each case with its expected values, its replay, and the thresholds that
    conformalforecast gives on the expected score windows, without calibre."""
    cases = {name: _cases.make(name) for name in _cases.NAMES}
    expected = {name: _cases.expected(case) for name, case in cases.items()}
    cells, scores, alphas = [], [], []
    for name, case in cases.items():
        exp = expected[name]
        n_nodes, n_columns, n_origins = exp.score.shape
        for n in range(n_nodes):
            for c in range(n_columns):
                for o in range(n_origins):
                    values = exp.score[n, c, exp.windows[n][c][o]]
                    scores.append(values[np.isfinite(values)])
                    alphas.append(case.alphas[n])
                    cells.append((name, n, c, o))
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in scores])])
    out = _reference.run(
        "quantile",
        {
            "scores": np.concatenate(scores).astype(np.float32),
            "offsets": offsets,
            "alphas": np.array(alphas, np.float64),
        },
        tmp_path_factory.mktemp("quantile"),
    )
    thresholds = {
        name: np.full(exp.score.shape, np.nan, np.float32) for name, exp in expected.items()
    }
    for (name, n, c, o), value in zip(cells, out["threshold"], strict=True):
        thresholds[name][n, c, o] = value
    return {
        name: (case, expected[name], case.run(), thresholds[name]) for name, case in cases.items()
    }


@pytest.mark.parametrize("name", _cases.NAMES)
def test_thresholds_equal_conformal_quantile(runs, name):
    """C2, C3: the score of each cell is conformalforecast's score of its point and truth, and
    the threshold of a ready cell equals `conformal_quantile` of the finite scores of the
    `window` latest origins known at its origin, at alpha = 1 - level, bit for bit."""
    case, exp, rep, reference = runs[name]
    assert_bits(rep.score, exp.score)
    ready = np.isfinite(reference)
    assert ready.any() and not ready.all()
    assert_bits(rep.threshold[ready], reference[ready])
    assert rep.origins.equals(case.forecasts().origins)


@pytest.mark.parametrize("name", _cases.NAMES)
def test_not_ready_is_inf(runs, name):
    """C4: a cell with fewer than ceil(1 / alpha) finite scores in its window is not ready,
    and its threshold is +inf."""
    case, exp, rep, reference = runs[name]
    counts = _expected.window_counts(exp.score, exp.windows)
    need = np.array([_expected.ceil9(1.0 / alpha) for alpha in case.alphas])
    ready = counts >= need[:, None, None]
    assert np.array_equal(ready, np.isfinite(reference))
    assert np.all(rep.threshold[~ready] == np.inf)


@pytest.mark.parametrize("name", _cases.NAMES)
def test_bands_and_bounds(runs, name):
    """C5: a band is [point - threshold, point + threshold]. A bound is point + threshold with
    -inf below. Where the point is NaN, lower and upper are NaN."""
    case, exp, rep, reference = runs[name]
    missing = np.isnan(exp.point)
    assert missing.any()
    upper = exp.point + reference
    lower = exp.point - reference if not case.signed else np.full_like(upper, -np.inf)
    lower[missing] = np.nan
    upper[missing] = np.nan
    assert_bits(rep.point, exp.point)
    assert_bits(rep.upper, upper)
    assert_bits(rep.lower, lower)


def test_missing_scores_keep_their_place(runs):
    """C2: a missing score keeps its place in the window. Taking the latest finite scores
    instead gives other thresholds on these cases."""
    differs = False
    for case, exp, rep, _ in runs.values():
        finite_windows = []
        for n in range(exp.score.shape[0]):
            rows = []
            for c in range(exp.score.shape[1]):
                cells = []
                for o in range(exp.score.shape[2]):
                    known = (
                        np.arange(exp.windows[n][c][o][-1] + 1) if len(exp.windows[n][c][o]) else []
                    )
                    known = [j for j in known if np.isfinite(exp.score[n, c, j])]
                    cells.append(np.array(known[-case.window :], dtype=int))
                rows.append(cells)
            finite_windows.append(rows)
        other = _expected.rank_thresholds(exp.score, finite_windows, case.alphas)
        differs |= not np.array_equal(other, rep.threshold)
    assert differs


def test_known_is_read_from_dates(runs):
    """C2: known is read from dates. With weekly origins on daily data, the score of step 8
    is known two origins later, not eight."""
    case, exp, rep, _ = runs["weekly-origins-on-daily-step-signed"]
    counted = []
    n_origins = len(case.origin_idx)
    for n in range(case.n_nodes):
        rows = []
        for c in range(exp.last.shape[1]):
            lag = int(exp.last[n, c])
            rows.append(
                [
                    np.arange(max(0, o - lag + 1 - case.window), max(0, o - lag + 1))
                    for o in range(n_origins)
                ]
            )
        counted.append(rows)
    by_count = _expected.rank_thresholds(exp.score, counted, case.alphas)
    by_date = _expected.rank_thresholds(exp.score, exp.windows, case.alphas)
    assert_bits(rep.threshold, by_date)
    assert not np.array_equal(by_count, by_date)


def test_levels_per_node():
    """C3: a level per node gives each node the rank of its own level, and a 0-d array is one
    level for every node."""
    case = _cases.make("daily-step-absolute", seed=1)
    exp = _cases.expected(case)
    per_node = case.run()
    assert_bits(per_node.threshold, _expected.rank_thresholds(exp.score, exp.windows, case.alphas))
    alphas = np.full(case.n_nodes, 1.0 - 0.8)
    one = _expected.rank_thresholds(exp.score, exp.windows, alphas)
    assert not np.array_equal(one, per_node.threshold)
    for level in (0.8, np.array(0.8), np.full(case.n_nodes, 0.8)):
        assert_bits(case.run(level=level).threshold, one)


def test_replay_input_conditions():
    """C2: replay raises TypeError for an input of the wrong type, and ValueError for forecasts
    of more than one model, series that are not those of truth, an origin that is not a period
    of truth, a level outside (0, 1) or of the wrong length, and a span outside the horizon."""
    case = _cases.make("irregular-span-signed")
    forecasts, truth = case.forecasts(), case.truth()
    target, method, n_nodes = case.target, case.method, case.n_nodes
    call: Any = replay  # the arguments below have wrong types or values on purpose
    wrong_types = [
        (case.points, truth, target, method, case.level),
        (forecasts, case.dense.values, target, method, case.level),
        (forecasts, truth, "span", method, 0.9),
        (forecasts, truth, target, {"window": 5}, 0.9),
        (forecasts, truth, target, method, "0.9"),
    ]
    for args in wrong_types:
        with pytest.raises(TypeError):
            call(*args[:2], target=args[2], method=args[3], level=args[4])
    two = Forecasts(
        np.concatenate([case.points, case.points]),
        np.array(["a", "b"]),
        forecasts.series,
        forecasts.origins,
    )
    reordered = Forecasts(
        case.points[:, ::-1].copy(),
        np.array(["m"]),
        forecasts.series[::-1].copy(),
        forecasts.origins,
    )
    shifted = Forecasts(
        case.points, np.array(["m"]), forecasts.series, forecasts.origins + np.timedelta64(1, "h")
    )
    wrong_values = [
        (two, target, case.level),
        (reordered, target, case.level),
        (shifted, target, case.level),
        (forecasts, target, 1.0),
        (forecasts, target, 0.0),
        (forecasts, target, np.nan),
        (forecasts, target, np.full(n_nodes, 1.5)),
        (forecasts, target, np.full(n_nodes - 1, 0.9)),
        (forecasts, Span(1, 8), 0.9),
        (forecasts, Span(0, 3), 0.9),
    ]
    for given, span, level in wrong_values:
        with pytest.raises(ValueError):
            call(given, truth, target=span, method=method, level=level)


def test_split_settings():
    """C3: Split holds a conformalforecast score and a window of at least 1, in strict mode
    with no extra field. The level is not one of its settings."""
    split = Split(score=SignedResidual(), window=3)
    assert split.window == 3
    assert isinstance(Split(score=AbsoluteResidual(), window=1).score, AbsoluteResidual)
    broken = [
        dict(score=SignedResidual(), window=0),
        dict(score=SignedResidual(), window=3.0),
        dict(score=SignedResidual(), window="3"),
        dict(score="signed", window=3),
        dict(score=QuantileScore(), window=3),
        dict(score=SignedResidual(), window=3, level=0.9),
        dict(window=3),
    ]
    make: Any = Split  # the settings below break the conditions on purpose
    for settings in broken:
        with pytest.raises(pydantic.ValidationError):
            make(**settings)
    assert issubclass(pydantic.ValidationError, ValueError)
