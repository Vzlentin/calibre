"""T1, T2: the columns of `Step` and `Span`, and the point and truth of each column."""

import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from calibre import Span, Step, Target

from . import _cases
from ._expected import assert_bits


def test_step_cover():
    """T1: Step has H columns, and column c covers step c + 1."""
    cover = Step().cover(5, 4)
    assert isinstance(Step(), Target)
    assert cover.dtype == np.bool_
    assert cover.shape == (5, 4, 4)
    for c in range(4):
        expected = np.zeros(4, bool)
        expected[c] = True
        assert np.array_equal(cover[:, c, :], np.broadcast_to(expected, (5, 4)))


def test_span_cover():
    """T1: Span has one column, which covers the steps from `first` to `last` of each node,
    given as one value or one value per node."""
    steps = np.arange(1, 7)
    cases = [
        (2, 4, [2] * 3, [4] * 3),
        (np.array([1, 2, 6]), np.array([3, 2, 6]), [1, 2, 6], [3, 2, 6]),
        (1, np.array([1, 6, 4]), [1, 1, 1], [1, 6, 4]),
        (np.array([3, 1, 5]), 5, [3, 1, 5], [5, 5, 5]),
    ]
    for first, last, firsts, lasts in cases:
        span = Span(first, last)
        assert isinstance(span, Target)
        cover = span.cover(3, 6)
        assert cover.dtype == np.bool_
        assert cover.shape == (3, 1, 6)
        for n in range(3):
            expected = (steps >= firsts[n]) & (steps <= lasts[n])
            assert np.array_equal(cover[n, 0], expected)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, Span(1, 2)).first = 3


def test_cover_conditions():
    """T1: cover raises ValueError unless 1 <= first <= last <= horizon and each array has one
    value per node, and TypeError for steps that are not int."""
    wrong_values = [
        (0, 3),
        (2, 1),
        (1, 7),
        (np.array([1, 0, 1]), 3),
        (np.array([1, 2, 3]), np.array([1, 1, 3])),
        (1, np.array([2, 7, 3])),
        (np.array([1, 2]), 3),
        (1, np.array([2, 3, 4, 5])),
    ]
    for first, last in wrong_values:
        with pytest.raises(ValueError):
            Span(first, last).cover(3, 6)
    make: Any = Span  # the steps below have wrong types on purpose
    for first, last in [(1.0, 3), (1, "3"), (None, 3)]:
        with pytest.raises(TypeError):
            make(first, last).cover(3, 6)


@pytest.mark.parametrize("name", _cases.NAMES)
def test_point_and_truth_of_columns(name):
    """T2: the point and the truth of a column are the sums over its steps, in float64, stored
    as float32, and either is NaN when one of its step values is NaN. The score is the score of
    the stored point and truth."""
    case = _cases.make(name)
    exp = _cases.expected(case)
    rep = case.run()
    assert_bits(rep.point, exp.point)
    assert_bits(rep.truth, exp.truth)
    assert_bits(rep.score, exp.score)
    assert np.isnan(exp.truth).any() and np.isfinite(exp.truth).any()


def test_column_sums_need_float64():
    """T2: the cases hold column sums that a float32 sum would round."""
    rounded = False
    for name in _cases.NAMES:
        case = _cases.make(name)
        exp = _cases.expected(case)
        values = np.moveaxis(case.points[0], 2, 1)  # [N, O, H]
        for c in range(case.cover.shape[1]):
            float32_sum = np.zeros(values.shape[:2], np.float32)
            for h in range(values.shape[2]):
                step = np.where(case.cover[:, c, h][:, None], values[:, :, h], np.float32(0))
                float32_sum = float32_sum + step
            finite = np.isfinite(exp.point[:, c, :])
            rounded |= bool(np.any(float32_sum[finite] != exp.point[:, c, :][finite]))
    assert rounded
