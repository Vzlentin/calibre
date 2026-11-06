from dataclasses import replace

import numpy as np
import pytest

from calibre import CQR, Absolute, Asymmetric, Feedback, Signed


@pytest.mark.parametrize(
    "score,outside,counts",
    [
        (Absolute(), [True, True], [[0, 2], [0, 2]]),
        (Signed(), [False, True], [[0, 1], [0, 2]]),
        (Asymmetric(), [True, True], [[0, 2], [0, 2]]),
        (CQR(), [True, True], [[0, 2], [0, 2]]),
    ],
)
def test_feedback_exposes_row_misses_and_counts_only_finite_scores(score, outside, counts):
    targets = np.array(
        [
            [0, 10],
            [-1, 11],
            [5, 5],
            [np.nan, np.nan],
            [np.inf, -np.inf],
            [5, 5],
            [5, 5],
            [1e38, -1e38],
        ],
        dtype=np.float32,
    )
    point = np.full((8, 2), 5, dtype=np.float32)
    point[-1] = [-3e38, 3e38]
    issued = np.full((8, 2), 5, dtype=np.float32)
    issued[5] = -1  # Empty absolute bands; signed upper bounds below the targets.
    issued[6] = np.inf
    if score.input_shape():
        point = np.tile([0, 10], (8, 2, 1)).astype(np.float32)
        point[-1] = [[-3e38, -3e38], [3e38, 3e38]]
        issued = np.zeros((8, 2), dtype=np.float32)
        issued[5] = -6  # Empty corrected native intervals.
        issued[6] = np.inf
    if score.parameter_shape():
        issued = np.tile([-5, 5], (8, 2, 1)).astype(np.float32)
        issued[5] = [1, -1]  # Empty asymmetric intervals.
        issued[6] = [-np.inf, np.inf]
    feedback = Feedback(
        origin=np.arange(8),
        column=np.arange(8) % 2,
        point=point,
        target=targets,
        issued=issued,
        censored=np.zeros((8, 2), dtype=bool),
        score=score,
    )
    expected = np.zeros((8, 2), dtype=bool)
    expected[1] = outside
    expected[5] = True
    with np.errstate(over="ignore", invalid="ignore"):
        assert feedback.miss.dtype == np.bool_
        np.testing.assert_array_equal(feedback.miss, expected)
        misses, known = feedback.misses(2)
    np.testing.assert_array_equal(misses, counts)
    np.testing.assert_array_equal(known, [[3, 2], [3, 2]])


def test_feedback_scores_are_cached_per_instance():
    feedback = Feedback(
        origin=np.array([0]),
        column=np.array([0]),
        point=np.array([[1, 4]], dtype=np.float32),
        target=np.array([[3, 1]], dtype=np.float32),
        issued=np.ones((1, 2), dtype=np.float32),
        censored=np.zeros((1, 2), dtype=bool),
        score=Absolute(),
    )
    scores = feedback.scores
    assert scores.dtype == np.float32
    np.testing.assert_array_equal(scores, [[2, 3]])
    np.testing.assert_array_equal(feedback.miss, [[True, True]])
    assert feedback.scores is scores
    misses, known = feedback.misses(1)
    np.testing.assert_array_equal(misses, [[1], [1]])
    np.testing.assert_array_equal(known, [[1], [1]])
    assert feedback.scores is scores

    later = replace(feedback, target=np.array([[1, 4]], dtype=np.float32))
    np.testing.assert_array_equal(later.scores, [[0, 0]])
    assert later.scores is not scores
    assert later.scores is later.scores


def test_feedback_with_no_rows_has_no_misses_or_known_scores():
    feedback = Feedback(
        origin=np.empty(0, dtype=np.int64),
        column=np.empty(0, dtype=np.int64),
        point=np.empty((0, 2)),
        target=np.empty((0, 2)),
        issued=np.empty((0, 2)),
        censored=np.empty((0, 2), dtype=bool),
        score=Absolute(),
    )
    assert feedback.miss.shape == (0, 2)
    misses, known = feedback.misses(3)
    np.testing.assert_array_equal(misses, np.zeros((2, 3)))
    np.testing.assert_array_equal(known, np.zeros((2, 3)))
