import numpy as np
import pytest

from calibre.conformal.quantile import retained_quantile, score_quantile


def step_thresholds(resid, rows, level, window=None, groups=None):
    """Per-step band thresholds from absolute scores of the first rows per step."""
    steps = [
        retained_quantile(np.abs(resid[: int(rows[s]), :, s]), level, window, groups)
        for s in range(resid.shape[2])
    ]
    return np.stack(steps, axis=1)


def window_upper(point, resid, rows, level, steps, window=None, groups=None):
    """Point total plus the quantile of signed window sums over the first rows."""
    sums = resid[:rows, : point.shape[0], :steps].sum(axis=2)
    return point[:, :steps].sum(axis=1) + retained_quantile(sums, level, window, groups)


def test_score_quantile_uses_the_same_rank_for_pooled_and_per_series_scores():
    scores = np.array([[-4, 20], [3, -5], [4, 2]], dtype=np.float32)
    assert score_quantile(scores, np.array([0.5, 0.75])).tolist() == [3.0, 20.0]
    assert score_quantile(scores.ravel(), 0.5) == 3
    assert score_quantile(scores.ravel(), np.array([0.5, 0.75])).tolist() == [3.0, 20.0]
    assert np.isinf(score_quantile(scores.ravel(), 0.9))
    assert np.isinf(score_quantile(np.empty((0, 2)), 0.5)).all()
    assert np.isinf(score_quantile(np.array([]), 0.5))
    # A normalized pool can be formed without passing a model or its training history.
    assert score_quantile(scores.ravel() / 2, 0.5) * 2 == 3


@pytest.mark.parametrize(
    ("scores", "level"),
    [
        ([1, np.nan], 0.5),
        ([np.inf], 0.5),
        ([1], 0),
        ([1], 1),
        ([1], np.nan),
        (np.zeros((2, 2, 2)), 0.5),
        (np.zeros((2, 2)), [0.5, 0.6, 0.7]),
    ],
)
def test_score_quantile_rejects_invalid_inputs(scores, level):
    with pytest.raises(ValueError):
        score_quantile(np.asarray(scores), level)


def test_grouped_calibration_keeps_steps_windows_and_bound_rows_separate():
    resid = np.full((4, 3, 2), np.nan, dtype=np.float32)
    resid[:3] = [
        [[100, 100], [100, 100], [999, 999]],
        [[-1, 10], [3, 20], [999, 999]],
        [[-2, 30], [4, 40], [999, 999]],
    ]
    groups = np.array([7, 7, 9])
    out = step_thresholds(resid, np.array([3, 2]), 0.5, window=2, groups=groups)
    # Step 1 uses origins 1 and 2, step 2 uses origins 0 and 1. Tail NaNs are not read.
    assert out.tolist() == [[3.0, 100.0], [3.0, 100.0], [999.0, 999.0]]
    bound = window_upper(
        np.zeros((2, 2), dtype=np.float32),
        resid,
        3,
        np.array([0.5, 0.75]),
        2,
        window=2,
        groups=groups[:2],
    )
    # Signed sums [9, 23, 28, 44], with per-series ranks 3 and 4. No aggregate scores.
    assert bound.tolist() == [28.0, 44.0]
    assert np.isinf(step_thresholds(resid, np.array([0, 0]), 0.5, groups=groups)).all()


def test_step_thresholds_is_the_conformal_order_statistic():
    resid = np.full((6, 2, 2), np.nan, dtype=np.float32)
    resid[:5, 0, 0] = [1, -3, 2, -5, 4]
    resid[:5, 1, 0] = [10, 20, 30, 40, 50]
    resid[:2, 0, 1] = [7, -8]
    resid[:2, 1, 1] = [1, 2]

    out = step_thresholds(resid, rows=np.array([5, 2]), level=0.5)
    # k = 5, rank = ceil(6 * 0.5) = 3: third smallest absolute residual.
    assert out[:, 0].tolist() == [3.0, 30.0]
    # k = 2, rank = ceil(3 * 0.5) = 2: the largest of two.
    assert out[:, 1].tolist() == [8.0, 2.0]
    assert out.dtype == np.float32


def test_step_thresholds_with_window_uses_the_last_retained_rows():
    resid = np.full((8, 1, 1), np.nan, dtype=np.float32)
    resid[:6, 0, 0] = [100, 90, 1, 2, 3, 4]

    # k = 6 resolved, window 4 retains rows 2..5: rank ceil(5 * 0.9) = 5 > 4, not ready.
    assert np.isinf(step_thresholds(resid, rows=np.array([6]), level=0.9, window=4)).all()
    # At 0.5 the rank is ceil(5 * 0.5) = 3: third smallest of [1, 2, 3, 4].
    assert step_thresholds(resid, rows=np.array([6]), level=0.5, window=4).tolist() == [[3.0]]
    # Unbounded at 0.5: rank ceil(7 * 0.5) = 4 over all six, the 90 is inside the window.
    assert step_thresholds(resid, rows=np.array([6]), level=0.5, window=None).tolist() == [[4.0]]
    # A window wider than the resolved prefix is the unbounded case.
    assert step_thresholds(resid, rows=np.array([6]), level=0.5, window=10).tolist() == [[4.0]]
    # Window 2 at 0.5: rank ceil(3 * 0.5) = 2, the larger of the last two.
    assert step_thresholds(resid, rows=np.array([6]), level=0.5, window=2).tolist() == [[4.0]]


def test_window_upper_calibrates_signed_window_sums_per_row():
    point = np.array([[5, 7, 100], [10, 20, 100]], dtype=np.float32)
    resid = np.zeros((4, 2, 3), dtype=np.float32)
    resid[:3, 0, :2] = [[1, 2], [-5, 1], [2, 2]]
    resid[:3, 1, :2] = [[10, 10], [-3, -2], [1, 1]]

    out = window_upper(
        point,
        resid,
        rows=3,
        level=np.array([0.5, 0.75]),
        steps=2,
    )

    # Series 0: rank 2 of residual sums [-4, 3, 4] is 3, plus point sum 12.
    # Series 1: rank 3 of [-5, 2, 20] is 20, plus point sum 30.
    assert out.tolist() == [15.0, 50.0]
    # Step 3 is outside the window and does not affect either bound.
    assert out.max() < 100


def test_window_upper_uses_the_last_window_and_stays_infinite_until_ready():
    point = np.zeros((1, 2), dtype=np.float32)
    resid = np.array([[[50, 50]], [[1, 1]], [[2, 2]], [[3, 3]]], dtype=np.float32)
    level = np.array([0.75], dtype=np.float32)

    assert np.isinf(window_upper(point, resid, 2, level, 2, window=2)).all()
    # The last three sums are [2, 4, 6], and rank ceil(4 * .75) = 3.
    assert window_upper(point, resid, 4, level, 2, window=3).tolist() == [6.0]


def test_window_upper_is_not_projected_onto_support():
    point = np.ones((1, 2), dtype=np.float32)
    resid = np.array([[[-5, -4]]], dtype=np.float32)
    # One score -9 at rank ceil(2 * .5) = 1. Real-valued targets keep the negative bound.
    assert window_upper(point, resid, 1, 0.5, 2).tolist() == [-7.0]


def test_only_finite_signed_scores_enter_rank_selection():
    resid = np.array([[[-np.inf]], [[1]], [[2]], [[np.nan]]], dtype=np.float32)
    bound = window_upper(np.zeros((1, 1)), resid, 4, np.array([0.5]), 1)
    assert bound.tolist() == [2]
    assert np.isinf(window_upper(np.zeros((1, 1)), resid[:1], 1, np.array([0.5]), 1)).all()


def test_window_scores_require_every_step_and_use_finite_pool_counts():
    resid = np.array(
        [
            [[1, 2], [10, 20]],
            [[3, np.nan], [2, 4]],
            [[np.nan, 5], [np.nan, 1]],
            [[7, 8], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    point = np.zeros((2, 2), dtype=np.float32)
    local = window_upper(point, resid, 4, np.array([0.75, 0.5]), 2, window=2)
    assert local.tolist() == [np.inf, 30]
    grouped = window_upper(
        point, resid, 4, np.array([0.5, 0.75]), 2, window=2, groups=np.array([0, 0])
    )
    assert grouped.tolist() == [15, np.inf]  # Rows 1 and 3 supply just 6,15.


def test_absolute_step_scores_are_not_signed_window_scores():
    residuals = np.array([[[-8, 6]], [[-7, 5]], [[-6, 4]]], dtype=np.float32)
    point = np.array([[10, 10]], dtype=np.float32)
    band = step_thresholds(residuals, np.array([3, 3]), 0.5)
    bound = window_upper(point, residuals, 3, np.array([0.5]), 2)
    np.testing.assert_array_equal(band, [[7, 5]])
    np.testing.assert_array_equal(bound, [18])
    assert bound[0] != (point + band).sum()
    # Residual rows past `rows` must not enter a window score.
    extended = np.concatenate([residuals, np.full((2, 1, 2), 99999, dtype=np.float32)])
    np.testing.assert_array_equal(window_upper(point, extended, 3, np.array([0.5]), 2), bound)


def test_nine_scores_support_rank_nine_at_decimal_ninety_percent():
    assert score_quantile(np.arange(1, 10, dtype=np.float32), 0.9) == 9
    assert np.isinf(score_quantile(np.arange(1, 9, dtype=np.float32), 0.9))


def test_finite_counts_not_elapsed_rows_control_local_and_grouped_rank():
    resid = np.array(
        [
            [[1], [10], [np.nan]],
            [[np.nan], [20], [np.nan]],
            [[3], [np.nan], [np.nan]],
            [[np.nan], [40], [np.nan]],
        ],
        dtype=np.float32,
    )
    assert step_thresholds(resid, np.array([4]), 0.75, window=3)[:, 0].tolist() == [
        np.inf,
        40,
        np.inf,
    ]
    # Last two resolved rows in the group are rows 2 and 3: scores 3,40, not
    # the last two finite cells per member and not the last two pooled scores.
    grouped = step_thresholds(resid, np.array([4]), 0.5, window=2, groups=np.array([0, 0, 1]))
    assert grouped[:, 0].tolist() == [40, 40, np.inf]
    # A completely missing row is a hole, not one of the K retained rows.
    resid[3] = np.nan
    grouped = step_thresholds(resid, np.array([4]), 0.5, window=2, groups=np.array([0, 0, 1]))
    assert grouped[:, 0].tolist() == [20, 20, np.inf]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_missing_count_ranks_keep_existing_level_representation(dtype):
    resid = np.full((101, 2, 1), np.nan, dtype=np.float32)
    resid[:99, 0, 0] = np.arange(1, 100)
    resid[:49, 1, 0] = np.arange(1, 50)
    q = dtype(0.07)
    got = step_thresholds(resid, np.array([101]), q)
    assert got[0, 0] == score_quantile(resid[:99, 0, 0], q)
    assert got[1, 0] == score_quantile(resid[:49, 1, 0], q)


def test_step_thresholds_is_inf_until_ready_and_ignores_unresolved_rows():
    resid = np.zeros((20, 3, 1), dtype=np.float32)
    resid[9:] = np.nan
    assert np.isinf(step_thresholds(resid, rows=np.array([8]), level=0.9)).all()
    ready = step_thresholds(resid, rows=np.array([9]), level=0.9)
    assert ready.tolist() == [[0.0], [0.0], [0.0]]
    assert np.isinf(step_thresholds(resid, rows=np.array([0]), level=0.9)).all()


def test_window_must_keep_at_least_one_row():
    resid = np.ones((5, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="window"):
        step_thresholds(resid, np.array([5]), 0.5, window=0)
    with pytest.raises(ValueError, match="window"):
        window_upper(np.zeros((1, 1)), resid, 5, 0.5, 1, window=0)
