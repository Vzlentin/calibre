import io
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from calibre import (
    ACI,
    CQR,
    Absolute,
    EnbPI,
    Feedback,
    Forecasts,
    LeadTime,
    Miss,
    QuantileTracker,
    RiskControl,
    SplitQuantile,
    Step,
    initial_state,
    replay,
    step,
)
from calibre.online.state import flatten, unflatten


def native(origins, interval, points=None):
    interval = np.asarray(interval, dtype=np.float32)
    if points is None:
        points = np.full(interval.shape[:-1], 1000, dtype=np.float32)
    return Forecasts(np.asarray(origins), points, interval=interval, quantile_levels=(0.1, 0.9))


def test_cqr_scores_are_signed_and_independent_of_the_point():
    target = np.array([-2, 0, 3, 10, 12, np.nan])
    interval = np.tile([0, 10], (len(target), 1))
    expected = [-2, 0, 3, 0, -2, np.nan]
    np.testing.assert_allclose(
        CQR().score(target, np.full_like(target, 1000), interval=interval),
        -np.asarray(expected),
    )
    assert CQR().parameter_shape() == ()
    assert CQR().requires_interval()


def test_crossed_raw_quantiles_are_ordered_without_modifying_the_input():
    interval = np.array([[10.0, 0], [4, -2]])
    source = interval.copy()
    score = CQR()
    np.testing.assert_array_equal(
        score.score(np.array([3, 8]), np.zeros(2), interval=interval), [-3, 4]
    )
    lower, upper = score.bound(np.zeros(2), np.array([1, -1]), interval=interval)
    np.testing.assert_array_equal(lower, [-1, -1])
    np.testing.assert_array_equal(upper, [11, 3])
    np.testing.assert_array_equal(interval, source)


@pytest.mark.parametrize("interval", [None, np.zeros((2,)), np.zeros((2, 3))])
def test_cqr_rejects_missing_or_misshaped_intervals(interval):
    for operation in (CQR().score, CQR().bound):
        with pytest.raises(ValueError, match="native intervals"):
            operation(np.zeros(2), np.zeros(2), interval=interval)


def test_feedback_uses_archived_native_intervals_and_scalar_corrections():
    feedback = Feedback(
        origin=np.array([0, 1, 2]),
        column=np.array([0, 1, 0]),
        point=np.full((3, 2), 999.0),
        target=np.array([[5, 20], [2, np.nan], [10, -1]]),
        issued=np.array([[-4, 1], [0, 0], [0, 0]]),
        censored=np.zeros((3, 2), dtype=bool),
        score=CQR(),
        interval=np.tile([0, 10], (3, 2, 1)),
    )
    np.testing.assert_allclose(feedback.scores, [[-5, 10], [-2, np.nan], [0, 1]])
    misses, count = feedback.misses(2)
    np.testing.assert_array_equal(misses, [[0, 0], [2, 0]])
    np.testing.assert_array_equal(count, [[2, 1], [2, 0]])


def reference(forecasts, actuals, levels, window, groups):
    origins, bounds = forecasts.origins, forecasts.interval
    o_count, n_count, horizon, _ = bounds.shape
    scores = np.full((o_count, n_count, horizon), np.nan)
    thresholds = np.full_like(scores, np.inf)
    for i, origin in enumerate(origins):
        for n in range(n_count):
            for h in range(horizon):
                t = origin + h + 1
                if t < actuals.shape[1]:
                    y = actuals[n, t]
                    scores[i, n, h] = max(bounds[i, n, h, 0] - y, y - bounds[i, n, h, 1])
    for i, origin in enumerate(origins):
        for n in range(n_count):
            members = [n] if groups is None else np.flatnonzero(groups == groups[n])
            for h in range(horizon):
                rows = [
                    j
                    for j in range(i)
                    if origins[j] + h + 1 <= origin and np.isfinite(scores[j, members, h]).any()
                ]
                if window is not None:
                    rows = rows[-window:]
                retained = sorted(
                    scores[j, member, h]
                    for j in rows
                    for member in members
                    if np.isfinite(scores[j, member, h])
                )
                rank = int(np.ceil((len(retained) + 1) * levels[n, h]))
                if rank <= len(retained):
                    thresholds[i, n, h] = retained[rank - 1]
    return scores, thresholds


@pytest.mark.parametrize("window", [None, 2])
@pytest.mark.parametrize("groups", [None, np.array([0, 0])])
def test_replay_matches_independent_cqr_ranks_with_delays_gaps_holes_and_pooling(window, groups):
    rng = np.random.default_rng(4)
    origins = np.array([3, 4, 7, 8, 12, 17])
    raw = rng.normal(size=(len(origins), 2, 3, 2)).astype(np.float32)
    actuals = rng.normal(size=(2, 23)).astype(np.float32)
    actuals[0, 9] = np.nan
    forecasts = native(origins, raw)
    levels = np.array([[0.5, 0.6, 0.75], [0.7, 0.5, 0.5]])
    expected_scores, expected_thresholds = reference(forecasts, actuals, levels, window, groups)
    run = replay(
        forecasts,
        actuals,
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(levels, window=window, groups=groups),
    )
    np.testing.assert_array_equal(run.score, expected_scores)
    np.testing.assert_array_equal(run.threshold, expected_thresholds)
    np.testing.assert_array_equal(run.lower, forecasts.interval[..., 0] - run.threshold)
    np.testing.assert_array_equal(run.upper, forecasts.interval[..., 1] + run.threshold)
    expected_covered = (run.lower <= run.target) & (run.target <= run.upper)
    np.testing.assert_array_equal(
        run.covered, np.where(np.isfinite(expected_scores), expected_covered.astype(float), np.nan)
    )


def test_rank_n_plus_one_requires_nine_scores_at_ninety_percent():
    forecasts = native(np.arange(10), np.tile([0, 10], (10, 1, 1, 1)))
    run = replay(
        forecasts,
        np.full((1, 11), 5),
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(0.9),
    )
    assert np.isposinf(run.threshold[:9]).all()
    assert np.isneginf(run.lower[:9]).all()
    assert np.isposinf(run.upper[:9]).all()
    assert run.threshold[9, 0, 0] == -5
    assert run.lower[9, 0, 0] == run.upper[9, 0, 0] == 5


def test_negative_correction_can_issue_an_empty_set_without_reordering():
    forecasts = native([0, 1], [[[[0, 100]]], [[[0, 2]]]])
    run = replay(
        forecasts,
        np.array([[0, 50, 1]]),
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(0.5),
    )
    assert run.threshold[1, 0, 0] == -50
    assert run.lower[1, 0, 0] == 50
    assert run.upper[1, 0, 0] == -48
    np.testing.assert_array_equal(run.covered[:, 0, 0], [1, 0])


def test_cqr_feedback_and_replay_count_float32_endpoint_equality_as_a_hit():
    forecasts = native([0, 1], np.tile([-1, 0.1], (2, 1, 1, 1)))
    actuals = np.array([[0, 0.3, 0.3]], dtype=np.float32)
    run = replay(
        forecasts,
        actuals,
        target=Step(),
        score=CQR(),
        calibrator=QuantileTracker(0.5, lr=0.1, start=0.2),
    )
    assert run.target[0, 0, 0] == run.upper[0, 0, 0]
    assert run.score[0, 0, 0] > run.threshold[0, 0, 0]
    assert run.covered[0, 0, 0] == 1
    np.testing.assert_allclose(run.threshold[1], 0.15)
    np.testing.assert_array_equal(run.interval, forecasts.interval)


def test_missing_future_targets_remain_nan():
    forecasts = native([2, 4], np.tile([0, 10], (2, 1, 3, 1)))
    run = replay(
        forecasts,
        np.arange(6)[None, :],
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(0.5),
    )
    assert np.isnan(run.target[1, 0, 1:]).all()
    assert np.isnan(run.score[1, 0, 1:]).all()
    assert np.isnan(run.covered[1, 0, 1:]).all()


def test_future_actuals_and_future_native_forecasts_cannot_change_past_issues():
    rng = np.random.default_rng(9)
    origins = np.arange(5, 35)
    raw = rng.normal(size=(len(origins), 2, 4, 2)).astype(np.float32)
    actuals = rng.normal(size=(2, 40)).astype(np.float32)
    setup: dict[str, Any] = dict(
        target=Step(), score=CQR(), calibrator=SplitQuantile(0.7, window=7)
    )
    before = replay(native(origins, raw), actuals, **setup)
    actuals[:, 20:] += 100
    raw[origins >= 20] += 500
    after = replay(native(origins, raw), actuals, **setup)
    for name in ("threshold", "lower", "upper"):
        np.testing.assert_array_equal(
            getattr(before, name)[origins < 20], getattr(after, name)[origins < 20]
        )


@pytest.mark.parametrize("adaptive", [False, True])
def test_saved_state_resumes_exactly_with_pending_native_intervals(adaptive):
    rng = np.random.default_rng(6)
    origins = np.array([5, 6, 9, 12, 13, 16, 20, 22, 25])
    raw = rng.normal(size=(len(origins), 2, 5, 2)).astype(np.float32)
    forecasts = native(origins, raw)
    actuals = rng.normal(size=(2, 32)).astype(np.float32)
    calibrator = SplitQuantile(0.6, window=3)
    if adaptive:
        calibrator = ACI(calibrator, gamma=0.05)
    setup: dict[str, Any] = dict(target=Step(), score=CQR(), calibrator=calibrator)
    whole = replay(forecasts, actuals, **setup)
    left = replay(native(origins[:4], raw[:4]), actuals, **setup)
    buffer = io.BytesIO()
    flat: dict[str, Any] = flatten(left.state)
    np.savez(buffer, allow_pickle=False, **flat)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as arrays:
        state = unflatten(dict(arrays))
    right = replay(native(origins[4:], raw[4:]), actuals, state=state, **setup)
    for name in ("threshold", "lower", "upper", "point", "covered"):
        np.testing.assert_array_equal(
            np.concatenate([getattr(left, name), getattr(right, name)]), getattr(whole, name)
        )
    for name, value in flatten(whole.state).items():
        np.testing.assert_array_equal(flatten(right.state)[name], value)
    assert "ledger/interval" in flatten(right.state)


def test_online_step_archives_the_raw_interval_and_not_the_current_one():
    method = SplitQuantile(0.5)
    state = initial_state(Step(), method, 1, 2, score=CQR())
    state, first = step(
        state,
        0,
        np.zeros((1, 2)),
        np.array([[0]]),
        target=Step(),
        score=CQR(),
        calibrator=method,
        interval=np.array([[[10, 0], [100, 0]]]),
    )
    np.testing.assert_array_equal(first.interval, [[[0, 10], [0, 100]]])
    state, _ = step(
        state,
        2,
        np.zeros((1, 2)),
        np.array([[3, 40]]),
        target=Step(),
        score=CQR(),
        calibrator=method,
        interval=np.array([[[200, 300], [400, 500]]]),
    )
    np.testing.assert_array_equal(method.threshold(state["calibrator"]), [[-3, -40]])


@pytest.mark.parametrize(
    "interval", [None, np.zeros((1, 2)), np.full((1, 2, 2), np.nan), np.full((1, 2, 2), np.inf)]
)
def test_bad_online_intervals_are_rejected_before_mutating_state(interval):
    method = SplitQuantile(0.5)
    state = initial_state(Step(), method, 1, 2, score=CQR())
    before = {key: value.copy() for key, value in flatten(state).items()}
    with pytest.raises(ValueError, match="native intervals"):
        step(
            state,
            0,
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            target=Step(),
            score=CQR(),
            calibrator=method,
            interval=interval,
        )
    for key, value in flatten(state).items():
        np.testing.assert_array_equal(value, before[key])


def test_native_intervals_cannot_be_used_as_lead_time_quantiles():
    with pytest.raises(ValueError, match="Step target"):
        initial_state(LeadTime(2), SplitQuantile(0.5), 1, 2, score=CQR())
    with pytest.raises(ValueError, match="Step target"):
        replay(
            native([0], np.zeros((1, 1, 2, 2))),
            np.zeros((1, 3)),
            target=LeadTime(2),
            score=CQR(),
            calibrator=SplitQuantile(0.5),
        )


def test_online_cqr_requires_explicit_native_state_and_scalar_parameters():
    method = SplitQuantile(0.5)
    with pytest.raises(ValueError, match="initialize native interval state"):
        step(
            initial_state(Step(), method, 1, 1),
            0,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            target=Step(),
            score=CQR(),
            calibrator=method,
            interval=np.zeros((1, 1, 2)),
        )
    with pytest.raises(ValueError, match="parameter shape"):
        replay(
            native([0], np.zeros((1, 1, 1, 2))),
            np.zeros((1, 2)),
            target=Step(),
            score=CQR(),
            calibrator=EnbPI(0.5, 2),
        )


def test_replay_cqr_requires_native_forecasts():
    with pytest.raises(ValueError, match="requires forecasts with native intervals"):
        replay(
            Forecasts(np.array([0]), np.zeros((1, 1, 1))),
            np.zeros((1, 2)),
            target=Step(),
            score=CQR(),
            calibrator=SplitQuantile(0.5),
        )


def test_point_scores_can_ignore_native_forecasts_but_cannot_resume_native_state():
    forecasts = native([0], np.zeros((1, 1, 1, 2)))
    run = replay(
        forecasts, np.zeros((1, 2)), target=Step(), score=Absolute(), calibrator=SplitQuantile(0.5)
    )
    assert "interval" not in run.state["ledger"]
    with pytest.raises(ValueError, match="interval score"):
        replay(
            native([1], np.zeros((1, 1, 1, 2))),
            np.zeros((1, 3)),
            state=initial_state(Step(), SplitQuantile(0.5), 1, 1, score=CQR()),
            target=Step(),
            score=Absolute(),
            calibrator=SplitQuantile(0.5),
        )


def test_risk_control_evaluates_native_bounds_at_each_grid_correction():
    forecasts = native(np.arange(6), np.tile([0, 10], (6, 1, 1, 1)))
    actuals = np.array([[0, 5, 3, 9, 12, -2, 6]])
    setup: dict[str, Any] = dict(target=Step(), score=CQR())
    grid = np.arange(-10, 11, dtype=float)
    split = replay(forecasts, actuals, calibrator=SplitQuantile(0.5), **setup)
    risk = replay(forecasts, actuals, calibrator=RiskControl(Miss(), grid, 0.5), **setup)
    np.testing.assert_array_equal(risk.threshold, split.threshold)


def test_import_calibre_does_not_import_optional_quantile_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import calibre, sys; assert calibre.CQR; "
            "assert not {'mlforecast', 'neuralforecast', 'sklearn', 'torch'} & sys.modules.keys()",
        ],
        check=True,
    )
