import inspect
import io
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from calibre import (
    ACI,
    AR1,
    CQR,
    Absolute,
    ConformalPID,
    EnbPI,
    Feedback,
    Forecasts,
    ForecastScorecaster,
    LeadTime,
    Miss,
    Quantiles,
    QuantileTracker,
    RiskControl,
    Signed,
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
    return Forecasts(np.asarray(origins), points, quantiles=Quantiles(interval, (0.1, 0.9)))


def test_cqr_scores_are_signed_and_use_only_the_native_forecast():
    target = np.array([-2, 0, 3, 10, 12, np.nan])
    point = np.tile([0, 10], (len(target), 1))
    np.testing.assert_allclose(CQR().score(target, point), [2, 0, -3, 0, 2, np.nan])
    assert CQR().parameter_shape() == ()
    assert CQR().input_shape() == (2,)
    assert Absolute().input_shape() == Signed().input_shape() == ()


def test_crossed_raw_quantiles_are_ordered_at_step_without_modifying_forecasts():
    raw = np.array([[[[10.0, 0], [4, -2]]]])
    source = raw.copy()
    forecasts = native([0], raw)
    np.testing.assert_array_equal(forecasts.quantiles.values, source)
    run = replay(
        forecasts,
        np.array([[0, 3, 8]]),
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(0.5),
    )
    np.testing.assert_array_equal(run.point, [[[[0, 10], [-2, 4]]]])
    np.testing.assert_array_equal(run.score, [[[-3, 4]]])
    np.testing.assert_array_equal(forecasts.quantiles.values, source)
    np.testing.assert_array_equal(raw, source)


def test_initial_state_requires_the_score_and_uses_one_forecast_array():
    assert inspect.signature(initial_state).parameters["score"].default is inspect.Parameter.empty
    scalar = initial_state(Step(), SplitQuantile(0.5), 2, 3, score=Absolute())
    paired = initial_state(Step(), SplitQuantile(0.5), 2, 3, score=CQR())
    assert scalar["ledger"].keys() == paired["ledger"].keys()
    assert scalar["ledger"]["point"].shape == (3, 2, 3)
    assert paired["ledger"]["point"].shape == (3, 2, 3, 2)
    assert "interval" not in paired["ledger"]


def test_feedback_uses_archived_native_intervals_and_scalar_corrections():
    feedback = Feedback(
        origin=np.array([0, 1, 2]),
        column=np.array([0, 1, 0]),
        point=np.tile([0, 10], (3, 2, 1)),
        target=np.array([[5, 20], [2, np.nan], [10, -1]]),
        issued=np.array([[-4, 1], [0, 0], [0, 0]]),
        censored=np.zeros((3, 2), dtype=bool),
        score=CQR(),
    )
    np.testing.assert_allclose(feedback.scores, [[-5, 10], [-2, np.nan], [0, 1]])
    misses, count = feedback.misses(2)
    np.testing.assert_array_equal(misses, [[0, 0], [2, 0]])
    np.testing.assert_array_equal(count, [[2, 1], [2, 0]])


def reference(forecasts, actuals, levels, window, groups):
    origins = forecasts.origins
    bounds = np.sort(forecasts.quantiles.values, axis=-1)
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
    ordered = np.sort(forecasts.quantiles.values, axis=-1)
    np.testing.assert_array_equal(run.point, ordered)
    np.testing.assert_array_equal(run.lower, ordered[..., 0] - run.threshold)
    np.testing.assert_array_equal(run.upper, ordered[..., 1] + run.threshold)
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
    np.testing.assert_array_equal(run.point, forecasts.quantiles.values)


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
    assert "ledger/interval" not in flatten(right.state)
    assert right.state["ledger"]["point"].shape == (5, 2, 5, 2)


def test_online_step_archives_the_raw_interval_and_not_the_current_one():
    method = SplitQuantile(0.5)
    state = initial_state(Step(), method, 1, 2, score=CQR())
    state, first = step(
        state,
        0,
        np.array([[[10, 0], [100, 0]]]),
        np.array([[0]]),
        target=Step(),
        score=CQR(),
        calibrator=method,
    )
    np.testing.assert_array_equal(first.point, [[[0, 10], [0, 100]]])
    state, _ = step(
        state,
        2,
        np.array([[[200, 300], [400, 500]]]),
        np.array([[3, 40]]),
        target=Step(),
        score=CQR(),
        calibrator=method,
    )
    np.testing.assert_array_equal(method.threshold(state["calibrator"]), [[-3, -40]])


@pytest.mark.parametrize(
    "interval",
    [
        None,
        np.zeros((1, 2)),
        np.zeros((1, 2, 3)),
        np.full((1, 2, 2), np.nan),
        np.full((1, 2, 2), np.inf),
    ],
)
@pytest.mark.parametrize("pending", [False, True])
def test_bad_online_intervals_are_rejected_before_mutating_state(interval, pending):
    method = SplitQuantile(0.5)
    state = initial_state(Step(), method, 1, 2, score=CQR())
    if pending:
        state, _ = step(
            state,
            0,
            np.zeros((1, 2, 2)),
            np.zeros((1, 1)),
            target=Step(),
            score=CQR(),
            calibrator=method,
        )
    before = {key: value.copy() for key, value in flatten(state).items()}
    with pytest.raises(ValueError, match="shape|finite"):
        step(
            state,
            int(pending),
            interval,
            np.full((1, 1), 5),
            target=Step(),
            score=CQR(),
            calibrator=method,
        )
    for key, value in flatten(state).items():
        np.testing.assert_array_equal(value, before[key])


@pytest.mark.parametrize("steps", [1, 2])
def test_native_intervals_cannot_be_used_as_lead_time_quantiles(steps):
    with pytest.raises(ValueError, match="Step target"):
        initial_state(LeadTime(steps), SplitQuantile(0.5), 1, 2, score=CQR())
    with pytest.raises(ValueError, match="Step target"):
        replay(
            native([0], np.zeros((1, 1, 2, 2))),
            np.zeros((1, 3)),
            target=LeadTime(steps),
            score=CQR(),
            calibrator=SplitQuantile(0.5),
        )


def test_online_cqr_requires_matching_input_and_parameter_shapes():
    method = SplitQuantile(0.5)
    with pytest.raises(ValueError, match="input shape"):
        step(
            initial_state(Step(), method, 1, 1, score=Absolute()),
            0,
            np.zeros((1, 1, 2)),
            np.zeros((1, 1)),
            target=Step(),
            score=CQR(),
            calibrator=method,
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
    with pytest.raises(ValueError, match="requires forecasts with native quantiles"):
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
    with pytest.raises(ValueError, match="input shape"):
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


def test_risk_control_broadcasts_node_grids_before_the_forecast_input_axes():
    scales = np.array([1, 2], dtype=np.float32)
    raw = np.tile([0, 10], (6, 2, 2, 1)) * scales[None, :, None, None]
    forecasts = native(np.arange(6), raw)
    actuals = np.array([[0, 5, 3, 9, 12, -2, 6, 1]]) * scales[:, None]
    grid = scales[:, None] * np.arange(-10, 11)[None, :]
    setup: dict[str, Any] = dict(target=Step(), score=CQR())
    split = replay(forecasts, actuals, calibrator=SplitQuantile(0.5), **setup)
    risk = replay(forecasts, actuals, calibrator=RiskControl(Miss(), grid, 0.5), **setup)
    np.testing.assert_array_equal(risk.threshold, split.threshold)
    np.testing.assert_array_equal(risk.lower, split.lower)
    np.testing.assert_array_equal(risk.upper, split.upper)


@pytest.mark.parametrize("scorecaster", [None, ForecastScorecaster(AR1())])
@pytest.mark.parametrize("horizon", [1, 3])
def test_cqr_and_pid_are_rejected_at_setup_before_any_feedback(scorecaster, horizon):
    method = ConformalPID(0.9, scorecaster=scorecaster)
    with pytest.raises(ValueError, match="CQR scores can be negative"):
        initial_state(Step(), method, 1, horizon, score=CQR())
    with pytest.raises(ValueError, match="CQR scores can be negative"):
        replay(
            native([0], np.zeros((1, 1, horizon, 2))),
            np.zeros((1, horizon + 1)),
            target=Step(),
            score=CQR(),
            calibrator=method,
        )


def test_residual_coverage_and_feedback_remain_score_based():
    forecasts = Forecasts(np.array([0, 1]), np.full((2, 1, 1), 0.1, dtype=np.float32))
    run = replay(
        forecasts,
        np.array([[0, 0.3, 0.3]], dtype=np.float32),
        target=Step(),
        score=Absolute(),
        calibrator=QuantileTracker(0.5, lr=0.1, start=0.2),
    )
    assert run.target[0, 0, 0] == run.upper[0, 0, 0]
    assert run.score[0, 0, 0] > run.threshold[0, 0, 0]
    assert run.covered[0, 0, 0] == 0
    np.testing.assert_allclose(run.threshold[1], 0.25)


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
