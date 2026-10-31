from typing import Any

import numpy as np
import pytest

from calibre import (
    Absolute,
    Asymmetric,
    ConformalPID,
    Feedback,
    Forecasts,
    LeadTime,
    SeasonalNaive,
    Step,
)
from calibre.backtest import ForecastScorecaster, replay
from calibre.conformal.calibrators import QuantileTracker
from calibre.conformal.scores import Signed
from calibre.online import initial_state
from calibre.online.state import flatten


def seasonal_scorecaster():
    return ForecastScorecaster(SeasonalNaive(2), min_history=2)


def feedback(targets, issued, columns=None, points=None, score=None) -> Feedback:
    targets = np.atleast_2d(np.asarray(targets, dtype=np.float32))
    return Feedback(
        origin=np.arange(len(targets)),
        column=np.zeros(len(targets), dtype=np.int64) if columns is None else np.asarray(columns),
        point=np.zeros_like(targets) if points is None else np.asarray(points, dtype=np.float32),
        target=targets,
        issued=np.broadcast_to(np.asarray(issued, dtype=np.float32), targets.shape),
        censored=np.zeros_like(targets, dtype=bool),
        score=Absolute() if score is None else score,
    )


def seeded(method, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None, None]
    return method.seed(method.initial_state(*scores.shape[1:]), scores)


def test_pid_empty_state_requires_explicit_seed_without_inventing_feedback():
    method = ConformalPID(0.9, scorecaster=None)
    state = method.initial_state(2, 3)
    assert method.threshold(state).shape == (2, 3)
    assert np.isposinf(method.threshold(state)).all()
    assert not state["n_updates"].any()
    with pytest.raises(ValueError, match="seed before feedback"):
        method.update(state, feedback([[1, 2]], 0))
    assert not state["count"].any()


def test_pid_seed_keeps_last_finite_scores_per_cell_and_fixed_scale():
    method = ConformalPID(np.array([[0.5, 0.9], [0.9, 0.5]]), window=3, scorecaster=None)
    scores = np.array(
        [
            [[100, 1], [10, 5]],
            [[1, np.nan], [np.nan, 5]],
            [[2, np.nan], [10, np.nan]],
            [[4, np.nan], [10, 5]],
            [[8, np.nan], [np.nan, 5]],
        ]
    )
    state = seeded(method, scores)
    np.testing.assert_array_equal(state["count"], [[3, 1], [3, 3]])
    np.testing.assert_array_equal(np.sort(state["scores"][0, 0]), [2, 4, 8])
    np.testing.assert_allclose(state["scale"], [[np.sqrt(56) / 3, 0.1], [1, 0.5]])
    np.testing.assert_array_equal(method.threshold(state), [[4, 1], [10, 5]])
    assert method.threshold(state).dtype == np.float32
    assert not state["cumulative_error"].any()
    assert not state["n_updates"].any()
    assert all(v.shape[0] == 2 and v.dtype != object for v in flatten(state).values())


def test_pid_zero_seed_has_a_positive_fixed_scale_and_finite_threshold():
    method = ConformalPID(0.9, scorecaster=None)
    state = seeded(method, [0, 0, 0])
    np.testing.assert_array_equal(state["scale"], [[1e-8]])
    np.testing.assert_array_equal(method.threshold(state), [[0]])
    method.update(state, feedback([[100]], 0))
    np.testing.assert_array_equal(state["scale"], [[1e-8]])


def test_pid_p_only_matches_quantile_tracker_at_the_effective_learning_rate():
    method = ConformalPID(0.9, learning_rate=0.25, integral_gain=0, scorecaster=None)
    state = seeded(method, [0, 2])  # scale=1, empirical quantile=2
    tracker = QuantileTracker(0.9, lr=0.25, start=2)
    tracked = tracker.initial_state(1, 1)
    for target, issued in [(1, 0.5), (0, 3), (5, 1), (2, 2)]:
        row = feedback([[target]], issued)
        state = method.update(state, row)
        tracked = tracker.update(tracked, row)
        np.testing.assert_array_equal(method.threshold(state), tracker.threshold(tracked))
    assert state["n_updates"].item() == 4
    np.testing.assert_array_equal(state["scale"], [[1]])


def test_pid_updates_against_archived_threshold_and_archived_center():
    method = ConformalPID(0.9, learning_rate=0.25, integral_gain=0, scorecaster=None)
    state = seeded(method, [0, 2])
    # Current threshold=2 would hit. Archived center=10, target=11 and radius=.5 miss.
    method.update(state, feedback([[11]], 0.5, points=[[10]]))
    np.testing.assert_allclose(state["proportional"], [[2.225]])
    np.testing.assert_allclose(state["cumulative_error"], [[0.9]])
    assert state["scores"][0, 0, 2] == 1
    assert state["n_updates"].item() == 1


def test_pid_integrator_uses_the_number_of_valid_updates_and_log_time():
    method = ConformalPID(0.75, learning_rate=0.1, integral_gain=0.4, scorecaster=None)
    state = seeded(method, [0, 2])
    method.update(state, feedback([[5], [0]], [[2], [2]]))
    # One miss and one hit: cumulative error=.75-.25=.5, t=2, scale=1.
    expected = 0.4 * np.tan(0.5 * np.log(3) / 15)
    np.testing.assert_allclose(method._integrator(state), [[expected]], rtol=1e-14)
    np.testing.assert_allclose(state["proportional"], [[2.05]])
    np.testing.assert_allclose(method.threshold(state), [[2.05 + expected]], rtol=1e-7)


@pytest.mark.parametrize(("target", "sign"), [(5, 1), (0, -1)])
def test_pid_tangent_saturates_to_whole_line_or_empty_set(target, sign):
    method = ConformalPID(0.5, integral_gain=1, saturation_scale=0.01, scorecaster=None)
    state = seeded(method, [0, 2])
    method.update(state, feedback([[target]], 1))
    assert method._integrator(state).item() == sign * np.inf
    threshold = method.threshold(state)
    assert threshold.item() == sign * np.inf
    lower, upper = Absolute().bound(np.zeros((1, 1)), threshold)
    assert bool(lower.item() <= upper.item()) == (sign > 0)
    # Another revealed feedback leaves the finite P state recoverable.
    method.update(state, feedback([[0]], threshold))
    assert np.isfinite(state["proportional"]).all()


def test_pid_disabled_integrator_never_multiplies_zero_by_infinity():
    method = ConformalPID(0.9, integral_gain=0, saturation_scale=0.001, scorecaster=None)
    state = seeded(method, [1, 2, 3])
    method.update(state, feedback([[100]], 1))
    with np.errstate(all="raise"):
        assert np.isfinite(method.threshold(state)).all()
        np.testing.assert_array_equal(method._integrator(state), [[0]])


def test_pid_negative_radius_stays_an_empty_set_and_endpoints_are_inclusive():
    method = ConformalPID(0.5, learning_rate=10, integral_gain=0, scorecaster=None)
    state = seeded(method, [0, 2])  # P=0, scale=1
    method.update(state, feedback([[0]], 0))  # equality is a hit
    np.testing.assert_array_equal(method.threshold(state), [[-5]])
    np.testing.assert_allclose(state["cumulative_error"], [[-0.5]])
    method.update(state, feedback([[0]], -5))  # even score zero misses an empty set
    np.testing.assert_array_equal(method.threshold(state), [[0]])


@pytest.mark.parametrize(("target", "ahead"), [(Step(), np.arange(1, 4)), (LeadTime(3), 3)])
def test_pid_scorecast_respects_step_and_lead_time_horizons(target, ahead):
    history = np.array([2, 8, 3, 7])
    scales = np.array([1.0, 2.5])[:, None]
    method = ConformalPID(
        0.75, window=len(history), ahead=ahead, scorecaster=seasonal_scorecaster()
    )
    state = initial_state(target, method, 2, 3, score=Absolute())
    n_columns = len(target.cover(3))
    scores = np.broadcast_to(history[:, None, None] * scales, (len(history), 2, n_columns))
    method.seed(state["calibrator"], scores)
    expected = scales * np.where(np.atleast_1d(ahead) % 2, 3, 7)
    np.testing.assert_array_equal(method._scorecast(state["calibrator"]), expected)
    np.testing.assert_array_equal(
        method.threshold(state["calibrator"]), np.broadcast_to(7 * scales, (2, n_columns))
    )


def test_pid_scorecast_ring_wrap_preserves_time_order_and_its_own_window():
    method = ConformalPID(0.5, window=4, scorecast_window=3, scorecaster=seasonal_scorecaster())
    state = seeded(method, [9, 8, 1, 2])
    scale = state["scale"].copy()
    method.update(state, feedback([[3], [4], [5], [6], [7]], 100))
    # Last three chronological scores are [5,6,7], not the physical ring order.
    np.testing.assert_array_equal(np.sort(state["scores"][0, 0]), [4, 5, 6, 7])
    np.testing.assert_array_equal(method._scorecast(state), [[6]])
    np.testing.assert_array_equal(state["scale"], scale)


@pytest.mark.parametrize(("window", "scorecast_window"), [(1, 64), (3, 1), (3, 2)])
def test_pid_small_windows_are_supported(window, scorecast_window):
    method = ConformalPID(
        0.9,
        window=window,
        scorecast_window=scorecast_window,
        scorecaster=ForecastScorecaster(SeasonalNaive(1)),
    )
    state = seeded(method, [1, 2, 3])
    used = [1, 2, 3][-window:][-scorecast_window:]
    np.testing.assert_allclose(method._scorecast(state), [[np.mean(used)]])


def test_pid_missing_and_overflowed_scores_do_not_advance_any_state():
    method = ConformalPID(0.9, scorecaster=None)
    state = seeded(method, np.ones((3, 2, 1)))
    before = {key: value.copy() for key, value in state.items()}
    large = np.finfo(np.float32).max
    method.update(state, feedback([[np.nan, large]], 1, points=[[0, -large]]))
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_pid_batched_feedback_matches_sequential_updates_without_cross_cell_pooling():
    levels = np.array([[0.5, 0.9], [0.8, 0.75]])
    method = ConformalPID(levels, window=3, scorecaster=None)
    scores = np.array([[[0, 1], [2, 3]], [[2, 3], [4, 5]]])
    state = seeded(method, scores)
    separate = seeded(method, scores)
    targets = [[1, np.nan], [0, 8], [5, 1], [3, 0]]
    issued = [[0.5, 2], [1, 5], [6, 1], [2, 0]]
    columns = [0, 1, 0, 1]
    method.update(state, feedback(targets, issued, columns))
    for values, radii, column in zip(targets, issued, columns, strict=True):
        method.update(separate, feedback([values], [radii], [column]))
    for key in state:
        np.testing.assert_array_equal(state[key], separate[key])
    np.testing.assert_array_equal(state["n_updates"], [[2, 2], [1, 2]])
    np.testing.assert_allclose(state["cumulative_error"], [[0, 0.8], [-0.2, 0.5]])


def test_pid_multi_cell_scorecasting_matches_independent_single_cell_runs():
    levels = np.array([[0.5, 0.9], [0.8, 0.75]])
    ahead = np.array([1, 3])
    method = ConformalPID(
        levels, window=3, scorecast_window=3, ahead=ahead, scorecaster=seasonal_scorecaster()
    )
    seed = np.array([[[1, 10], [100, 2]], [[2, np.nan], [50, 2]], [[4, np.nan], [25, 2]]])
    state = seeded(method, seed)
    rows = feedback([[5, 20], [3, np.nan], [6, 10]], [[4, 25], [5, 3], [1, 100]], [0, 1, 0])
    method.update(state, rows)
    for node in range(2):
        for column in range(2):
            alone = ConformalPID(
                float(levels[node, column]),
                window=3,
                scorecast_window=3,
                ahead=int(ahead[column]),
                scorecaster=seasonal_scorecaster(),
            )
            single = seeded(alone, seed[:, node : node + 1, column : column + 1])
            selected = rows.column == column
            alone.update(
                single,
                feedback(
                    rows.target[selected, node : node + 1], rows.issued[selected, node : node + 1]
                ),
            )
            np.testing.assert_allclose(
                method._scorecast(state)[node, column], alone._scorecast(single).item()
            )
            np.testing.assert_array_equal(
                method.threshold(state)[node, column], alone.threshold(single).item()
            )


def test_pid_threshold_is_read_only_and_instances_do_not_hide_learned_state():
    method = ConformalPID(0.9, scorecaster=seasonal_scorecaster())
    left = seeded(method, [0, 2])
    right = seeded(method, [0, 2])
    before = {key: value.copy() for key, value in left.items()}
    parameters = vars(method).copy()
    for _ in range(3):
        method.threshold(left)
    for key in left:
        np.testing.assert_array_equal(left[key], before[key])
    method.update(left, feedback([[100]], 1))
    for key in right:
        np.testing.assert_array_equal(right[key], before[key])
    assert vars(method) == parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level": 0},
        {"level": 1},
        {"level": np.nan},
        {"level": np.full(2, 0.9)},
        {"learning_rate": 0},
        {"learning_rate": np.inf},
        {"learning_rate": [0.1]},
        {"integral_gain": -1},
        {"integral_gain": np.nan},
        {"saturation_scale": 0},
        {"saturation_scale": np.inf},
        {"window": 0},
        {"window": True},
        {"window": 2.5},
        {"scorecast_window": 0},
        {"scorecast_window": np.bool_(True)},
        {"scorecaster": "yes"},
        {"scorecaster": True},
        {"ahead": 0},
        {"ahead": -1},
        {"ahead": True},
        {"ahead": 1.5},
        {"ahead": np.nan},
        {"ahead": np.inf},
        {"ahead": [1, 0]},
        {"ahead": []},
        {"ahead": np.ones((1, 2), dtype=np.int64)},
        {"ahead": np.array([2**63], dtype=np.uint64)},
    ],
)
def test_pid_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ConformalPID(**({"level": 0.9, "scorecaster": None} | kwargs))


def test_pid_validates_level_shape_against_the_state():
    method = ConformalPID(np.full((2, 2), 0.9), scorecaster=None)
    with pytest.raises(ValueError):
        method.initial_state(3, 1)


def test_pid_validates_ahead_shape_against_the_state():
    method = ConformalPID(0.9, ahead=np.array([1, 3]), scorecaster=None)
    with pytest.raises(ValueError):
        method.initial_state(2, 3)


@pytest.mark.parametrize(
    "scores",
    [np.zeros((3, 2, 1)), np.zeros(3), [[[-1]]], [[[np.inf]]], [[[np.nan]]], np.empty((0, 1, 1))],
)
def test_pid_rejects_invalid_seed_without_mutating_state(scores):
    method = ConformalPID(0.9, scorecaster=None)
    state = method.initial_state(1, 1)
    with pytest.raises(ValueError, match="seed"):
        method.seed(state, scores)
    assert not state["count"].any()


def test_pid_cannot_seed_twice_and_rejects_negative_or_paired_scores():
    method = ConformalPID(0.9, scorecaster=None)
    state = seeded(method, [1, 2, 3])
    with pytest.raises(ValueError, match="empty"):
        method.seed(state, np.ones((3, 1, 1)))
    before = {key: value.copy() for key, value in state.items()}
    for score in (Signed(), Asymmetric()):
        with pytest.raises(ValueError, match="nonnegative scalar"):
            method.update(state, feedback([[-1]], 1, score=score))
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_pid_replay_uses_archived_points_and_thresholds_end_to_end():
    method = ConformalPID(0.9, integral_gain=0, scorecaster=None)
    state = initial_state(Step(), method, 1, 1, score=Absolute())
    state["calibrator"] = method.seed(state["calibrator"], np.array([0, 2]).reshape(2, 1, 1))
    result = replay(
        Forecasts(np.array([1, 2, 3]), np.array([[[10]], [[100]], [[0]]], dtype=np.float32)),
        np.array([[0, 0, 11, 110, 0]], dtype=np.float32),
        target=Step(),
        score=Absolute(),
        calibrator=method,
        state=state,
    )
    np.testing.assert_allclose(result.threshold[:, 0, 0], [2, 1.995, 2.04])
    np.testing.assert_array_equal(result.state["calibrator"]["scores"][0, 0, :4], [0, 2, 1, 10])


def test_pid_requires_an_explicit_scorecaster_choice():
    missing: dict[str, Any] = {"level": 0.9}
    with pytest.raises(TypeError, match="scorecaster"):
        ConformalPID(**missing)
    method = ConformalPID(0.9, scorecaster=None)
    state = method.initial_state(1, 1)
    np.testing.assert_array_equal(method._scorecast(state), [[0]])


def test_pid_rejects_nonfinite_scorecaster_output_after_the_mean_fallback():
    method = ConformalPID(0.9, scorecaster=ForecastScorecaster(SeasonalNaive(1)))
    state = seeded(method, [1, 2])
    state["scores"][0, 0, :2] = np.finfo(np.float64).max
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="scorecaster must return finite predictions"),
    ):
        method._scorecast(state)


def test_pid_setup_rejects_asymmetric_parameters():
    method = ConformalPID(0.9, scorecaster=None)
    with pytest.raises(ValueError, match="parameter shape"):
        initial_state(Step(), method, 1, 1, score=Asymmetric())
