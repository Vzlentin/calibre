"""Independent threshold checks for scorecaster composition and continuation."""

import io
import math
from typing import Any

import numpy as np
import pytest
from statsforecast.models import Theta

from calibre import (
    AR1,
    Absolute,
    ConformalPID,
    Forecasts,
    ForecastScorecaster,
    LeadTime,
    SeasonalNaive,
    Step,
    initial_state,
    replay,
    step,
)
from calibre.forecast.models.statsforecast import StatsForecastModel
from calibre.online.state import flatten, unflatten


def test_pid_clips_negative_score_predictions_not_generic_forecasts():
    method = ConformalPID(0.9, scorecaster=ForecastScorecaster(AR1()), ahead=10)
    state = method.seed(method.initial_state(1, 1), np.array([10, 9, 8, 7])[:, None, None])
    np.testing.assert_array_equal(method._scorecast(state), [[0]])


def scorecaster(name):
    if name == "none":
        return None
    if name == "ar1":
        return ForecastScorecaster(AR1())
    if name == "theta":
        return ForecastScorecaster(StatsForecastModel(Theta()), min_history=4)
    return ForecastScorecaster(SeasonalNaive(2), min_history=2)


def reference_scorecast(name, history, ahead):
    """Use raw vendor calls or independent formulas, never the tested adapter."""
    if name == "none":
        return 0.0
    minimum = {"ar1": 3, "theta": 4, "seasonal": 2}[name]
    values = np.asarray(history, dtype=np.float64)
    if len(values) < minimum:
        return float(np.mean(values))
    if name == "theta":
        # StatsForecastModel returns float32; use the vendor directly as the oracle.
        prediction = Theta().forecast(values, h=int(ahead))["mean"][-1]
        return max(0.0, float(np.float32(prediction)))
    if name == "seasonal":
        return float(values[-2 + (int(ahead) - 1) % 2])
    # Solve the augmented ridge system, then iterate its one-step recurrence.
    design = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    penalty = math.sqrt(1e-6 * float(np.var(values[:-1])))
    design = np.vstack([design, [0, penalty]])
    _, slope = np.linalg.lstsq(design, np.append(values[1:], 0), rcond=None)[0]
    slope = float(np.clip(slope, -0.99, 0.99))
    intercept = float(np.mean(values[1:]) - slope * np.mean(values[:-1]))
    prediction = float(values[-1])
    for _ in range(int(ahead)):
        prediction = intercept + slope * prediction
    return max(0.0, prediction)


@pytest.mark.parametrize("model", ["ar1", "theta", "seasonal", "none"])
@pytest.mark.parametrize(
    ("target", "covered"),
    [(Step(), [(1,), (2,), (3,)]), (LeadTime(3), [(1, 2, 3)])],
    ids=["step", "lead-time"],
)
def test_pid_thresholds_match_independent_delayed_feedback_calculation(model, target, covered):
    n_nodes, n_columns = 2, len(covered)
    window, scorecast_window = 6, 4
    ahead = np.array([steps[-1] for steps in covered], dtype=np.int64)
    levels = np.broadcast_to(np.array([[0.75], [0.8]]), (n_nodes, n_columns))
    learning_rate, integral_gain, saturation_scale = 0.2, 0.3, 5.0
    method = ConformalPID(
        levels,
        scorecaster=scorecaster(model),
        window=window,
        scorecast_window=scorecast_window,
        ahead=ahead,
        learning_rate=learning_rate,
        integral_gain=integral_gain,
        saturation_scale=saturation_scale,
    )
    seed = (
        np.array([2, 5, 3, 6, 4, 7, 5, 8])[:, None, None] * np.array([1, 2])[None, :, None]
        + np.arange(n_columns)[None, None, :]
    )
    state = initial_state(target, method, n_nodes, 3)
    method.seed(state["calibrator"], seed)
    history = [
        [list(seed[-window:, node, column]) for column in range(n_columns)]
        for node in range(n_nodes)
    ]
    scales = np.empty((n_nodes, n_columns))
    proportional = np.empty_like(scales)
    cumulative_error = np.zeros_like(scales)
    updates = np.zeros(scales.shape, dtype=np.int64)
    for node in range(n_nodes):
        for column in range(n_columns):
            values = history[node][column]
            scales[node, column] = max(float(np.std(values)), 0.1 * float(np.mean(values)), 1e-8)
            rank = math.ceil(len(values) * levels[node, column])
            quantile = sorted(values)[rank - 1]
            prediction = reference_scorecast(model, values[-scorecast_window:], ahead[column])
            proportional[node, column] = quantile - prediction

    periods = np.arange(32)
    pattern = np.resize([0, 3, 1, 4], len(periods))
    actuals = np.array(
        [10 + pattern + periods // 4, 20 + 2 * pattern + periods // 3], dtype=np.float32
    )
    origins = [10, 11, 12, 15, 16, 17, 20, 21, 24, 27]
    pending = []
    previous = -1
    for origin in origins:
        waiting = []
        # Each record is an archived issuance; resolve it only at its last covered step.
        for issued_origin, column, archived_point, archived_threshold in pending:
            if issued_origin + int(ahead[column]) > origin:
                waiting.append((issued_origin, column, archived_point, archived_threshold))
                continue
            for node in range(n_nodes):
                target_value = sum(float(actuals[node, issued_origin + h]) for h in covered[column])
                revealed = np.float32(abs(target_value - archived_point[node]))
                error = float(revealed > archived_threshold[node]) - (1 - levels[node, column])
                proportional[node, column] += learning_rate * scales[node, column] * error
                cumulative_error[node, column] += error
                updates[node, column] += 1
                history[node][column] = (history[node][column] + [float(revealed)])[-window:]
        pending = waiting
        expected = np.empty_like(scales, dtype=np.float32)
        for node in range(n_nodes):
            for column in range(n_columns):
                t = int(updates[node, column])
                argument = (
                    cumulative_error[node, column] * math.log(t + 1) / (saturation_scale * (t + 1))
                )
                assert abs(argument) < math.pi / 2
                integral = integral_gain * scales[node, column] * math.tan(argument)
                prediction = reference_scorecast(
                    model,
                    history[node][column][-scorecast_window:],
                    ahead[column],
                )
                expected[node, column] = proportional[node, column] + integral + prediction
        points = np.array([[3, 4, 5], [6, 7, 8]], dtype=np.float32) + origin % 3
        state, issue = step(
            state,
            origin,
            points,
            actuals[:, previous + 1 : origin + 1],
            target=target,
            score=Absolute(),
            calibrator=method,
        )
        np.testing.assert_allclose(issue.threshold, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(state["calibrator"]["proportional"], proportional, rtol=1e-11)
        np.testing.assert_array_equal(state["calibrator"]["n_updates"], updates)
        if model == "none":
            np.testing.assert_array_equal(
                method._scorecast(state["calibrator"]), np.zeros_like(scales)
            )
        for column, steps in enumerate(covered):
            archived_point = np.array(
                [sum(float(points[node, h - 1]) for h in steps) for node in range(n_nodes)]
            )
            pending.append((origin, column, archived_point, expected[:, column].copy()))
        previous = origin
    assert (updates > window).all()  # The oracle covers multiple score-ring wraps.
    assert cumulative_error.any()
    for array in state["calibrator"].values():
        assert isinstance(array, np.ndarray)
        assert array.dtype != object and array.shape[0] == n_nodes


@pytest.mark.parametrize(("target", "ahead"), [(Step(), np.arange(1, 4)), (LeadTime(3), 3)])
@pytest.mark.parametrize("model", ["ar1", "theta"])
def test_pid_replay_is_causal_with_delayed_targets_and_gapped_origins(target, ahead, model):
    origins = np.array([10, 11, 14, 16, 20, 21])
    points = np.zeros((len(origins), 2, 3), dtype=np.float32)
    actuals = np.tile(np.arange(30, dtype=np.float32), (2, 1))
    method = ConformalPID(0.9, window=4, ahead=ahead, scorecaster=scorecaster(model))
    setup: dict[str, Any] = {"target": target, "score": Absolute(), "calibrator": method}

    def run(values):
        state = initial_state(target, method, 2, 3)
        shape = state["calibrator"]["count"].shape
        state["calibrator"] = method.seed(state["calibrator"], np.ones((3, *shape)))
        return replay(Forecasts(origins, points), values, state=state, **setup)

    whole = run(actuals)
    changed = actuals.copy()
    changed[:, 19:] += 1000
    other = run(changed)
    np.testing.assert_array_equal(whole.threshold[origins < 19], other.threshold[origins < 19])
    assert not np.array_equal(whole.threshold[origins >= 19], other.threshold[origins >= 19])
    expected = [5, 4, 4] if isinstance(target, Step) else [4]
    np.testing.assert_array_equal(whole.state["calibrator"]["n_updates"], [expected, expected])
    known = np.isfinite(whole.score)
    inside = (whole.lower <= whole.target) & (whole.target <= whole.upper)
    np.testing.assert_array_equal(whole.covered[known], inside[known])


@pytest.mark.parametrize("model", ["ar1", "theta"])
def test_pid_saved_state_resumes_exactly_with_pending_multi_step_targets(model):
    rng = np.random.default_rng(7)
    actuals = rng.normal(size=(2, 70)).astype(np.float32)
    origins = np.array([10, 11, 14, 16, 20, 23, 25, 29, 32, 36, 40, 45, 50, 55])
    points = rng.normal(size=(len(origins), 2, 4)).astype(np.float32)
    seed = np.abs(rng.normal(size=(5, 2, 4)))
    method = ConformalPID(
        np.array([[0.8], [0.9]]), window=4, ahead=np.arange(1, 5), scorecaster=scorecaster(model)
    )
    setup: dict[str, Any] = {"target": Step(), "score": Absolute(), "calibrator": method}

    def start():
        state = initial_state(Step(), method, 2, 4)
        state["calibrator"] = method.seed(state["calibrator"], seed)
        return state

    whole = replay(Forecasts(origins, points), actuals, state=start(), **setup)
    left = replay(Forecasts(origins[:6], points[:6]), actuals, state=start(), **setup)
    buffer = io.BytesIO()
    arrays: dict[str, Any] = flatten(left.state)
    np.savez(buffer, allow_pickle=False, **arrays)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        state = unflatten(dict(saved))
    resumed = ConformalPID(
        np.array([[0.8], [0.9]]),
        window=4,
        ahead=np.arange(1, 5),
        scorecaster=scorecaster(model),
    )
    right = replay(
        Forecasts(origins[6:], points[6:]),
        actuals,
        state=state,
        target=Step(),
        score=Absolute(),
        calibrator=resumed,
    )
    np.testing.assert_array_equal(
        np.concatenate([left.threshold, right.threshold]), whole.threshold
    )
    for key, value in flatten(whole.state).items():
        np.testing.assert_array_equal(flatten(right.state)[key], value)
