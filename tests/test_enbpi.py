import io
from typing import Any

import numpy as np
import pytest

from calibre import Asymmetric, EnbPI, Feedback, Forecasts, LeadTime, SplitQuantile, Step
from calibre.backtest import replay
from calibre.conformal.calibrators.ranks import empirical_interval
from calibre.conformal.scores import Absolute, Signed
from calibre.online import initial_state, step
from calibre.online.state import flatten, unflatten


def feedback(values, points=None, column=0, origin=0):
    values = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return Feedback(
        origin=np.array([origin]),
        column=np.array([column]),
        point=np.zeros_like(values) if points is None else np.asarray(points, dtype=np.float32),
        target=values,
        issued=np.zeros(values.shape + (2,), dtype=np.float32),
        censored=np.zeros_like(values, dtype=bool),
        score=Asymmetric(),
    )


def test_empirical_interval_uses_signed_quantiles_and_searches_both_tails():
    # beta=0: [-10, 0], beta=.25: [-10, 1], beta=.5: [0, 2].
    result = empirical_interval(np.array([[-10.0], [0], [1], [2]]), 0.5, 3)
    np.testing.assert_array_equal(result, [[0, 2]])
    assert result.dtype == np.float32


def test_empirical_interval_ties_choose_the_first_beta():
    # All three candidate widths are 2: [0, 2], [1, 3], [2, 4].
    result = empirical_interval(np.arange(5.0)[:, None], 0.5, 3)
    np.testing.assert_array_equal(result, [[0, 2]])


def test_empirical_interval_handles_holes_empty_columns_and_per_node_levels():
    scores = np.array(
        [
            [-10, -50, np.nan],
            [0, 0, np.nan],
            [1, 1, np.nan],
            [2, 2, np.nan],
            [np.nan, 3, np.nan],
            [np.nan, 4, np.nan],
            [np.nan, 5, np.nan],
            [np.nan, 6, np.nan],
        ]
    )
    result = empirical_interval(scores, np.array([0.5, 0.75, 0.9]), 3)
    np.testing.assert_array_equal(result, [[0, 2], [0, 6], [-np.inf, np.inf]])
    np.testing.assert_array_equal(empirical_interval(np.array([[7.0]]), 0.99), [[7, 7]])
    np.testing.assert_array_equal(
        empirical_interval(np.empty((0, 2)), 0.9), [[-np.inf, np.inf], [-np.inf, np.inf]]
    )


@pytest.mark.parametrize(
    ("scores", "level", "grid", "match"),
    [
        (np.ones(3), 0.5, 3, "scores"),
        (np.array([[np.inf]]), 0.5, 3, "finite"),
        (np.ones((2, 1)), 1.0, 3, "level"),
        (np.ones((2, 1)), 0.5, 1, "beta_grid_size"),
        (np.ones((2, 1)), 0.5, 3.5, "beta_grid_size"),
    ],
)
def test_empirical_interval_validates_inputs(scores, level, grid, match):
    with pytest.raises(ValueError, match=match):
        empirical_interval(scores, level, grid)


def test_enbpi_seed_retains_last_finite_errors_per_node_and_column():
    method = EnbPI(0.5, window=3, beta_grid_size=3)
    seed = np.array(
        [
            [[1, 10], [100, np.nan]],
            [[np.nan, 11], [101, np.nan]],
            [[2, 12], [102, np.nan]],
            [[3, 13], [np.nan, np.nan]],
            [[4, np.nan], [103, np.nan]],
        ],
        dtype=np.float32,
    )
    state = method.seed(method.initial_state(2, 2), seed)
    np.testing.assert_array_equal(state["count"], [[3, 3], [3, 0]])
    np.testing.assert_array_equal(np.sort(state["residuals"][0, 0]), [2, 3, 4])
    np.testing.assert_array_equal(np.sort(state["residuals"][0, 1]), [11, 12, 13])
    np.testing.assert_array_equal(np.sort(state["residuals"][1, 0]), [101, 102, 103])
    assert np.isneginf(method.threshold(state)[1, 1, 0])
    assert np.isposinf(method.threshold(state)[1, 1, 1])
    assert all(isinstance(v, np.ndarray) and v.shape[0] == 2 for v in flatten(state).values())


def test_enbpi_update_uses_signed_error_of_the_archived_center():
    method = EnbPI(0.5, window=4, beta_grid_size=3)
    state = method.seed(method.initial_state(2, 1), np.zeros((4, 2, 1)))
    # Observations -3 and 12 against archived centers 2 and 10: errors -5 and 2.
    state = method.update(state, feedback([-3, 12], [[2, 10]]))
    np.testing.assert_array_equal(state["residuals"][:, 0, 0], [-5, 2])
    before = {key: value.copy() for key, value in state.items()}
    method.threshold(state)
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])
    state = method.update(state, feedback([np.nan, np.inf]))
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_enbpi_per_node_and_column_levels_are_not_pooled():
    samples = np.array([-50, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    seed = np.broadcast_to(samples[:, None, None], (8, 2, 2))
    levels = np.array([[0.5, 0.75], [0.75, 0.5]])
    method = EnbPI(levels, window=8, beta_grid_size=3)
    state = method.seed(method.initial_state(2, 2), seed)
    paired = method.threshold(state)
    for node in range(2):
        for column in range(2):
            alone = EnbPI(float(levels[node, column]), 8, 3)
            single = alone.seed(
                alone.initial_state(1, 1), seed[:, node : node + 1, column : column + 1]
            )
            np.testing.assert_array_equal(paired[node, column], alone.threshold(single)[0, 0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"level": 1.0, "window": 3}, "level"),
        ({"level": np.ones(2), "window": 3}, "shape"),
        ({"level": 0.9, "window": 0}, "window"),
        ({"level": 0.9, "window": True}, "window"),
        ({"level": 0.9, "window": 2.5}, "window"),
        ({"level": 0.9, "window": 3, "beta_grid_size": 1}, "beta_grid"),
    ],
)
def test_enbpi_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EnbPI(**kwargs)


def test_enbpi_seed_validates_shape_and_does_not_seed_twice():
    method = EnbPI(0.9, 3)
    state = method.initial_state(1, 1)
    with pytest.raises(ValueError, match="matching"):
        method.seed(state, np.zeros((3, 2, 1)))
    with pytest.raises(ValueError, match="finite"):
        method.seed(state, np.full((3, 1, 1), np.inf))
    state = method.seed(state, np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="empty"):
        method.seed(state, np.zeros((1, 1, 1)))


def test_asymmetric_bounds_and_feedback_count_both_tails():
    score = Asymmetric()
    points = np.array([[10, 10, 10, 10]], dtype=np.float32)
    parameters = np.broadcast_to([-2, 5], points.shape + (2,))
    lower, upper = score.bound(points, parameters)
    np.testing.assert_array_equal(lower, [[8, 8, 8, 8]])
    np.testing.assert_array_equal(upper, [[15, 15, 15, 15]])
    row = Feedback(
        np.array([0]),
        np.array([0]),
        points,
        np.array([[7, 8, 15, 16]]),
        parameters,
        np.zeros_like(points, dtype=bool),
        score,
    )
    misses, known = row.misses(1)
    np.testing.assert_array_equal(misses[:, 0], [1, 0, 0, 1])
    np.testing.assert_array_equal(known[:, 0], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="shape"):
        score.bound(points, np.zeros_like(points))


@pytest.mark.parametrize(
    ("score", "calibrator"),
    [(Absolute(), EnbPI(0.9, 3)), (Signed(), EnbPI(0.9, 3)), (Asymmetric(), SplitQuantile(0.9))],
)
def test_online_rejects_incompatible_score_and_calibrator(score, calibrator):
    state = initial_state(Step(), calibrator, 2, 1)
    with pytest.raises(ValueError, match="parameter shape"):
        step(
            state,
            3,
            np.zeros((2, 1)),
            np.zeros((2, 4)),
            target=Step(),
            score=score,
            calibrator=calibrator,
        )


def test_replay_keeps_paired_parameters_and_updates_archived_centers():
    origins = np.array([1, 2, 3])
    forecasts = Forecasts(origins, np.array([[[5.0]], [[100.0]], [[0.0]]]))
    actuals = np.array([[0, 0, 8, 103, 0]], dtype=np.float32)
    method = EnbPI(0.5, 4, 3)
    state = initial_state(Step(), method, 1, 1)
    state["calibrator"] = method.seed(
        state["calibrator"], np.array([-4, -1, 0, 2], dtype=np.float32).reshape(4, 1, 1)
    )
    run = replay(
        forecasts, actuals, target=Step(), score=Asymmetric(), calibrator=method, state=state
    )
    assert run.threshold.shape == (3, 1, 1, 2)
    np.testing.assert_array_equal(run.threshold[0, 0, 0], [-4, -1])
    np.testing.assert_array_equal(np.sort(run.state["calibrator"]["residuals"][0, 0]), [0, 2, 3, 3])
    np.testing.assert_array_equal(
        run.covered, ((run.lower <= run.target) & (run.target <= run.upper)).astype(float)
    )


def test_asymmetric_replay_is_causal_with_delayed_and_gapped_origins():
    origins = np.array([10, 11, 14, 16, 20, 23])
    points = np.zeros((len(origins), 2, 3), dtype=np.float32)
    actuals = np.tile(np.arange(30, dtype=np.float32), (2, 1))
    setup: dict[str, Any] = {
        "target": LeadTime(3),
        "score": Asymmetric(),
        "calibrator": EnbPI(0.5, 3, 3),
    }
    run = replay(Forecasts(origins, points), actuals, **setup)
    assert np.isinf(run.threshold[:2]).all()
    # At origin 14: totals 11+12+13 and 12+13+14 are now both known.
    np.testing.assert_array_equal(np.sort(run.state["calibrator"]["residuals"][0, 0]), [48, 54, 66])
    changed = actuals.copy()
    changed[:, 19:] += 1000
    other = replay(Forecasts(origins, points), changed, **setup)
    np.testing.assert_array_equal(run.threshold[origins < 19], other.threshold[origins < 19])


def test_saved_asymmetric_state_resumes_replay_exactly_including_pending_targets():
    rng = np.random.default_rng(42)
    actuals = rng.normal(size=(2, 60)).astype(np.float32)
    origins = np.arange(10, 56)
    points = rng.normal(size=(len(origins), 2, 3)).astype(np.float32)
    setup: dict[str, Any] = {
        "target": Step(),
        "score": Asymmetric(),
        "calibrator": EnbPI(0.75, 7, 5),
    }
    whole = replay(Forecasts(origins, points), actuals, **setup)
    left = replay(Forecasts(origins[:20], points[:20]), actuals, **setup)
    buffer = io.BytesIO()
    flat: dict[str, Any] = flatten(left.state)
    np.savez(buffer, allow_pickle=False, **flat)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        state = unflatten(dict(saved))
    right = replay(Forecasts(origins[20:], points[20:]), actuals, state=state, **setup)
    np.testing.assert_array_equal(
        np.concatenate([left.threshold, right.threshold]), whole.threshold
    )
    for name, value in flatten(whole.state).items():
        np.testing.assert_array_equal(flatten(right.state)[name], value)
