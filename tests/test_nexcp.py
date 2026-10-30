import io
from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np
import pytest

from calibre import (
    ACI,
    Absolute,
    Asymmetric,
    Feedback,
    Forecasts,
    LeadTime,
    NexCP,
    Signed,
    SplitQuantile,
    Step,
    initial_state,
    replay,
    step,
)
from calibre.conformal.calibrators.ranks import weighted_score_quantile
from calibre.online.state import flatten, unflatten


def reference(scores, level, rho, window=None):
    values = [float(value) for value in scores if np.isfinite(value)]
    if window is not None:
        values = values[-window:]
    weights = [Fraction(rho) ** (len(values) - i) for i in range(len(values))]
    target = Fraction(level) * (1 + sum(weights))
    cumulative = Fraction(0)
    for value, weight in sorted(zip(values, weights, strict=True)):
        cumulative += weight
        if cumulative >= target:
            return value
    return np.inf


def feedback(values, column=0, origin=0, score=None, points=None):
    values = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return Feedback(
        origin=np.array([origin]),
        column=np.array([column]),
        point=np.zeros_like(values) if points is None else np.asarray(points, dtype=np.float32),
        target=values,
        issued=np.zeros_like(values),
        censored=np.zeros_like(values, dtype=bool),
        score=Signed() if score is None else score,
    )


def test_weighted_rank_keeps_the_unit_test_mass_and_does_not_interpolate():
    scores = np.array([[4], [1], [3], [2]], dtype=np.float32)
    weights = np.array([0.125, 0.25, 0.5, 0.75])
    # Sorted cumulative masses: .25, 1, 1.5, 1.625. Total including test: 2.625.
    np.testing.assert_array_equal(weighted_score_quantile(scores, weights, 0.5), [3])
    np.testing.assert_array_equal(weighted_score_quantile(scores, weights, 0.6), [4])
    assert np.isposinf(weighted_score_quantile(scores, weights, 0.7)).all()
    assert weighted_score_quantile(scores, weights, 0.5).dtype == np.float32


def test_weighted_rank_includes_exact_boundaries_ties_and_zero_weights():
    scores = np.array([[3.0], [1], [2]])
    weights = np.array([0.5, 1, 0.5])
    np.testing.assert_array_equal(weighted_score_quantile(scores, weights, 0.5), [2])
    np.testing.assert_array_equal(
        weighted_score_quantile(scores, weights, np.nextafter(0.5, 1)), [3]
    )
    np.testing.assert_array_equal(
        weighted_score_quantile(np.array([[0.0], [4], [4]]), np.array([0, 0.25, 0.75]), 0.5),
        [4],
    )
    assert np.isposinf(weighted_score_quantile(scores, np.zeros(3), 0.5)).all()


def test_weighted_rank_handles_per_cell_weights_levels_and_missing_scores():
    scores = np.array([[1, 10, np.nan], [2, np.nan, np.nan], [3, 30, np.nan]])
    weights = np.array([[0.5, 1, 1], [0.5, 1, 1], [1, 1, 1]])
    result = weighted_score_quantile(scores, weights, np.array([0.5, 0.5, 0.9]))
    np.testing.assert_array_equal(result, [3, 30, np.inf])
    assert np.isposinf(weighted_score_quantile(np.empty((0, 2)), np.empty(0), 0.5)).all()


@pytest.mark.parametrize(
    ("scores", "weights", "level", "match"),
    [
        (np.ones(3), np.ones(3), 0.5, "scores"),
        (np.ones((3, 1)), np.ones(2), 0.5, "weights"),
        (np.ones((3, 1)), np.ones((3, 2)), 0.5, "weights"),
        (np.ones((1, 1)), np.array([-0.1]), 0.5, "weights"),
        (np.ones((1, 1)), np.array([1.1]), 0.5, "weights"),
        (np.ones((1, 1)), np.array([np.nan]), 0.5, "weights"),
        (np.ones((1, 1)), np.array([np.inf]), 0.5, "weights"),
        (np.array([[np.inf]]), np.ones(1), 0.5, "finite"),
        (np.array([[-np.inf]]), np.ones(1), 0.5, "finite"),
        (np.ones((1, 1)), np.ones(1), 0, "level"),
        (np.ones((1, 1)), np.ones(1), 1, "level"),
        (np.ones((1, 1)), np.ones(1), np.nan, "level"),
    ],
)
def test_weighted_rank_rejects_invalid_inputs(scores, weights, level, match):
    with pytest.raises(ValueError, match=match):
        weighted_score_quantile(scores, weights, level)


def test_newest_score_has_weight_rho_not_one():
    method = NexCP(0.5, rho=0.5)
    state = method.seed(method.initial_state(1, 1), np.array([[[7.0]]]))
    # Finite mass .5 cannot reach .5 * (1 + .5) = .75.
    assert np.isposinf(method.threshold(state)).all()
    np.testing.assert_array_equal(method.threshold_at(state, 0.25), [[7]])


@pytest.mark.parametrize(("rho", "level"), [(0.5, 0.75), (0.5, 0.5), (0.75, 0.75), (0.9, 0.9)])
def test_strong_decay_remains_unready_even_when_the_limiting_mass_rounds_up(rho, level):
    method = NexCP(level, rho=rho)
    state = method.seed(method.initial_state(1, 1), np.ones((2000, 1, 1)))
    # For every finite n, S/(1+S) < rho. Equality only occurs as n tends to infinity.
    assert np.isposinf(method.threshold(state)).all()
    assert int(state["count"][0, 0]) == 2000


@pytest.mark.parametrize("rho", [0.5, 0.75, 1.0])
@pytest.mark.parametrize("window", [None, 1, 3, 7])
def test_nexcp_matches_an_independent_exact_weighted_cdf(rho, window):
    rng = np.random.default_rng(527)
    samples = rng.integers(-20, 20, size=(17, 2, 2)).astype(np.float32)
    samples[::3, 0, 1] = np.nan
    samples[:, 1, 1] = np.nan
    levels = np.array([[0.25, 0.5], [0.75, 0.9]])
    method = NexCP(levels, rho=rho, window=window)
    state = method.seed(method.initial_state(2, 2), samples)
    expected = np.array(
        [
            [reference(samples[:, n, c], levels[n, c], rho, window) for c in range(2)]
            for n in range(2)
        ]
    )
    np.testing.assert_array_equal(method.threshold(state), expected)
    assert method.threshold(state).dtype == np.float32
    assert all(value.shape[0] == 2 for value in flatten(state).values())
    if window is not None:
        assert state["scores"].shape[-1] <= window


@pytest.mark.parametrize("window", [None, 3])
def test_rho_one_recovers_split_quantile_including_missing_scores_and_delays(window):
    levels = np.array([[0.5, 0.75], [0.25, 0.5]])
    method = NexCP(levels, rho=1, window=window)
    split = SplitQuantile(levels, window=window)
    state, ordinary = method.initial_state(2, 2), split.initial_state(2, 2)
    for origin in range(9):
        for column in range(2):
            row = feedback(
                [origin * (column + 1), np.nan if origin % 3 == 0 else -origin],
                column=column,
                origin=origin,
            )
            state = method.update(state, row)
            ordinary = split.update(ordinary, row)
        np.testing.assert_array_equal(method.threshold(state), split.threshold(ordinary))


def test_nexcp_retains_finite_scores_and_preserves_recency_when_the_ring_wraps():
    samples = np.array(
        [
            [[20, 2], [100, np.nan]],
            [[np.nan, 8], [101, np.nan]],
            [[5, 4], [102, np.nan]],
            [[1, 6], [np.nan, np.nan]],
            [[9, np.nan], [103, np.nan]],
        ],
        dtype=np.float32,
    )
    method = NexCP(0.5, rho=0.75, window=3)
    state = method.seed(method.initial_state(2, 2), samples)
    np.testing.assert_array_equal(state["count"], [[3, 3], [3, 0]])
    np.testing.assert_array_equal(method.threshold(state), [[9, 8], [103, np.inf]])
    before = {key: value.copy() for key, value in state.items()}
    method.threshold(state)
    method.threshold_at(state, 0.25)
    method.update(state, feedback([np.nan, np.inf], origin=100))
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_nexcp_scores_archived_points_and_allows_signed_one_sided_thresholds():
    method = NexCP(0.25, rho=0.75)
    state = method.initial_state(2, 1)
    state = method.update(state, feedback([7, 13], score=Absolute(), points=[[10, 5]]))
    np.testing.assert_array_equal(method.threshold(state), [[3], [8]])
    signed = method.update(method.initial_state(2, 1), feedback([7, 13], points=[[10, 5]]))
    np.testing.assert_array_equal(method.threshold(signed), [[-3], [8]])
    overflow = feedback([np.finfo(np.float32).max] * 2, points=[[-np.finfo(np.float32).max] * 2])
    before = {key: value.copy() for key, value in state.items()}
    method.update(state, overflow)
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_nexcp_can_be_used_as_an_aci_base_without_mutating_calibration_state():
    method = NexCP(0.5, rho=0.75)
    wrapper = ACI(method, gamma=0.1)
    state = wrapper.initial_state(1, 1)
    state["base"] = method.seed(state["base"], np.array([1, 2, 3]).reshape(3, 1, 1))
    state["level"] = np.array([[0.25]])
    np.testing.assert_array_equal(
        wrapper.threshold(state), method.threshold_at(state["base"], 0.25)
    )


@pytest.mark.parametrize("rho", [0.5, 0.75, 1.0])
def test_fixed_weight_split_coverage_by_enumerating_exchangeable_scores(rho):
    covered = 0
    for scores in permutations([1.0, 2.0, 3.0, 4.0]):
        method = NexCP(0.5, rho=rho)
        state = method.seed(method.initial_state(1, 1), np.array(scores[:3]).reshape(3, 1, 1))
        covered += scores[3] <= method.threshold(state)[0, 0]
    assert covered / 24 >= 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"level": 0}, "level"),
        ({"level": 1}, "level"),
        ({"level": np.nan}, "level"),
        ({"level": np.ones(2)}, "shape"),
        ({"level": 0.9, "rho": 0}, "rho"),
        ({"level": 0.9, "rho": -0.1}, "rho"),
        ({"level": 0.9, "rho": 1.1}, "rho"),
        ({"level": 0.9, "rho": np.nan}, "rho"),
        ({"level": 0.9, "rho": np.inf}, "rho"),
        ({"level": 0.9, "rho": True}, "rho"),
        ({"level": 0.9, "rho": [0.9]}, "rho"),
        ({"level": 0.9, "window": 0}, "window"),
        ({"level": 0.9, "window": True}, "window"),
        ({"level": 0.9, "window": 2.5}, "window"),
    ],
)
def test_nexcp_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NexCP(**kwargs)


def test_nexcp_validates_seed_and_alternative_levels():
    method = NexCP(0.9)
    state = method.initial_state(1, 1)
    with pytest.raises(ValueError, match="matching"):
        method.seed(state, np.zeros((3, 2, 1)))
    with pytest.raises(ValueError, match="finite"):
        method.seed(state, np.full((3, 1, 1), np.inf))
    with pytest.raises(ValueError, match="level"):
        method.threshold_at(state, 1)
    state = method.seed(state, np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="empty"):
        method.seed(state, np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="scalar scores"):
        method.update(state, feedback([1], score=Asymmetric()))


def test_online_rejects_paired_scores_for_nexcp():
    method = NexCP(0.9)
    state = initial_state(Step(), method, 1, 1)
    with pytest.raises(ValueError, match="parameter shape"):
        step(
            state,
            0,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            target=Step(),
            score=Asymmetric(),
            calibrator=method,
        )


@pytest.mark.parametrize("target", [Step(), LeadTime(3)])
def test_replay_matches_independent_delayed_calibration_and_is_causal(target):
    origins = np.array([10, 11, 14, 16, 20, 23])
    rng = np.random.default_rng(527)
    points = rng.normal(size=(len(origins), 2, 3)).astype(np.float32)
    actuals = rng.normal(size=(2, 30)).astype(np.float32)
    setup: dict[str, Any] = {
        "target": target,
        "score": Absolute(),
        "calibrator": NexCP(0.5, rho=0.75, window=3),
    }
    run = replay(Forecasts(origins, points), actuals, **setup)
    covers = [[0], [1], [2]] if isinstance(target, Step) else [[0, 1, 2]]
    expected = np.full(run.threshold.shape, np.inf, dtype=np.float32)
    for row, now in enumerate(origins):
        for column, covered in enumerate(covers):
            for node in range(2):
                scores = []
                for prior, issued in enumerate(origins[:row]):
                    if issued + max(covered) + 1 <= now:
                        observed = actuals[node, issued + 1 + np.array(covered)].sum()
                        point = points[prior, node, covered].sum()
                        scores.append(abs(observed - point))
                expected[row, node, column] = reference(scores, 0.5, 0.75, 3)
    np.testing.assert_array_equal(run.threshold, expected)
    changed = actuals.copy()
    changed[:, 19:] += 1000
    other = replay(Forecasts(origins, points), changed, **setup)
    np.testing.assert_array_equal(run.threshold[origins < 19], other.threshold[origins < 19])


@pytest.mark.parametrize("window", [None, 5])
def test_saved_state_resumes_with_a_new_calibrator_and_pending_targets(window):
    rng = np.random.default_rng(527)
    actuals = rng.normal(size=(2, 60)).astype(np.float32)
    origins = np.arange(10, 56)
    points = rng.normal(size=(len(origins), 2, 3)).astype(np.float32)
    setup: dict[str, Any] = {"target": Step(), "score": Absolute()}
    method = NexCP(0.5, rho=0.75, window=window)
    whole = replay(Forecasts(origins, points), actuals, calibrator=method, **setup)
    left = replay(Forecasts(origins[:20], points[:20]), actuals, calibrator=method, **setup)
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **flatten(left.state))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        state = unflatten(dict(saved))
    restored = NexCP(0.5, rho=0.75, window=window)
    right = replay(
        Forecasts(origins[20:], points[20:]), actuals, calibrator=restored, state=state, **setup
    )
    np.testing.assert_array_equal(
        np.concatenate([left.threshold, right.threshold]), whole.threshold
    )
    for name, value in flatten(whole.state).items():
        np.testing.assert_array_equal(flatten(right.state)[name], value)
