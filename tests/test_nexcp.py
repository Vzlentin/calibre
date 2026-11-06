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
    critical_ratio,
    initial_state,
    replay,
)
from calibre.conformal.calibrators.ranks import _weighted_quantile, recency_quantile
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
    weights = np.array([[0.125], [0.25], [0.5], [0.75]])
    # Sorted cumulative masses: .25, 1, 1.5, 1.625. Total including test: 2.625.
    np.testing.assert_array_equal(_weighted_quantile(scores, weights, 0.5), [3])
    np.testing.assert_array_equal(_weighted_quantile(scores, weights, 0.6), [4])
    assert np.isposinf(_weighted_quantile(scores, weights, 0.7)).all()
    assert _weighted_quantile(scores, weights, 0.5).dtype == np.float32


def test_weighted_rank_includes_exact_boundaries_ties_and_zero_weights():
    scores = np.array([[3.0], [1], [2]])
    weights = np.array([[0.5], [1], [0.5]])
    np.testing.assert_array_equal(_weighted_quantile(scores, weights, 0.5), [2])
    np.testing.assert_array_equal(_weighted_quantile(scores, weights, np.nextafter(0.5, 1)), [3])
    np.testing.assert_array_equal(
        _weighted_quantile(np.array([[0.0], [4], [4]]), np.array([[0], [0.25], [0.75]]), 0.5),
        [4],
    )
    assert np.isposinf(_weighted_quantile(scores, np.zeros_like(scores), 0.5)).all()


def test_weighted_rank_handles_per_cell_weights_levels_and_missing_scores():
    scores = np.array([[1, 10, np.nan], [2, np.nan, np.nan], [3, 30, np.nan]])
    weights = np.array([[0.5, 1, 0], [0.5, 0, 0], [1, 1, 0]])
    result = _weighted_quantile(scores, weights, np.array([0.5, 0.5, 0.9]))
    np.testing.assert_array_equal(result, [3, 30, np.inf])
    assert np.isposinf(_weighted_quantile(np.empty((0, 2)), np.empty((0, 2)), 0.5)).all()


def test_newest_score_has_weight_rho_not_one():
    method = NexCP(0.25, rho=0.5)
    state = method.seed(method.initial_state(1, 1), np.array([[[7.0]]]))
    # Finite mass .5 cannot reach .5 * (1 + .5) = .75.
    assert np.isposinf(method.threshold_at(state, 0.5)).all()
    np.testing.assert_array_equal(method.threshold(state), [[7]])


@pytest.mark.parametrize(("rho", "level"), [(0.5, 0.75), (0.5, 0.5), (0.75, 0.75), (0.9, 0.9)])
def test_alternative_levels_remain_unready_when_the_limiting_mass_rounds_up(rho, level):
    method = NexCP(0.25, rho=rho)
    state = method.seed(method.initial_state(1, 1), np.ones((2000, 1, 1)))
    # For every finite n, S/(1+S) < rho. Equality only occurs as n tends to infinity.
    assert np.isposinf(method.threshold_at(state, level)).all()
    assert int(state["size"]) == 2000


@pytest.mark.parametrize("rho", [0.5, 0.75, 1.0])
@pytest.mark.parametrize("window", [None, 1, 3, 7])
def test_nexcp_matches_an_independent_exact_weighted_cdf(rho, window):
    rng = np.random.default_rng(527)
    samples = rng.integers(-20, 20, size=(17, 2, 2)).astype(np.float32)
    samples[::3, 0, 1] = np.nan
    samples[:, 1, 1] = np.nan
    levels = np.array([[0.25, 0.375], [0.125, 0.25]])
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
    for column in range(2):
        np.testing.assert_array_equal(
            recency_quantile(samples[:, :, column], levels[:, column], window=window, rho=rho),
            expected[:, column],
        )
    assert sorted(state) == ["known", "origin", "scores", "size"]


def test_resolved_age_weighting_matches_exact_cdf_across_random_histories():
    rng = np.random.default_rng(528)
    for _ in range(200):
        scores = rng.integers(-20, 20, size=(rng.integers(0, 30), 3)).astype(np.float32)
        scores[rng.random(scores.shape) < 0.3] = np.nan
        rho = float(rng.choice([0.3, 0.5, 0.75, 0.9, 1.0]))
        window = [None, 1, 3, 7][int(rng.integers(4))]
        levels = rng.uniform(0.1, 0.95, size=3)
        expected = [reference(scores[:, node], levels[node], rho, window) for node in range(3)]
        np.testing.assert_array_equal(
            recency_quantile(scores, levels, window=window, rho=rho), expected
        )


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
        for name in ordinary:
            np.testing.assert_array_equal(state[name], ordinary[name])


def test_nexcp_applies_the_finite_score_window_without_mutating_split_state():
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
    assert int(state["size"]) == len(samples)
    np.testing.assert_array_equal(state["scores"], samples.transpose(2, 0, 1))
    np.testing.assert_array_equal(method.threshold(state), [[9, 8], [103, np.inf]])
    before = {key: value.copy() for key, value in state.items()}
    method.threshold(state)
    method.threshold_at(state, 0.25)
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])
    state = method.update(state, feedback([np.nan, np.inf], origin=100))
    np.testing.assert_array_equal(method.threshold(state), [[9, 8], [103, np.inf]])


def test_nexcp_scores_archived_points_and_allows_signed_one_sided_thresholds():
    method = NexCP(0.25, rho=0.75)
    state = method.initial_state(2, 1)
    state = method.update(state, feedback([7, 13], score=Absolute(), points=[[10, 5]]))
    np.testing.assert_array_equal(method.threshold(state), [[3], [8]])
    signed = method.update(method.initial_state(2, 1), feedback([7, 13], points=[[10, 5]]))
    np.testing.assert_array_equal(method.threshold(signed), [[-3], [8]])
    overflow = feedback([np.finfo(np.float32).max] * 2, points=[[-np.finfo(np.float32).max] * 2])
    before = method.threshold(state)
    with np.errstate(over="ignore", invalid="ignore"):
        state = method.update(state, feedback(overflow.target[0], origin=1, points=overflow.point))
    np.testing.assert_array_equal(method.threshold(state), before)


@pytest.mark.parametrize(("working_level", "expected"), [(0.25, 2), (0.75, np.inf), (0.9, np.inf)])
def test_nexcp_can_be_used_as_an_aci_base_without_mutating_calibration_state(
    working_level, expected
):
    method = NexCP(0.5, rho=0.75)
    wrapper = ACI(method, gamma=0.1)
    state = wrapper.initial_state(1, 1)
    state["base"] = method.seed(state["base"], np.array([1, 2, 3]).reshape(3, 1, 1))
    state["level"] = np.array([[working_level]])
    before = {name: value.copy() for name, value in state["base"].items()}
    np.testing.assert_array_equal(wrapper.threshold(state), [[expected]])
    for name, value in before.items():
        np.testing.assert_array_equal(state["base"][name], value)
    np.testing.assert_array_equal(method.level, 0.5)


@pytest.mark.parametrize(("rho", "level"), [(0.5, 0.25), (0.75, 0.5), (1.0, 0.5)])
def test_fixed_weight_split_coverage_by_enumerating_exchangeable_scores(rho, level):
    covered = 0
    for scores in permutations([1.0, 2.0, 3.0, 4.0]):
        method = NexCP(level, rho=rho)
        state = method.seed(method.initial_state(1, 1), np.array(scores[:3]).reshape(3, 1, 1))
        covered += scores[3] <= method.threshold(state)[0, 0]
    assert covered / 24 >= level


@pytest.mark.parametrize("rho", [0, -0.1, 1.1, np.nan, np.inf])
def test_nexcp_rejects_invalid_rho(rho):
    with pytest.raises(ValueError, match="rho"):
        NexCP(0.9, rho=rho)


@pytest.mark.parametrize(
    ("level", "rho"),
    [
        (0.99, 0.99),
        (0.999, 0.99),
        (0.75, 0.75),
        (0.9, 0.75),
        (np.array([[0.25, 0.5], [0.5, 0.75]]), 0.75),
    ],
)
def test_nexcp_rejects_configured_levels_at_or_above_rho(level, rho):
    with pytest.raises(ValueError, match="level must be strictly below rho"):
        NexCP(level, rho=rho)


def test_default_rho_rejects_unattainable_cost_calibrated_levels():
    levels = critical_ratio(np.array([[0.01], [0.005]]), 1.0)
    with pytest.raises(ValueError, match="level must be strictly below rho"):
        NexCP(levels)


@pytest.mark.parametrize("rho", [0.5, 0.99, 1.0])
def test_nexcp_accepts_configured_levels_strictly_below_rho(rho):
    level = np.nextafter(rho, 0)
    method = NexCP(level, rho=rho)
    np.testing.assert_array_equal(method.level, level)


def test_online_rejects_paired_scores_for_nexcp():
    with pytest.raises(ValueError, match="parameter shape"):
        initial_state(Step(), NexCP(0.9), 1, 1, score=Asymmetric())


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


@pytest.mark.parametrize(("window", "capacity"), [(None, None), (5, None), (5, 10)])
def test_saved_state_resumes_with_a_new_calibrator_and_pending_targets(window, capacity):
    rng = np.random.default_rng(527)
    actuals = rng.normal(size=(2, 60)).astype(np.float32)
    origins = np.arange(10, 56)
    points = rng.normal(size=(len(origins), 2, 3)).astype(np.float32)
    setup: dict[str, Any] = {"target": Step(), "score": Absolute()}
    method = NexCP(0.5, rho=0.75, window=window, capacity=capacity)
    whole = replay(Forecasts(origins, points), actuals, calibrator=method, **setup)
    left = replay(Forecasts(origins[:20], points[:20]), actuals, calibrator=method, **setup)
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **flatten(left.state))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        state = unflatten(dict(saved))
    restored = NexCP(0.5, rho=0.75, window=window, capacity=capacity)
    right = replay(
        Forecasts(origins[20:], points[20:]), actuals, calibrator=restored, state=state, **setup
    )
    np.testing.assert_array_equal(
        np.concatenate([left.threshold, right.threshold]), whole.threshold
    )
    for name, value in flatten(whole.state).items():
        np.testing.assert_array_equal(flatten(right.state)[name], value)
