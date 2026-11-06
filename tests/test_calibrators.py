from typing import Any

import numpy as np
import pytest

from calibre.backtest import Forecasts, replay
from calibre.conformal.calibrators import ACI, Feedback, NexCP, QuantileTracker, SplitQuantile
from calibre.conformal.scores import Signed
from calibre.conformal.targets import Step
from calibre.decision import critical_ratio
from calibre.metrics import coverage
from calibre.online.state import flatten, unflatten


def feedback(origin: int, score: float, issued: float, column: int = 0) -> Feedback:
    """One known score for one node."""
    return Feedback(
        origin=np.array([origin]),
        column=np.array([column]),
        point=np.zeros((1, 1), dtype=np.float32),
        target=np.array([[score]], dtype=np.float32),
        issued=np.array([[issued]], dtype=np.float32),
        censored=np.array([[False]]),
        score=Signed(),
    )


def test_aci_raises_the_level_after_a_miss_and_lowers_it_after_a_hit():
    aci = ACI(SplitQuantile(0.9), gamma=0.1)
    state = aci.initial_state(1, 1)
    missed = aci.update(state, feedback(0, 5.0, 1.0))
    hit = aci.update(state, feedback(0, 0.5, 1.0))
    # Level moves by gamma * (miss - (1 - level)): +0.09 after a miss, -0.01 after a hit.
    np.testing.assert_allclose(missed["level"], [[0.99]])
    np.testing.assert_allclose(hit["level"], [[0.89]])
    assert np.isnan(state["level"]).all()


def test_aci_threshold_is_infinite_at_level_one_and_empty_at_zero():
    aci = ACI(SplitQuantile(0.9), gamma=1.0)
    state = aci.initial_state(2, 1)
    state["level"] = np.array([[1.2], [-0.1]])
    assert aci.threshold(state).tolist() == [[np.inf], [-np.inf]]


def test_quantile_tracker_moves_its_threshold_without_stored_scores():
    tracker = QuantileTracker(0.9, lr=1.0, start=2.0)
    state = tracker.update(tracker.initial_state(1, 1), feedback(0, 5.0, 2.0))
    assert state["threshold"].tolist() == [[2.9]]
    state = tracker.update(state, feedback(1, 0.0, 2.9))
    np.testing.assert_allclose(state["threshold"], [[2.8]])
    assert list(flatten(state)) == ["threshold"]


def test_split_quantile_capacity_keeps_the_last_origins_and_fills_late_columns():
    split = SplitQuantile(0.5, capacity=2)
    state = split.initial_state(1, 2)
    for origin in range(4):
        state = split.update(state, feedback(origin, float(origin), 0.0))
    state = split.update(state, feedback(3, 7.0, 0.0, column=1))
    size = int(state["size"])
    assert state["origin"][:size].tolist() == [2, 3]
    by_origin = state["scores"][:, :size, 0].T  # [R, C]
    assert np.nan_to_num(by_origin, nan=-1).tolist() == [[2, -1], [3, 7]]
    assert len(state["origin"]) == 2


@pytest.mark.parametrize("method", [SplitQuantile(0.5, window=2), NexCP(0.5, rho=0.75, window=2)])
def test_split_seed_keeps_held_out_origins_before_online_feedback(method):
    samples = np.array(
        [
            [[100, 10], [200, 20]],
            [[1, 11], [np.nan, 21]],
            [[np.nan, 12], [3, np.nan]],
            [[4, 13], [5, 23]],
        ],
        dtype=np.float32,
    )
    state = method.seed(method.initial_state(2, 2), samples)
    assert sorted(state) == ["known", "origin", "scores", "size"]
    np.testing.assert_array_equal(state["origin"], [-4, -3, -2, -1])
    np.testing.assert_array_equal(state["known"], np.ones((4, 2), dtype=bool))
    np.testing.assert_array_equal(method.threshold(state), [[4, 13], [5, 23]])
    samples[:] = 999
    np.testing.assert_array_equal(method.threshold(state), [[4, 13], [5, 23]])
    for column, values in enumerate([[6, 7], [14, 24]]):
        row = Feedback(
            origin=np.array([0]),
            column=np.array([column]),
            point=np.zeros((1, 2), dtype=np.float32),
            target=np.array([values], dtype=np.float32),
            issued=np.zeros((1, 2), dtype=np.float32),
            censored=np.zeros((1, 2), dtype=bool),
            score=Signed(),
        )
        state = method.update(state, row)
    np.testing.assert_array_equal(state["origin"][: int(state["size"])], [-4, -3, -2, -1, 0])
    np.testing.assert_array_equal(method.threshold(state), [[6, 14], [7, 24]])


def test_split_seed_preserves_grouped_resolved_row_retention():
    method = SplitQuantile(0.5, window=2, groups=np.array([0, 0]))
    samples = np.array([[[100], [200]], [[1], [np.nan]], [[np.nan], [3]], [[4], [5]]])
    state = method.seed(method.initial_state(2, 1), samples)
    np.testing.assert_array_equal(method.threshold(state), [[4], [4]])


@pytest.mark.parametrize("method", [SplitQuantile(0.5), NexCP(0.5)])
def test_split_validates_seed_and_alternative_levels(method):
    state = method.initial_state(1, 2)
    with pytest.raises(ValueError, match="level"):
        method.threshold_at(state, 1)
    with pytest.raises(ValueError, match="matching"):
        method.seed(state, np.zeros((3, 2, 1)))
    with pytest.raises(ValueError, match="finite"):
        method.seed(state, np.full((3, 1, 2), np.inf))
    state = method.seed(state, np.empty((0, 1, 2)))
    assert np.isposinf(method.threshold(state)).all()
    state = method.update(state, feedback(0, np.nan, 0))
    with pytest.raises(ValueError, match="empty"):
        method.seed(state, np.ones((1, 1, 2)))


@pytest.mark.parametrize(
    "method",
    [SplitQuantile(0.5, window=2, capacity=4), NexCP(0.5, rho=0.75, window=2, capacity=4)],
)
def test_split_capacity_bounds_seed_and_subsequent_origins(method):
    samples = np.arange(7, dtype=np.float32).reshape(7, 1, 1)
    state = method.seed(method.initial_state(1, 1), samples)
    np.testing.assert_array_equal(state["origin"], [-4, -3, -2, -1])
    np.testing.assert_array_equal(state["scores"], [[[3], [4], [5], [6]]])
    for origin in range(6):
        state = method.update(state, feedback(origin, origin + 7, 0))
        assert int(state["size"]) == len(state["origin"]) == 4
    np.testing.assert_array_equal(state["origin"], [2, 3, 4, 5])
    np.testing.assert_array_equal(method.threshold(state), [[12]])


@pytest.mark.parametrize("method_type", [SplitQuantile, NexCP])
def test_capacity_can_discard_finite_scores_across_missing_origins(method_type):
    samples = np.array([9, np.nan, 1, np.nan]).reshape(4, 1, 1)
    unbounded = method_type(0.6, window=2)
    bounded = method_type(0.6, window=2, capacity=3)
    full = unbounded.seed(unbounded.initial_state(1, 1), samples)
    limited = bounded.seed(bounded.initial_state(1, 1), samples)
    np.testing.assert_array_equal(unbounded.threshold(full), [[9]])
    assert np.isposinf(bounded.threshold(limited)).all()


def test_aci_holds_long_run_coverage_through_a_shift_where_split_does_not():
    rng = np.random.default_rng(0)
    n = 1500
    noise = rng.normal(size=n) * np.where(np.arange(n) < 500, 1.0, 3.0)
    actuals = noise[None].astype(np.float32)
    origins = np.arange(50, n - 1)
    forecasts = Forecasts(origins, np.zeros((len(origins), 1, 1), dtype=np.float32))
    setup: dict[str, Any] = {"target": Step(), "score": Signed()}
    split = replay(forecasts, actuals, calibrator=SplitQuantile(0.9), **setup)
    aci = replay(forecasts, actuals, calibrator=ACI(SplitQuantile(0.9), gamma=0.01), **setup)
    after = slice(500, None)
    split_coverage = coverage(split.target[after], split.lower[after], split.upper[after])
    aci_coverage = coverage(aci.target[after], aci.lower[after], aci.upper[after])
    assert abs(aci_coverage - 0.9) < 0.02
    assert split_coverage < 0.85


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: SplitQuantile(0.9, window=0), "window"),
        (lambda: SplitQuantile(0.9, window=5, capacity=3), "capacity"),
        (lambda: ACI(SplitQuantile(0.9), gamma=0), "gamma"),
        (lambda: QuantileTracker(0.9, lr=0), "lr"),
    ],
)
def test_calibrators_reject_invalid_settings(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_state_flattens_to_named_arrays_and_back():
    state = ACI(SplitQuantile(0.9)).initial_state(3, 2)
    flat = flatten(state)
    assert sorted(flat) == ["base/known", "base/origin", "base/scores", "base/size", "level"]
    back = unflatten(flat)
    assert back["base"]["scores"].shape == (2, 0, 3)
    with pytest.raises(ValueError, match="contains"):
        flatten({"a/b": np.zeros(1)})


def test_a_per_node_level_gives_each_node_the_threshold_of_its_own_level():
    rng = np.random.default_rng(3)
    actuals = rng.normal(size=(1, 200)).repeat(2, axis=0).astype(np.float32)
    origins = np.arange(20, 190)
    forecasts = Forecasts(origins, np.zeros((len(origins), 2, 2), dtype=np.float32))
    # Shortage costs 1 for both nodes, holding 1 and 0.25: levels 0.5 and 0.8.
    level = critical_ratio(np.array([[1.0], [0.25]]), 1.0)
    setup: dict[str, Any] = {"target": Step(), "score": Signed()}
    both = replay(forecasts, actuals, calibrator=SplitQuantile(level, window=30), **setup)
    for node, scalar in enumerate([0.5, 0.8]):
        alone = replay(forecasts, actuals, calibrator=SplitQuantile(scalar, window=30), **setup)
        np.testing.assert_array_equal(both.threshold[:, node], alone.threshold[:, node])


def test_aci_targets_the_level_of_its_base_per_node():
    aci = ACI(SplitQuantile(np.array([[0.9], [0.5]])), gamma=0.1)
    hits = Feedback(
        origin=np.array([0]),
        column=np.array([0]),
        point=np.zeros((1, 2), dtype=np.float32),
        target=np.zeros((1, 2), dtype=np.float32),
        issued=np.ones((1, 2), dtype=np.float32),
        censored=np.zeros((1, 2), dtype=bool),
        score=Signed(),
    )
    state = aci.update(aci.initial_state(2, 1), hits)
    # A hit moves each working level by -gamma * (1 - level): -0.01 and -0.05.
    np.testing.assert_allclose(state["level"], [[0.89], [0.45]])


def test_quantile_tracker_moves_each_node_by_its_own_level():
    tracker = QuantileTracker(np.array([[0.9], [0.5]]), lr=1.0)
    misses = Feedback(
        origin=np.array([0]),
        column=np.array([0]),
        point=np.zeros((1, 2), dtype=np.float32),
        target=np.ones((1, 2), dtype=np.float32),
        issued=np.zeros((1, 2), dtype=np.float32),
        censored=np.zeros((1, 2), dtype=bool),
        score=Signed(),
    )
    state = tracker.update(tracker.initial_state(2, 1), misses)
    np.testing.assert_allclose(state["threshold"], [[0.9], [0.5]])


@pytest.mark.parametrize(
    ("level", "match"),
    [(1.0, "between"), (0.0, "between"), (np.nan, "between"), (np.full(3, 0.9), "shape")],
)
def test_calibrators_reject_invalid_levels(level, match):
    with pytest.raises(ValueError, match=match):
        SplitQuantile(level)
    with pytest.raises(ValueError, match=match):
        QuantileTracker(level, lr=1.0)
