import numpy as np
import pytest

from calibre.backtest import Forecasts, replay
from calibre.conformal.calibrators import ACI, Feedback, QuantileTracker, SplitQuantile
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
        scores=np.array([[score]], dtype=np.float32),
        issued=np.array([[issued]], dtype=np.float32),
        censored=np.array([[False]]),
    )


def test_aci_raises_the_level_after_a_miss_and_lowers_it_after_a_hit():
    aci = ACI(SplitQuantile(0.9), gamma=0.1)
    state = aci.init(1, 1)
    missed = aci.update(state, feedback(0, 5.0, 1.0))
    hit = aci.update(state, feedback(0, 0.5, 1.0))
    # Level moves by gamma * (miss - (1 - level)): +0.09 after a miss, -0.01 after a hit.
    np.testing.assert_allclose(missed["level"], [[0.99]])
    np.testing.assert_allclose(hit["level"], [[0.89]])
    assert np.isnan(state["level"]).all()


def test_aci_threshold_is_infinite_at_level_one_and_empty_at_zero():
    aci = ACI(SplitQuantile(0.9), gamma=1.0)
    state = aci.init(2, 1)
    state["level"] = np.array([[1.2], [-0.1]])
    assert aci.threshold(state).tolist() == [[np.inf], [-np.inf]]


def test_quantile_tracker_moves_its_threshold_without_stored_scores():
    tracker = QuantileTracker(0.9, lr=1.0, start=2.0)
    state = tracker.update(tracker.init(1, 1), feedback(0, 5.0, 2.0))
    assert state["threshold"].tolist() == [[2.9]]
    state = tracker.update(state, feedback(1, 0.0, 2.9))
    np.testing.assert_allclose(state["threshold"], [[2.8]])
    assert list(flatten(state)) == ["threshold"]


def test_split_quantile_capacity_keeps_the_last_origins_and_fills_late_columns():
    split = SplitQuantile(0.5, capacity=2)
    state = split.init(1, 2)
    for origin in range(4):
        state = split.update(state, feedback(origin, float(origin), 0.0))
    state = split.update(state, feedback(3, 7.0, 0.0, column=1))
    size = int(state["size"])
    assert state["origin"][:size].tolist() == [2, 3]
    by_origin = state["scores"][:, :size, 0].T  # [R, C]
    assert np.nan_to_num(by_origin, nan=-1).tolist() == [[2, -1], [3, 7]]
    assert len(state["origin"]) == 2


def test_aci_holds_long_run_coverage_through_a_shift_where_split_does_not():
    rng = np.random.default_rng(0)
    n = 1500
    noise = rng.normal(size=n) * np.where(np.arange(n) < 500, 1.0, 3.0)
    actuals = noise[None].astype(np.float32)
    origins = np.arange(50, n - 1)
    forecasts = Forecasts(origins, np.zeros((len(origins), 1, 1), dtype=np.float32))
    setup = {"target": Step(), "score": Signed()}
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
    state = ACI(SplitQuantile(0.9)).init(3, 2)
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
    setup = {"target": Step(), "score": Signed()}
    both = replay(forecasts, actuals, calibrator=SplitQuantile(level, window=30), **setup)
    for node, scalar in enumerate([0.5, 0.8]):
        alone = replay(forecasts, actuals, calibrator=SplitQuantile(scalar, window=30), **setup)
        np.testing.assert_array_equal(both.threshold[:, node], alone.threshold[:, node])


def test_aci_targets_the_level_of_its_base_per_node():
    aci = ACI(SplitQuantile(np.array([[0.9], [0.5]])), gamma=0.1)
    hits = Feedback(
        origin=np.array([0]),
        column=np.array([0]),
        scores=np.zeros((1, 2), dtype=np.float32),
        issued=np.ones((1, 2), dtype=np.float32),
        censored=np.zeros((1, 2), dtype=bool),
    )
    state = aci.update(aci.init(2, 1), hits)
    # A hit moves each working level by -gamma * (1 - level): -0.01 and -0.05.
    np.testing.assert_allclose(state["level"], [[0.89], [0.45]])


def test_quantile_tracker_moves_each_node_by_its_own_level():
    tracker = QuantileTracker(np.array([[0.9], [0.5]]), lr=1.0)
    misses = Feedback(
        origin=np.array([0]),
        column=np.array([0]),
        scores=np.ones((1, 2), dtype=np.float32),
        issued=np.zeros((1, 2), dtype=np.float32),
        censored=np.zeros((1, 2), dtype=bool),
    )
    state = tracker.update(tracker.init(2, 1), misses)
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
