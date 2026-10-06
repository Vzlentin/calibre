import math

import numpy as np
import pytest

from calibre.backtest import Forecasts, replay
from calibre.conformal.calibrators import Feedback, MinRisk
from calibre.conformal.losses import Miss, Newsvendor
from calibre.conformal.scores import Absolute, Signed
from calibre.conformal.targets import LeadTime
from calibre.online.state import flatten


def rows(point: np.ndarray, target: np.ndarray, first: int = 0) -> Feedback:
    """Known targets `[K, N]` of column 0, one row per origin."""
    k = len(target)
    return Feedback(
        origin=np.arange(first, first + k),
        column=np.zeros(k, dtype=np.int64),
        point=point.astype(np.float32),
        target=target.astype(np.float32),
        issued=np.zeros(target.shape, dtype=np.float32),
        censored=np.zeros(target.shape, dtype=bool),
        score=Signed(),
    )


def test_newsvendor_risk_minimum_is_the_empirical_critical_ratio_quantile():
    rng = np.random.default_rng(0)
    point = rng.integers(0, 20, size=(37, 3))
    target = rng.integers(0, 20, size=(37, 3))
    scores = target - point
    grid = np.arange(-25, 26)
    calibrator = MinRisk(Newsvendor(holding=1.0, shortage=3.0), grid)
    state = calibrator.initial_state(3, 1)
    state = calibrator.update(state, rows(point[:20], target[:20]))
    state = calibrator.update(state, rows(point[20:], target[20:], first=20))
    # Pinball at 3 / (1 + 3) is minimized at rank ceil(37 * 0.75) = 28 of the scores.
    expected = np.sort(scores, axis=0)[27]
    assert calibrator.threshold(state)[:, 0].tolist() == expected.tolist()


def test_a_per_node_grid_scales_the_threshold_with_the_node():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 10, size=(25, 1)).astype(float)
    scale = np.array([1.0, 10.0])
    base = np.arange(-12, 13, dtype=float)
    calibrator = MinRisk(Newsvendor(1.0, 1.0), base[None] * scale[:, None])
    state = calibrator.update(
        calibrator.initial_state(2, 1), rows(np.zeros((25, 2)), target * scale)
    )
    median = np.sort(target[:, 0])[12]
    assert calibrator.threshold(state)[:, 0].tolist() == [median, 10 * median]


def test_min_risk_is_not_ready_before_a_known_target_and_ignores_missing_ones():
    calibrator = MinRisk(Newsvendor(1.0, 1.0), np.arange(5.0))
    state = calibrator.initial_state(2, 1)
    assert np.isinf(calibrator.threshold(state)).all()
    state = calibrator.update(state, rows(np.zeros((1, 2)), np.array([[3.0, np.nan]])))
    assert state["count"].tolist() == [[1], [0]]
    assert calibrator.threshold(state).tolist() == [[3.0], [np.inf]]
    assert sorted(flatten(state)) == ["count", "risk"]


def test_replay_issues_from_the_lead_time_targets_known_at_each_origin():
    rng = np.random.default_rng(2)
    actuals = rng.integers(0, 6, size=(2, 120)).astype(np.float32)
    origins = np.arange(30, 110)
    points = np.full((len(origins), 2, 4), 2.0, dtype=np.float32)
    calibrator = MinRisk(Newsvendor(1.0, 3.0), np.arange(-10, 31))
    run = replay(
        Forecasts(origins, points),
        actuals,
        target=LeadTime(3),
        score=Signed(),
        calibrator=calibrator,
    )
    for index, origin in enumerate(origins):
        # A three-step target issued at o is known at o + 3.
        known = run.score[: max(index - 2, 0), :, 0]
        if len(known) == 0:
            assert np.isinf(run.threshold[index]).all()
            continue
        rank = math.ceil(len(known) * 0.75)
        expected = np.sort(known, axis=0)[rank - 1]
        assert run.threshold[index, :, 0].tolist() == expected.tolist(), origin
    np.testing.assert_array_equal(run.upper, run.point + run.threshold)


def test_losses_score_bounds_against_known_targets():
    target = np.array([1.0, 4.0, np.nan])
    lower, upper = Absolute().bound(np.full(3, 2.0), np.array(1.5))
    censored = np.zeros(3, dtype=bool)
    assert np.nan_to_num(Miss().loss(lower, upper, target, censored), nan=-1).tolist() == [
        0.0,
        1.0,
        -1.0,
    ]
    cost = Newsvendor(holding=0.2, shortage=1.0).loss(lower, upper, target, censored)
    np.testing.assert_allclose(cost[:2], [0.2 * 2.5, 1.0 * 0.5])
    assert np.isnan(cost[2])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: MinRisk(Miss(), np.array([1.0, 1.0])), "increasing"),
        (lambda: MinRisk(Miss(), np.array([0.0, np.inf])), "finite"),
        (lambda: MinRisk(Miss(), np.zeros((2, 2, 2))), "grid"),
        (lambda: Newsvendor(-1.0, 1.0), "negative"),
    ],
)
def test_risk_settings_are_validated(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_a_per_node_grid_must_match_the_nodes():
    with pytest.raises(ValueError, match="rows"):
        MinRisk(Miss(), np.zeros((3, 1))).initial_state(2, 1)
