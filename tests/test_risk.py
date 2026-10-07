import numpy as np
import pytest

from calibre.backtest import Forecasts, replay
from calibre.conformal.calibrators import Feedback, RiskControl, SplitQuantile
from calibre.conformal.losses import Miss
from calibre.conformal.scores import Absolute, Signed
from calibre.conformal.targets import LeadTime
from calibre.online.state import flatten


def rows(point: np.ndarray, target: np.ndarray) -> Feedback:
    """Known targets `[K, N]` of column 0, one row per origin."""
    k = len(target)
    return Feedback(
        origin=np.arange(k),
        column=np.zeros(k, dtype=np.int64),
        point=point.astype(np.float32),
        target=target.astype(np.float32),
        issued=np.zeros(target.shape, dtype=np.float32),
        censored=np.zeros(target.shape, dtype=bool),
        score=Signed(),
    )


def test_miss_is_one_outside_the_bounds():
    lower, upper = Absolute().bound(np.full(3, 2.0), np.array(1.5))
    assert Miss().loss(lower, upper, np.array([1.0, 4.0, 0.0])).tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("n", [3, 37])
def test_risk_control_of_misses_is_the_split_conformal_threshold(n):
    rng = np.random.default_rng(3)
    point = rng.integers(0, 20, size=(n, 3))
    target = rng.integers(0, 20, size=(n, 3))
    feedback = rows(point, target)
    control = RiskControl(Miss(), np.arange(-25, 26), alpha=0.2)
    split = SplitQuantile(0.8)
    controlled = control.threshold(control.update(control.initial_state(3, 1), feedback))
    expected = split.threshold(split.update(split.initial_state(3, 1), feedback))
    # n = 3 is too few for a level of 0.8: both are not ready.
    assert controlled.tolist() == expected.tolist()


def test_risk_control_issues_the_smallest_grid_threshold_with_a_corrected_risk_below_alpha():
    calibrator = RiskControl(Miss(), np.array([0.0, 1.0, 2.0, 3.0]), alpha=0.3)
    # Scores 0.5, 1.5, 2.5. Misses at 1 and 2: 2 and 1, so (2 + 1) / 4 > 0.3 and
    # (1 + 1) / 4 > 0.3. At 3: (0 + 1) / 4 <= 0.3.
    feedback = rows(np.zeros((3, 1)), np.array([[0.5], [1.5], [2.5]]))
    state = calibrator.update(calibrator.initial_state(1, 1), feedback)
    assert calibrator.threshold(state).tolist() == [[3.0]]


def test_a_per_node_grid_scales_the_threshold_with_the_node():
    target = np.arange(9.0)[:, None]
    scale = np.array([1.0, 10.0])
    base = np.arange(10.0)
    calibrator = RiskControl(Miss(), base[None] * scale[:, None], alpha=0.2)
    state = calibrator.update(
        calibrator.initial_state(2, 1), rows(np.zeros((9, 2)), target * scale)
    )
    # (misses + 1) / 10 <= 0.2 first at threshold 7: one miss, the target 8.
    assert calibrator.threshold(state)[:, 0].tolist() == [7.0, 70.0]


def test_risk_control_ignores_missing_targets_and_keeps_only_sums():
    calibrator = RiskControl(Miss(), np.arange(5.0), alpha=0.5)
    state = calibrator.initial_state(2, 1)
    assert np.isinf(calibrator.threshold(state)).all()
    state = calibrator.update(state, rows(np.zeros((1, 2)), np.array([[3.0, np.nan]])))
    assert state["count"].tolist() == [[1], [0]]
    # Node 0: (0 + 1) / 2 <= 0.5 from threshold 3. Node 1: 1 / 1 > 0.5 everywhere.
    assert calibrator.threshold(state).tolist() == [[3.0], [np.inf]]
    assert sorted(flatten(state)) == ["count", "loss_sum"]


def test_replay_with_risk_control_of_misses_matches_split_quantile():
    rng = np.random.default_rng(2)
    actuals = rng.integers(0, 6, size=(2, 120)).astype(np.float32)
    origins = np.arange(30, 110)
    points = np.full((len(origins), 2, 4), 2.0, dtype=np.float32)
    forecasts = Forecasts(origins, points)
    run = {
        name: replay(forecasts, actuals, target=LeadTime(3), score=Signed(), calibrator=c)
        for name, c in [
            ("control", RiskControl(Miss(), np.arange(-10, 31), alpha=0.25)),
            ("split", SplitQuantile(0.75)),
        ]
    }
    np.testing.assert_array_equal(run["control"].threshold, run["split"].threshold)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: RiskControl(Miss(), np.array([1.0, 1.0]), 0.1), "increasing"),
        (lambda: RiskControl(Miss(), np.array([0.0, np.inf]), 0.1), "finite"),
        (lambda: RiskControl(Miss(), np.zeros((2, 2, 2)), 0.1), "grid"),
        (lambda: RiskControl(Miss(), np.arange(3.0), 1.0), "alpha"),
        (lambda: RiskControl(Miss(), np.zeros((3, 1)), 0.1).initial_state(2, 1), "rows"),
    ],
)
def test_risk_settings_are_validated(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
