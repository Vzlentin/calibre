import numpy as np

from calibre.metrics import coverage, interval_score, newsvendor_cost, pinball, width

TARGET = np.array([[1.0, 5.0, np.nan], [3.0, 0.0, 2.0]])
LOWER = np.zeros((2, 3))
UPPER = np.full((2, 3), 2.0)


def test_unknown_targets_do_not_count():
    # Known cells: 1, 5, 3, 0, 2. Inside [0, 2]: 1, 0, 2.
    assert coverage(TARGET, LOWER, UPPER) == 3 / 5
    assert coverage(TARGET, LOWER, UPPER, axis=1).tolist() == [0.5, 2 / 3]
    assert width(TARGET, LOWER, UPPER) == 2.0


def test_interval_score_adds_the_scaled_miss_to_the_width():
    # level 0.8: misses 3 (target 5) and 1 (target 3), times 2 / 0.2 = 10.
    assert np.isclose(interval_score(TARGET, LOWER, UPPER, 0.8), (5 * 2 + 10 * 4) / 5)


def test_newsvendor_cost_is_scaled_pinball_at_the_critical_fractile():
    order = np.full((2, 3), 2.0)
    holding, shortage = 0.2, 1.0
    cost = newsvendor_cost(TARGET, order, holding, shortage)
    # Over: 1, 0, 2, 0 units at 0.2. Under: 3, 1 units at 1.0.
    assert np.isclose(cost, (0.2 * 3 + 4) / 5)
    tau = shortage / (holding + shortage)
    assert np.isclose(cost, (holding + shortage) * pinball(TARGET, order, tau))


def test_an_infinite_bound_gives_an_infinite_width():
    upper = UPPER.copy()
    upper[0, 0] = np.inf
    assert np.isinf(width(TARGET, LOWER, upper))
