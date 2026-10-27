import io
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from calibre import SPCI, Absolute, Asymmetric, Feedback, Forecasts, LeadTime, Step
from calibre.backtest import replay
from calibre.online import initial_state, step
from calibre.online.state import flatten, unflatten


def method(**kwargs: Any) -> SPCI:
    configuration: dict[str, Any] = dict(
        level=0.5,
        window=12,
        n_lags=1,
        ahead=1,
        min_train_samples=1,
        beta_grid_size=3,
        n_estimators=3,
        min_samples_leaf=100,
        bootstrap=False,
    )
    return SPCI(**(configuration | kwargs))


def seeded(calibrator, values, origins=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None, None]
    return calibrator.seed(
        calibrator.initial_state(*values.shape[1:]),
        values,
        origins=np.arange(len(values)) if origins is None else np.asarray(origins),
    )


def feedback(values, *, origin, column=0, points=None, score=None):
    targets = np.asarray(values, dtype=np.float32)[None, :]
    return Feedback(
        origin=np.array([origin]),
        column=np.array([column]),
        point=np.zeros_like(targets) if points is None else np.asarray(points, dtype=np.float32),
        target=targets,
        issued=np.zeros((*targets.shape, 2), dtype=np.float32),
        censored=np.zeros_like(targets, dtype=bool),
        score=Asymmetric() if score is None else score,
    )


def test_empty_and_short_histories_are_unready_without_a_marginal_fallback():
    calibrator = method(n_lags=2, min_train_samples=3, ahead=3)
    state = calibrator.initial_state(2, 1)
    np.testing.assert_array_equal(calibrator.threshold(state), [[[-np.inf, np.inf]]] * 2)
    # Six errors allow only two contexts: targets at origins 4 and 5.
    state = calibrator.seed(state, np.zeros((6, 2, 1)), origins=np.arange(6))
    assert np.isinf(calibrator.threshold(state)).all()
    state = calibrator.update(state, feedback([0, 0], origin=6))
    np.testing.assert_array_equal(calibrator.threshold(state), np.zeros((2, 1, 2)))


def test_root_leaf_quantiles_use_the_inverse_cdf_not_interpolation_or_n_plus_one():
    calibrator = method(window=5)
    state = seeded(calibrator, [-99, -10, 0, 1, 2])
    # Training labels are [-10,0,1,2]. Candidate intervals at level .5:
    # beta=0 -> [-10,0]; beta=.25 -> [-10,1]; beta=.5 -> [0,2].
    np.testing.assert_array_equal(calibrator.threshold(state), [[[0, 2]]])
    assert calibrator.threshold(state).dtype == np.float32


def test_beta_width_ties_choose_the_first_beta():
    calibrator = method()
    state = seeded(calibrator, [-99, 0, 1, 2, 3, 4])
    # [0,2], [1,3], [2,4] all have width 2.
    np.testing.assert_array_equal(calibrator.threshold(state), [[[0, 2]]])


@pytest.mark.parametrize(("ahead", "expected"), [(1, -10), (2, 10), (3, -10)])
def test_conditional_forests_learn_a_different_direct_target_at_each_horizon(ahead, expected):
    calibrator = method(
        level=0.9,
        window=40,
        ahead=ahead,
        n_lags=1,
        min_train_samples=2,
        min_samples_leaf=1,
        max_depth=1,
    )
    state = seeded(calibrator, np.tile([-10, 10], 20))
    # Alternating errors: one step flips sign, two steps preserves it.
    # A marginal residual quantile cannot produce these context-dependent intervals.
    np.testing.assert_array_equal(calibrator.threshold(state), [[[expected, expected]]])


def test_training_contexts_respect_calendar_origins_including_gaps():
    calibrator = method(n_lags=2, ahead=3)
    values = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    origins = np.array([0, 1, 4, 7, 8, 11])
    features, targets = calibrator._training_pairs(values, origins, 3)
    np.testing.assert_array_equal(features, [[10, 20], [20, 30], [20, 30], [40, 50]])
    np.testing.assert_array_equal(targets, [30, 40, 50, 60])
    # At issuing origin 8, error at origin 7 is not known until period 10.
    assert 40 not in features[2]


def test_quantile_forest_matches_independent_leaf_mass_calculation():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    targets = np.array([-10, 4, -2, 9, 1, 20], dtype=np.float32)
    context = np.array([5, 6], dtype=np.float32)
    calibrator = method(
        level=0.6, beta_grid_size=5, n_estimators=7, min_samples_leaf=1, max_depth=2, bootstrap=True
    )
    forest = RandomForestRegressor(
        n_estimators=7, min_samples_leaf=1, max_depth=2, bootstrap=True, random_state=0, n_jobs=1
    ).fit(features, targets)
    masses = [0.0] * len(targets)
    for tree in forest.estimators_:
        leaf = tree.apply(context[None, :])[0]
        members = [i for i, label in enumerate(tree.apply(features)) if label == leaf]
        for i in members:
            masses[i] += 1 / (len(members) * len(forest.estimators_))
    support = sorted(
        (float(value), mass) for value, mass in zip(targets, masses, strict=True) if mass > 0
    )

    def quantile(q):
        cumulative = 0.0
        for value, mass in support:
            cumulative += mass
            if cumulative >= q * sum(masses):
                return value
        return support[-1][0]

    candidates = [(quantile(beta), quantile(0.6 + beta)) for beta in [0, 0.1, 0.2, 0.3, 0.4]]
    expected = min(candidates, key=lambda pair: pair[1] - pair[0])
    np.testing.assert_array_equal(calibrator._interval(features, targets, context, 0.6), expected)


def test_seed_retains_last_finite_errors_and_their_original_dates_per_cell():
    calibrator = method(window=3)
    values = np.array(
        [
            [[1, 10], [100, np.nan]],
            [[np.nan, 11], [101, np.nan]],
            [[2, 12], [102, np.nan]],
            [[3, 13], [np.nan, np.nan]],
            [[4, np.nan], [103, np.nan]],
        ]
    )
    state = seeded(calibrator, values, [0, 2, 5, 6, 9])
    np.testing.assert_array_equal(state["count"], [[3, 3], [3, 0]])
    for node, column, expected_values, expected_dates in [
        (0, 0, [2, 3, 4], [5, 6, 9]),
        (0, 1, [11, 12, 13], [2, 5, 6]),
        (1, 0, [101, 102, 103], [2, 5, 9]),
    ]:
        slots = (state["next"][node, column] + np.arange(3)) % 3
        np.testing.assert_array_equal(state["residuals"][node, column, slots], expected_values)
        np.testing.assert_array_equal(state["origins"][node, column, slots], expected_dates)
    assert np.isinf(calibrator.threshold(state)[1, 1]).all()
    assert all(v.shape[0] == 2 and v.dtype != object for v in flatten(state).values())


def test_feedback_uses_archived_centers_and_skips_nonfinite_constructed_errors():
    calibrator = method(window=3)
    state = seeded(calibrator, np.zeros((3, 2, 1)))
    state = calibrator.update(state, feedback([-3, 12], points=[[2, 10]], origin=3))
    np.testing.assert_array_equal(state["residuals"][:, 0, 0], [-5, 2])
    np.testing.assert_array_equal(state["origins"][:, 0, 0], [3, 3])
    before = {key: value.copy() for key, value in state.items()}
    calibrator.update(state, feedback([np.nan, np.inf], origin=4))
    calibrator.update(state, feedback([3e38, 3e38], points=[[-3e38, -3e38]], origin=5))
    for key in ["residuals", "origins", "count", "next"]:
        np.testing.assert_array_equal(state[key], before[key])
    # Even a missing feedback row is consumed, but it occupies no residual slot.
    np.testing.assert_array_equal(state["last_origin"], [[5], [5]])


def test_threshold_is_pure_and_deterministic_without_fitted_instance_state():
    calibrator = method(min_samples_leaf=1, bootstrap=True)
    state = seeded(calibrator, np.arange(20) % 3)
    before = {key: value.copy() for key, value in state.items()}
    configuration = dict(vars(calibrator))
    first = calibrator.threshold(state)
    for _ in range(3):
        np.testing.assert_array_equal(calibrator.threshold(state), first)
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])
    assert vars(calibrator).keys() == configuration.keys()
    assert all(not isinstance(value, RandomForestRegressor) for value in vars(calibrator).values())


def test_nodes_and_horizons_are_independent_including_levels_and_ring_wrap():
    levels = np.array([[0.5, 0.75], [0.75, 0.5]])
    rng = np.random.default_rng(4)
    values = rng.normal(size=(30, 2, 2)).astype(np.float32)
    values[2::3, 0, 1] = np.nan
    calibrator = method(level=levels, ahead=np.array([1, 3]), min_samples_leaf=1, bootstrap=True)
    state = seeded(calibrator, values)
    result = calibrator.threshold(state)
    for node in range(2):
        for column, ahead in enumerate([1, 3]):
            single = method(
                level=float(levels[node, column]), ahead=ahead, min_samples_leaf=1, bootstrap=True
            )
            alone = seeded(single, values[:, node : node + 1, column : column + 1])
            np.testing.assert_array_equal(result[node, column], single.threshold(alone)[0, 0])


def reference_interval(history, *, ahead, n_lags, level, minimum):
    labels = []
    for origin, value in history:
        available = [v for previous, v in history if previous + ahead <= origin]
        if len(available) >= n_lags:
            labels.append(value)
    if len(labels) < minimum:
        return [-np.inf, np.inf]
    labels.sort()

    def quantile(q):
        return labels[max(0, min(len(labels) - 1, math.ceil(len(labels) * q) - 1))]

    candidates = [
        [quantile(beta), quantile(level + beta)] for beta in [0, (1 - level) / 2, 1 - level]
    ]
    return min(candidates, key=lambda pair: pair[1] - pair[0])


@pytest.mark.parametrize("target", [Step(), LeadTime(3)])
def test_replay_matches_independent_delayed_origin_reference(target):
    origins = np.array([10, 11, 14, 15, 19, 22, 25, 26, 30])
    rng = np.random.default_rng(2)
    actuals = rng.normal(10, 2, size=(2, 35)).astype(np.float32)
    points = rng.normal(9, 3, size=(len(origins), 2, 3)).astype(np.float32)
    covered = [(1,), (2,), (3,)] if isinstance(target, Step) else [(1, 2, 3)]
    ahead = np.array([steps[-1] for steps in covered])
    levels = np.array([[0.5] * len(covered), [0.75] * len(covered)])
    calibrator = method(level=levels, ahead=ahead, n_lags=2, min_train_samples=2, window=7)
    seed = rng.normal(size=(6, 2, len(covered))).astype(np.float32)
    seed[2, 0, 0] = np.nan
    state = initial_state(target, calibrator, 2, 3)
    state["calibrator"] = calibrator.seed(state["calibrator"], seed, origins=np.arange(6))
    result = replay(
        Forecasts(origins, points),
        actuals,
        target=target,
        score=Asymmetric(),
        calibrator=calibrator,
        state=state,
    )
    histories = [
        [
            [(o, float(v)) for o, v in enumerate(seed[:, node, column]) if np.isfinite(v)]
            for column in range(len(covered))
        ]
        for node in range(2)
    ]
    released = set()
    for index, origin in enumerate(origins):
        for previous in range(index):
            for column, steps in enumerate(covered):
                if (previous, column) in released or origins[previous] + steps[-1] > origin:
                    continue
                released.add((previous, column))
                # Sums in float32 match the ledger's archived target and point columns.
                dates = [int(origins[previous]) + h for h in steps]
                target_values = actuals[:, dates].sum(axis=1)
                centers = points[previous][:, np.array(steps) - 1].sum(axis=1)
                errors = target_values - centers
                for node in range(2):
                    histories[node][column].append((int(origins[previous]), float(errors[node])))
                    histories[node][column] = histories[node][column][-7:]
        for node in range(2):
            for column, steps in enumerate(covered):
                expected = reference_interval(
                    histories[node][column],
                    ahead=steps[-1],
                    n_lags=2,
                    level=float(levels[node, column]),
                    minimum=2,
                )
                np.testing.assert_array_equal(result.threshold[index, node, column], expected)
    np.testing.assert_array_equal(
        result.covered,
        np.where(
            np.isfinite(result.score),
            ((result.lower <= result.target) & (result.target <= result.upper)).astype(float),
            np.nan,
        ),
    )


@pytest.mark.parametrize(("target", "ahead"), [(Step(), np.arange(1, 4)), (LeadTime(3), 3)])
def test_multihorizon_replay_cannot_learn_from_future_observations_or_points(target, ahead):
    rng = np.random.default_rng(17)
    origins = np.array([10, 11, 13, 16, 20, 21, 25, 28, 30, 35, 38, 40])
    actuals = rng.normal(size=(2, 45)).astype(np.float32)
    points = rng.normal(size=(len(origins), 2, 3)).astype(np.float32)
    calibrator = method(ahead=ahead, min_samples_leaf=1, bootstrap=True)
    setup: dict[str, Any] = dict(target=target, score=Asymmetric(), calibrator=calibrator)
    result = replay(Forecasts(origins, points), actuals, **setup)
    changed = actuals.copy()
    changed[:, 25:] += 1000
    later_points = points.copy()
    later_points[origins >= 25] -= 500
    other = replay(Forecasts(origins, later_points), changed, **setup)
    np.testing.assert_array_equal(result.threshold[origins < 25], other.threshold[origins < 25])
    assert np.isfinite(result.threshold[-1]).all()


def test_numpy_state_resumes_exactly_with_a_new_calibrator_and_pending_horizons():
    rng = np.random.default_rng(42)
    actuals = rng.normal(size=(2, 60)).astype(np.float32)
    origins = np.array([10, 11, 14, 17, 18, 21, 24, 25, 29, 31, 34, 38, 40, 44, 47, 50])
    points = rng.normal(size=(len(origins), 2, 3)).astype(np.float32)

    def run(dates, forecasts, state=None):
        return replay(
            Forecasts(dates, forecasts),
            actuals,
            target=Step(),
            score=Asymmetric(),
            calibrator=method(ahead=np.arange(1, 4), min_samples_leaf=1, bootstrap=True, window=5),
            state=state,
        )

    whole = run(origins, points)
    left = run(origins[:8], points[:8])
    buffer = io.BytesIO()
    flat: dict[str, Any] = flatten(left.state)
    np.savez(buffer, **flat)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        state = unflatten(dict(saved))
    right = run(origins[8:], points[8:], state)
    np.testing.assert_array_equal(
        np.concatenate([left.threshold, right.threshold]), whole.threshold
    )
    for name, value in flatten(whole.state).items():
        np.testing.assert_array_equal(flatten(right.state)[name], value)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"level": 0}, "level"),
        ({"level": np.ones(2)}, "shape"),
        ({"window": True}, "window"),
        ({"window": 1}, "window"),
        ({"n_lags": 0}, "n_lags"),
        ({"n_lags": 1.5}, "n_lags"),
        ({"min_train_samples": 0}, "min_train_samples"),
        ({"min_train_samples": 12}, "window"),
        ({"beta_grid_size": 1}, "beta_grid_size"),
        ({"n_estimators": False}, "n_estimators"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": True}, "max_depth"),
        ({"max_depth": 1.5}, "max_depth"),
        ({"bootstrap": 1}, "bootstrap"),
        ({"random_state": None}, "random_state"),
        ({"random_state": True}, "random_state"),
        ({"random_state": -1}, "random_state"),
        ({"random_state": 2**32}, "random_state"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_jobs": False}, "n_jobs"),
        ({"n_jobs": 1.5}, "n_jobs"),
        ({"ahead": 0}, "ahead"),
        ({"ahead": True}, "ahead"),
        ({"ahead": 1.5}, "ahead"),
        ({"ahead": []}, "ahead"),
        ({"ahead": [[1]]}, "ahead"),
        ({"ahead": np.array([2**63], dtype=np.uint64)}, "ahead"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        method(**kwargs)


def test_configuration_broadcasts_are_checked_on_initialization():
    with pytest.raises(ValueError):
        method(ahead=np.array([1, 2])).initial_state(2, 3)
    with pytest.raises(ValueError):
        method(level=np.full((2, 2), 0.9)).initial_state(2, 3)


@pytest.mark.parametrize(
    ("values", "origins", "match"),
    [
        (np.ones((3, 2)), np.arange(3), "residuals"),
        (np.ones((3, 2, 1)), np.arange(3), "residuals"),
        (np.full((3, 1, 1), np.inf), np.arange(3), "finite"),
        (np.full((3, 1, 1), 1e100), np.arange(3), "finite"),
        (np.ones((3, 1, 1)), [0, 1], "origins"),
        (np.ones((3, 1, 1)), [0, 1, 1], "origins"),
        (np.ones((3, 1, 1)), [0, 2, 1], "origins"),
        (np.ones((3, 1, 1)), [-1, 0, 1], "origins"),
        (np.ones((3, 1, 1)), [0.0, 1.0, 2.0], "origins"),
        (np.ones((3, 1, 1)), np.array([0, 1, 2**63], dtype=np.uint64), "origins"),
    ],
)
def test_invalid_seed_is_rejected_without_mutation(values, origins, match):
    calibrator = method()
    state = calibrator.initial_state(1, 1)
    before = {key: value.copy() for key, value in state.items()}
    with pytest.raises(ValueError, match=match):
        calibrator.seed(state, values, origins=np.asarray(origins))
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_seed_cannot_be_repeated_even_when_all_errors_were_missing():
    calibrator = method()
    state = seeded(calibrator, [np.nan, np.nan])
    with pytest.raises(ValueError, match="empty"):
        calibrator.seed(state, np.zeros((3, 1, 1)), origins=np.arange(3))


def test_duplicate_and_reordered_feedback_is_rejected():
    calibrator = method()
    state = seeded(calibrator, [1, 2])
    for origin in [0, 1]:
        with pytest.raises(ValueError, match="origins"):
            calibrator.update(state, feedback([0], origin=origin))


def test_scalar_scores_are_rejected_by_both_update_and_online_initialization():
    calibrator = method()
    with pytest.raises(ValueError, match="paired"):
        calibrator.update(calibrator.initial_state(1, 1), feedback([1], origin=0, score=Absolute()))
    with pytest.raises(ValueError, match="parameter shape"):
        step(
            initial_state(Step(), calibrator, 1, 1),
            3,
            np.zeros((1, 1)),
            np.zeros((1, 4)),
            target=Step(),
            score=Absolute(),
            calibrator=calibrator,
        )


def test_importing_spci_does_not_import_the_optional_forest_library():
    script = """
import sys
from calibre import SPCI
from calibre.conformal import SPCI as conformal_spci
from calibre.conformal.calibrators import SPCI as calibrator_spci
assert SPCI is conformal_spci is calibrator_spci
assert 'sklearn' not in sys.modules
method = SPCI(.9, 32, 2, ahead=1)
assert method.threshold(method.initial_state(1, 1)).shape == (1, 1, 2)
assert 'sklearn' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
