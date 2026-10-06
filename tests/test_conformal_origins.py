import io

import numpy as np
import pytest

from calibre.conformal import Feedback, State
from calibre.conformal.calibrators import ACI, SplitQuantile
from calibre.conformal.origins import Conformal
from calibre.conformal.scores import AbsoluteResidual, SignedResidual, WindowSum
from calibre.conformal.state import flatten, unflatten
from calibre.forecast.origins import Forecasts


class CensoredCount:
    """Threshold = number of censored known cells so far, per node and column."""

    def init(self, n_nodes: int, n_columns: int) -> State:
        return {"count": np.zeros((n_nodes, n_columns))}

    def update(self, state: State, feedback: Feedback, level: float) -> State:
        count = state["count"].copy()
        for row, column in enumerate(feedback.column):
            count[:, column] += feedback.censored[row]
        return {"count": count}

    def threshold(self, state: State, level: float) -> np.ndarray:
        return state["count"].astype(np.float32)


def ramp(n_periods: int, n_nodes: int = 1) -> np.ndarray:
    return np.tile(np.arange(n_periods, dtype=np.float32), (n_nodes, 1))


def zero_forecasts(origins: np.ndarray, n_nodes: int, horizon: int) -> Forecasts:
    return Forecasts(origins, np.zeros((len(origins), n_nodes, horizon), dtype=np.float32))


def test_gapped_origins_see_only_targets_known_at_the_origin():
    origins = np.array([100, 103, 104, 110])
    run = Conformal(SignedResidual(), SplitQuantile(), 0.5).replay(
        zero_forecasts(origins, 1, 8), ramp(200)
    )
    scores = run.state["calibrator"]["scores"]  # [C, R, N]
    known = np.isfinite(scores[:, :, 0]).sum(axis=1)
    # At origin 110, step 1 targets 101, 104, 105: all three earlier origins are known.
    # Step 7 targets 107, 110, 111: the first two. Step 8 targets 108, 111, 112: one.
    assert known.tolist() == [3, 3, 3, 3, 3, 3, 2, 1]


def test_a_later_actual_never_changes_an_issued_threshold():
    origins = np.arange(20, 60)
    actuals = np.random.default_rng(0).normal(size=(3, 80)).astype(np.float32)
    conformal = Conformal(AbsoluteResidual(), SplitQuantile(window=10), 0.8)
    before = conformal.replay(zero_forecasts(origins, 3, 4), actuals)
    changed = actuals.copy()
    changed[:, 45] = 1000
    after = conformal.replay(zero_forecasts(origins, 3, 4), changed)
    # Period 45 is first known at origin 45, index 25, and that origin already uses it.
    np.testing.assert_array_equal(after.threshold[:25], before.threshold[:25])
    assert not np.array_equal(after.threshold[25], before.threshold[25])


def test_a_window_score_is_known_only_when_its_last_step_is():
    origins = np.arange(10, 20)
    run = Conformal(WindowSum(3), SplitQuantile(), 0.5).replay(
        zero_forecasts(origins, 1, 5), ramp(40)
    )
    # Origin 10 + 3 steps is known at origin 13: 7 of 10 origins are scored by the end.
    assert np.isfinite(run.state["calibrator"]["scores"]).sum() == 7
    # The score is the sum of the three targets: 11 + 12 + 13 for origin 10.
    assert run.score[0, 0, 0] == 36
    assert np.isinf(run.threshold[:3]).all()


def test_feedback_marks_a_window_censored_when_any_step_is():
    origins = np.arange(10, 16)
    censored = np.zeros((1, 30), dtype=bool)
    censored[0, 12] = True
    run = Conformal(WindowSum(2), CensoredCount(), 0.5).replay(
        zero_forecasts(origins, 1, 2), ramp(30), censored
    )
    # Period 12 is in the windows of origins 10 and 11. They are known at origins 12
    # and 13, before those origins issue.
    assert run.threshold[:, 0, 0].tolist() == [0, 0, 1, 2, 2, 2]
    assert run.censored[:, 0, 0].tolist() == [True, True, False, False, False, False]


def test_a_saved_and_reloaded_state_continues_exactly():
    rng = np.random.default_rng(1)
    actuals = rng.normal(size=(4, 120)).astype(np.float32)
    points = rng.normal(size=(60, 4, 3)).astype(np.float32)
    origins = np.arange(40, 100)
    conformal = Conformal(AbsoluteResidual(), ACI(SplitQuantile(window=20), gamma=0.05), 0.9)
    whole = conformal.replay(Forecasts(origins, points), actuals)

    state = conformal.start(n_nodes=4, horizon=3)
    previous, issued = -1, []
    for index, origin in enumerate(origins):
        if index == 30:
            buffer = io.BytesIO()
            np.savez(buffer, **flatten(state))
            buffer.seek(0)
            state = unflatten(dict(np.load(buffer)))
        seen = slice(previous + 1, origin + 1)
        state, out = conformal.step(state, int(origin), points[index], actuals[:, seen])
        issued.append(out.threshold)
        previous = int(origin)
    np.testing.assert_array_equal(np.stack(issued), whole.threshold)


def test_step_needs_every_period_since_the_last_origin():
    conformal = Conformal(SignedResidual(), SplitQuantile(), 0.5)
    state, _ = conformal.step(conformal.start(1, 2), 10, np.zeros((1, 2)), np.zeros((1, 11)))
    with pytest.raises(ValueError, match="needs 3 new periods"):
        conformal.step(state, 13, np.zeros((1, 2)), np.zeros((1, 2)))


def test_covered_matches_the_issued_interval():
    origins = np.arange(30, 70)
    actuals = np.random.default_rng(2).normal(size=(2, 90)).astype(np.float32)
    run = Conformal(AbsoluteResidual(), SplitQuantile(), 0.8).replay(
        zero_forecasts(origins, 2, 3), actuals
    )
    inside = (run.lower <= run.target) & (run.target <= run.upper)
    known = np.isfinite(run.target)
    np.testing.assert_array_equal(run.covered[known], inside[known])
    assert np.isnan(run.covered[~known]).all()
