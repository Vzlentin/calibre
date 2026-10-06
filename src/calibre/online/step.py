"""One origin of online calibration: learn from the targets now known, then issue.

`step` is the production call. It takes the state, the periods observed since the
last call, and the new points, and returns the new state and the issued bounds. The
caller owns the origins and stores the state between calls.
"""

from dataclasses import dataclass

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, State
from calibre.conformal.scores import Score
from calibre.conformal.targets import Target, columns
from calibre.online import ledger


@dataclass(frozen=True)
class Issue:
    """What one origin issues, per node and target column `[N, C]`."""

    point: np.ndarray
    threshold: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def start(target: Target, calibrator: Calibrator, n_nodes: int, horizon: int) -> State:
    """Empty state for `n_nodes` nodes and `horizon` steps."""
    cover = target.cover(horizon)
    return {
        "ledger": ledger.start(n_nodes, cover),
        "calibrator": calibrator.init(n_nodes, len(cover)),
    }


def step(
    state: State,
    origin: int,
    points: np.ndarray,
    actuals: np.ndarray,
    *,
    target: Target,
    score: Score,
    calibrator: Calibrator,
    censored: np.ndarray | None = None,
) -> tuple[State, Issue]:
    """Observe the periods up to `origin`, update the calibrator, and issue for `points`.

    `actuals` is `[N, k]`, the node values of the k periods that end at `origin` and
    follow the previous origin. On the first call, k can be any length. `points` is
    `[N, H]`, the forecast issued at `origin`. The calibrator sees a score only when
    the last step of its column is known, and before `origin` issues.
    """
    if censored is None:
        censored = np.zeros(actuals.shape, dtype=bool)
    cover = target.cover(points.shape[1])
    ledger_state, matured = ledger.observe(state["ledger"], cover, origin, actuals, censored)
    calibrator_state = state["calibrator"]
    if matured is not None:
        feedback = Feedback(
            origin=matured.origin,
            column=matured.column,
            scores=score.score(matured.target, matured.point).astype(np.float32),
            issued=matured.issued,
            censored=matured.censored,
        )
        calibrator_state = calibrator.update(calibrator_state, feedback)
    threshold = calibrator.threshold(calibrator_state)
    point = columns(points.astype(np.float32), cover)
    lower, upper = score.bound(point, threshold)
    ledger_state = ledger.issue(ledger_state, origin, points, threshold)
    return {"ledger": ledger_state, "calibrator": calibrator_state}, Issue(
        point, threshold, lower, upper
    )
