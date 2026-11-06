"""One origin of online calibration: learn from the targets now known, then issue.

`step` is the production call. It takes the state, the periods observed since the
last call, and the new points, and returns the new state and the issued bounds. The
caller owns the origins and stores the state between calls.
"""

from dataclasses import dataclass

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, State
from calibre.conformal.scores import Score
from calibre.conformal.targets import Target, columns
from calibre.online import ledger


@dataclass(frozen=True)
class Issue:
    """Forecasts [N, C, *input_shape] and bounds [N, C] with issued parameters."""

    point: np.ndarray
    threshold: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def initial_state(
    target: Target,
    calibrator: Calibrator,
    n_nodes: int,
    horizon: int,
    *,
    score: Score,
) -> State:
    """Empty ledger and calibrator state shaped by the score and target."""
    cover = target.cover(horizon)
    _check_cover(cover, score.input_shape())
    calibration = calibrator.initial_state(n_nodes, len(cover))
    if calibrator.threshold(calibration).shape != (n_nodes, len(cover)) + score.parameter_shape():
        raise ValueError("score parameter shape does not match the calibrator")
    return {
        "ledger": ledger.initial_state(
            n_nodes, cover, score.input_shape(), score.parameter_shape()
        ),
        "calibrator": calibration,
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
    """Observe [N, k] actuals, then issue from [N, H, *score.input_shape()] forecasts.

    k periods end at origin and follow the previous origin. On the first call, k
    can be any length. Native quantiles are validated and ordered here, on a copy,
    before any state is changed.
    """
    points = np.asarray(points, dtype=np.float32)
    shape = score.input_shape()
    if (
        points.ndim != 2 + len(shape)
        or points.shape[2:] != shape
        or points.shape[0] != state["ledger"]["point"].shape[1]
    ):
        raise ValueError("forecasts must have shape [N, H] + score.input_shape()")
    if state["ledger"]["point"].shape[3:] != shape:
        raise ValueError("score input shape does not match the saved state")
    if state["ledger"]["issued"].shape[3:] != score.parameter_shape():
        raise ValueError("score parameter shape does not match the saved state")
    if shape:
        if not np.isfinite(points).all():
            raise ValueError("native quantiles must be finite")
        points = np.sort(points, axis=-1)
    if censored is None:
        censored = np.zeros(actuals.shape, dtype=bool)
    cover = target.cover(points.shape[1])
    _check_cover(cover, shape)
    if (
        len(cover) != state["ledger"]["issued"].shape[2]
        or ledger.reach(cover) != state["ledger"]["point"].shape[2]
    ):
        raise ValueError("forecast horizon and target do not match the saved state")
    ledger_state, feedback = ledger.observe(
        state["ledger"], cover, score, origin, actuals, censored
    )
    calibrator_state = state["calibrator"]
    if feedback is not None:
        calibrator_state = calibrator.update(calibrator_state, feedback)
    threshold = calibrator.threshold(calibrator_state)
    point = np.moveaxis(columns(np.moveaxis(points, 1, -1), cover), -1, 1)
    lower, upper = score.bound(point, threshold)
    ledger_state = ledger.issue(ledger_state, origin, points, threshold)
    return {"ledger": ledger_state, "calibrator": calibrator_state}, Issue(
        point, threshold, lower, upper
    )


def _check_cover(cover: np.ndarray, input_shape: tuple[int, ...]) -> None:
    if input_shape and (cover.sum(axis=1) != 1).any():
        raise ValueError("native quantiles require each target column to cover exactly one step")
