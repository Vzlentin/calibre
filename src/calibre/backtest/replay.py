"""Replay: online calibration over the origins of a backtest, kept for assessment."""

from dataclasses import dataclass

import numpy as np

from calibre.backtest.forecasts import Forecasts
from calibre.conformal.calibrators.base import Calibrator, State
from calibre.conformal.scores import Score
from calibre.conformal.targets import Target, columns
from calibre.online.step import start, step


@dataclass(frozen=True)
class Replay:
    """Calibration over origins `[O]`. Arrays are `[O, N, C]`.

    `target` and `score` are NaN when the target is not known at the end of the data.
    `state` continues the run with `calibre.online.step`.
    """

    origins: np.ndarray
    point: np.ndarray
    threshold: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    target: np.ndarray
    score: np.ndarray
    censored: np.ndarray
    state: State

    @property
    def covered(self) -> np.ndarray:
        """1.0 when the score is within its threshold, NaN when not known."""
        known = np.isfinite(self.score)
        return np.where(known, (self.score <= self.threshold).astype(np.float32), np.nan)


def replay(
    forecasts: Forecasts,
    actuals: np.ndarray,
    *,
    target: Target,
    score: Score,
    calibrator: Calibrator,
    level: float,
    censored: np.ndarray | None = None,
) -> Replay:
    """Run `calibre.online.step` at each origin of `forecasts` on node actuals `[N, T]`."""
    origins = forecasts.origins
    n_origins, n_nodes, horizon = forecasts.points.shape
    if censored is None:
        censored = np.zeros(actuals.shape, dtype=bool)
    state = start(target, calibrator, n_nodes, horizon)
    issued = []
    previous = -1
    for index, origin in enumerate(origins):
        seen = slice(previous + 1, int(origin) + 1)
        state, out = step(
            state,
            int(origin),
            forecasts.points[index],
            actuals[:, seen],
            target=target,
            score=score,
            calibrator=calibrator,
            level=level,
            censored=censored[:, seen],
        )
        issued.append(out)
        previous = int(origin)
    cover = target.cover(horizon)
    targets, target_censored = _targets(actuals, censored, origins, horizon, cover)
    point = np.stack([out.point for out in issued])
    return Replay(
        origins=origins,
        point=point,
        threshold=np.stack([out.threshold for out in issued]),
        lower=np.stack([out.lower for out in issued]),
        upper=np.stack([out.upper for out in issued]),
        target=targets,
        score=score.score(targets, point),
        censored=target_censored,
        state=state,
    )


def _targets(
    actuals: np.ndarray, censored: np.ndarray, origins: np.ndarray, horizon: int, cover: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Targets and censoring per column `[O, N, C]`, NaN after the last period."""
    n_periods = actuals.shape[1]
    steps = origins[:, None] + np.arange(1, horizon + 1)
    known = steps < n_periods
    clipped = np.minimum(steps, n_periods - 1)
    values = np.where(known[:, None, :], actuals[:, clipped].transpose(1, 0, 2), np.nan)
    flags = censored[:, clipped].transpose(1, 0, 2) & known[:, None, :]
    return columns(values.astype(np.float32), cover), columns(flags, cover)
