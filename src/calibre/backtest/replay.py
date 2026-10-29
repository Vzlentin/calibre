"""Replay: online calibration over the origins of a backtest, kept for assessment."""

from dataclasses import dataclass

import numpy as np

from calibre.backtest.forecasts import Forecasts
from calibre.conformal.calibrators.base import Calibrator, State
from calibre.conformal.scores import Score
from calibre.conformal.targets import Target, columns
from calibre.online.step import initial_state, step


@dataclass(frozen=True)
class Replay:
    """Calibration over origins `[O]`. Arrays are `[O, N, C]`.

    Asymmetric thresholds have an additional trailing lower/upper axis of size two.
    `interval` optionally retains native inputs `[O, N, H, 2]`. Native and paired
    coverage uses issued endpoints, including floating-point boundary hits.

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
    interval: np.ndarray | None = None

    @property
    def covered(self) -> np.ndarray:
        """Coverage against issued scalar thresholds or paired bounds; NaN if unknown."""
        known = np.isfinite(self.score)
        if self.interval is None and self.threshold.ndim == self.score.ndim:
            inside = self.score <= self.threshold
        else:
            inside = (self.lower <= self.target) & (self.target <= self.upper)
        return np.where(known, inside.astype(np.float32), np.nan)


def replay(
    forecasts: Forecasts,
    actuals: np.ndarray,
    *,
    target: Target,
    score: Score,
    calibrator: Calibrator,
    censored: np.ndarray | None = None,
    state: State | None = None,
) -> Replay:
    """Replay node actuals `[N, T]`, optionally from a seeded or saved online state."""
    origins = forecasts.origins
    n_origins, n_nodes, horizon = forecasts.points.shape
    if censored is None:
        censored = np.zeros(actuals.shape, dtype=bool)
    native = forecasts.interval if score.requires_interval() else None
    if score.requires_interval() and native is None:
        raise ValueError("this score requires forecasts with native intervals")
    if state is None:
        state = initial_state(target, calibrator, n_nodes, horizon, score=score)
    issued = []
    previous = int(state["ledger"]["last"])
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
            censored=censored[:, seen],
            interval=None if native is None else native[index],
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
        score=score.score(targets, point, interval=native),
        censored=target_censored,
        state=state,
        interval=native,
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
