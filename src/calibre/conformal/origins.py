"""The calibration driver: one origin at a time, or a replay over many origins.

`Conformal.step` is the production call. It takes the state, the periods observed
since the last call, and the new points, and returns the new state and the issued
intervals. `Conformal.replay` is a backtest: the same steps over a list of origins.
"""

from dataclasses import dataclass

import numpy as np

from calibre.conformal import Calibrator, Feedback, Score, State
from calibre.forecast.origins import Forecasts


@dataclass(frozen=True)
class Issued:
    """What one origin issues, in score columns `[N, C]`."""

    point: np.ndarray
    threshold: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class Calibrated:
    """A replay over origins `[O]`. Arrays are `[O, N, C]`.

    `target` and `score` are NaN when the target is not known at the end of the data.
    `state` continues the run with `Conformal.step`.
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


@dataclass(frozen=True)
class Conformal:
    """A score, a calibrator, and a level, run origin by origin."""

    score: Score
    calibrator: Calibrator
    level: float

    def start(self, n_nodes: int, horizon: int) -> State:
        """Empty state for `n_nodes` nodes and `horizon` steps."""
        cover = self.score.cover(horizon)
        n_columns, reach = len(cover), _reach(cover)
        # An origin waits until its last scored step, `reach` periods later. So at most
        # `reach` origins wait, and slot origin % reach is free again when reused. Only
        # the scored steps are kept. Origin -1 marks an empty slot.
        return {
            "last": np.array(-1, dtype=np.int64),
            "pending": {
                "origin": np.full(reach, -1, dtype=np.int64),
                "point": np.zeros((reach, n_nodes, reach), dtype=np.float32),
                "actual": np.full((reach, n_nodes, reach), np.nan, dtype=np.float32),
                "censored": np.zeros((reach, n_nodes, reach), dtype=bool),
                "issued": np.zeros((reach, n_nodes, n_columns), dtype=np.float32),
                "known": np.zeros((reach, n_columns), dtype=bool),
            },
            "calibrator": self.calibrator.init(n_nodes, n_columns),
        }

    def step(
        self,
        state: State,
        origin: int,
        points: np.ndarray,
        actuals: np.ndarray,
        censored: np.ndarray | None = None,
    ) -> tuple[State, Issued]:
        """Observe the periods up to `origin`, update, and issue for the new points.

        `actuals` is `[N, k]`, the node values of the k periods that end at `origin`
        and follow the previous origin. On the first call, k can be any length.
        `points` is `[N, H]`, the forecast issued at `origin`.
        """
        last = int(state["last"])
        n_new = actuals.shape[1]
        if last >= 0 and (origin <= last or n_new != origin - last):
            raise ValueError(f"origin {origin} after {last} needs {origin - last} new periods")
        last = origin - n_new
        if censored is None:
            censored = np.zeros(actuals.shape, dtype=bool)
        cover = self.score.cover(points.shape[1])
        reach = _reach(cover)
        pending = _observe(state["pending"], actuals, censored, last, origin)
        pending, feedback = _resolve(self.score, cover[:, :reach], pending, origin)
        calibrator_state = state["calibrator"]
        if feedback is not None:
            calibrator_state = self.calibrator.update(calibrator_state, feedback, self.level)
        threshold = self.calibrator.threshold(calibrator_state, self.level)
        point = _columns(points.astype(np.float32), cover)
        lower, upper = self.score.interval(point, threshold)
        pending = _append(pending, origin, points[:, :reach], threshold)
        new_state = {"last": np.array(origin), "pending": pending, "calibrator": calibrator_state}
        return new_state, Issued(point, threshold, lower, upper)

    def replay(
        self,
        forecasts: Forecasts,
        actuals: np.ndarray,
        censored: np.ndarray | None = None,
    ) -> Calibrated:
        """Run `step` at each origin of `forecasts` on node actuals `[N, T]`."""
        origins = forecasts.origins
        n_origins, n_nodes, horizon = forecasts.points.shape
        if censored is None:
            censored = np.zeros(actuals.shape, dtype=bool)
        state = self.start(n_nodes, horizon)
        issued = []
        previous = -1
        for index, origin in enumerate(origins):
            seen = slice(previous + 1, int(origin) + 1)
            state, out = self.step(
                state, int(origin), forecasts.points[index], actuals[:, seen], censored[:, seen]
            )
            issued.append(out)
            previous = int(origin)
        cover = self.score.cover(horizon)
        target, target_censored = _targets(actuals, censored, origins, horizon, cover)
        point = np.stack([out.point for out in issued])
        return Calibrated(
            origins=origins,
            point=point,
            threshold=np.stack([out.threshold for out in issued]),
            lower=np.stack([out.lower for out in issued]),
            upper=np.stack([out.upper for out in issued]),
            target=target,
            score=self.score.score(target, point),
            censored=target_censored,
            state=state,
        )


def _reach(cover: np.ndarray) -> int:
    """Number of steps up to the last step that any score column covers."""
    return int(np.flatnonzero(cover.any(axis=0))[-1]) + 1


def _observe(
    pending: State, actuals: np.ndarray, censored: np.ndarray, last: int, origin: int
) -> State:
    """Write the newly observed periods into the targets of waiting origins, in place."""
    reach = pending["point"].shape[2]
    targets = pending["origin"][:, None] + np.arange(1, reach + 1)
    waiting = (pending["origin"] >= 0)[:, None]
    slots, steps = np.nonzero(waiting & (targets > last) & (targets <= origin))
    columns = targets[slots, steps] - last - 1
    pending["actual"][slots, :, steps] = actuals[:, columns].T
    pending["censored"][slots, :, steps] = censored[:, columns].T
    return pending


def _resolve(
    score: Score, cover: np.ndarray, pending: State, origin: int
) -> tuple[State, Feedback | None]:
    """Score the columns whose last step is now known, and free finished slots."""
    last_step = cover.shape[1] - np.argmax(cover[:, ::-1], axis=1)
    waiting = pending["origin"] >= 0
    known = waiting[:, None] & (pending["origin"][:, None] + last_step <= origin)
    new = known & ~pending["known"]
    slots, columns = np.nonzero(new)
    order = np.lexsort((columns, pending["origin"][slots]))
    slots, columns = slots[order], columns[order]
    feedback = None
    if len(slots):
        n_nodes = pending["point"].shape[1]
        target = np.empty((len(slots), n_nodes), dtype=np.float32)
        point = np.empty((len(slots), n_nodes), dtype=np.float32)
        flags = np.empty((len(slots), n_nodes), dtype=bool)
        for column in np.unique(columns):
            rows = np.flatnonzero(columns == column)
            steps = np.flatnonzero(cover[column])
            target[rows] = pending["actual"][slots[rows]][:, :, steps].sum(axis=-1)
            point[rows] = pending["point"][slots[rows]][:, :, steps].sum(axis=-1)
            flags[rows] = pending["censored"][slots[rows]][:, :, steps].any(axis=-1)
        feedback = Feedback(
            origin=pending["origin"][slots],
            column=columns,
            scores=score.score(target, point).astype(np.float32),
            issued=pending["issued"][slots, :, columns],
            censored=flags,
        )
    pending["known"] |= new
    pending["origin"][waiting & pending["known"].all(axis=1)] = -1
    return pending, feedback


def _append(pending: State, origin: int, points: np.ndarray, threshold: np.ndarray) -> State:
    """Put the new origin in its slot, in place."""
    slot = origin % len(pending["origin"])
    if pending["origin"][slot] >= 0:
        raise ValueError(
            f"origin {origin} reuses the slot of waiting origin {pending['origin'][slot]}"
        )
    pending["origin"][slot] = origin
    pending["point"][slot] = points
    pending["actual"][slot] = np.nan
    pending["censored"][slot] = False
    pending["issued"][slot] = threshold
    pending["known"][slot] = False
    return pending


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
    return _columns(values.astype(np.float32), cover), _columns(flags, cover)


def _columns(values: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """Steps `[..., H]` to score columns `[..., C]`: a sum, or any for flags.

    A missing covered step makes the column missing. Uncovered steps are ignored.
    """
    if cover.shape[0] == cover.shape[1] and (cover == np.eye(len(cover), dtype=bool)).all():
        return values
    reduce = np.any if values.dtype == bool else np.sum
    out = [reduce(values[..., steps], axis=-1) for steps in cover]
    return np.stack(out, axis=-1)
