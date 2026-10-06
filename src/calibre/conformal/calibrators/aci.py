"""Adaptive conformal inference: a working level that reacts to misses."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, QuantileCalibrator, State


class ACI(Calibrator):
    """Adaptive conformal inference (Gibbs and Candès 2021), per node and column.

    After each known score, the working level moves by `gamma * (miss - (1 - level))`,
    where a miss is a score above the threshold that was issued for it. The base
    calibrator gives the threshold at the working level. The target level is the level
    of the base. A working level at or above one gives an infinite threshold, at or
    below zero an empty interval.
    """

    def __init__(self, base: QuantileCalibrator, gamma: float = 0.005) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.base = base
        self.level = base.level
        self.gamma = gamma

    def init(self, n_nodes: int, n_columns: int) -> State:
        return {
            "base": self.base.init(n_nodes, n_columns),
            "level": np.full((n_nodes, n_columns), np.nan),
        }

    def update(self, state: State, feedback: Feedback) -> State:
        working = np.where(np.isnan(state["level"]), self.level, state["level"])
        misses, known = feedback.misses(working.shape[1])
        working = working + self.gamma * (misses - (1 - self.level) * known)
        return {"base": self.base.update(state["base"], feedback), "level": working}

    def threshold(self, state: State) -> np.ndarray:
        working = np.where(np.isnan(state["level"]), self.level, state["level"])
        inside = np.clip(working, 1e-6, 1 - 1e-6)
        out = self.base.threshold_at(state["base"], inside)
        out = np.where(working >= 1, np.inf, out)
        return np.where(working <= 0, -np.inf, out).astype(np.float32)
