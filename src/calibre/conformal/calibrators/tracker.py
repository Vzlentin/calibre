"""Online quantile tracking: a threshold that moves after each known score."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, Level, State, check_level


class QuantileTracker(Calibrator):
    """Online quantile tracking, the P term of conformal PID (Angelopoulos et al. 2023).

    The threshold moves by `lr * (miss - (1 - level))` after each known score, so its
    misses converge to `1 - level`. It needs
    no stored scores, so its state is one value per node and column. The integral and
    scorecaster terms of conformal PID are not here.
    """

    def __init__(self, level: Level, lr: float, start: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.level = check_level(level)
        self.lr = lr
        self.start = start

    def init(self, n_nodes: int, n_columns: int) -> State:
        return {"threshold": np.full((n_nodes, n_columns), self.start, dtype=np.float64)}

    def update(self, state: State, feedback: Feedback) -> State:
        misses, known = feedback.misses(state["threshold"].shape[1])
        step = misses - (1 - self.level) * known
        return {"threshold": state["threshold"] + self.lr * step}

    def threshold(self, state: State) -> np.ndarray:
        return state["threshold"].astype(np.float32)
