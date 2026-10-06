"""Calibrators: from the scores known so far to a threshold per node and column."""

import numpy as np

from calibre.conformal import Feedback, State
from calibre.conformal.quantile import retained_quantile


class SplitQuantile:
    """Split conformal quantile of the known scores, per column.

    `window` keeps the last resolved origins per pool, `groups` pools nodes (one
    integer per node). `capacity` bounds the stored origins, so the saved state has a
    fixed size. With a window, keep `capacity` above the window plus the number of
    origins whose scores are not known yet. None keeps every origin.
    """

    def __init__(
        self,
        window: int | None = None,
        groups: np.ndarray | None = None,
        capacity: int | None = None,
    ) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        if capacity is not None and window is not None and capacity < window:
            raise ValueError(f"capacity {capacity} is smaller than window {window}")
        self.window = window
        self.groups = None if groups is None else np.asarray(groups)
        self.capacity = capacity

    def init(self, n_nodes: int, n_columns: int) -> State:
        """Stored origins in order. Rows after `size` are spare, with origin -1."""
        return {
            "size": np.array(0, dtype=np.int64),
            "origin": np.empty(0, dtype=np.int64),
            "known": np.empty((0, n_columns), dtype=bool),
            "scores": np.empty((n_columns, 0, n_nodes), dtype=np.float32),
        }

    def update(self, state: State, feedback: Feedback, level: float) -> State:
        for origin in np.unique(feedback.origin):
            state, _ = _row(state, int(origin), self.capacity)
        size = int(state["size"])
        rows = np.searchsorted(state["origin"][:size], feedback.origin)
        state["scores"][feedback.column, rows] = feedback.scores
        state["known"][rows, feedback.column] = True
        return state

    def threshold(self, state: State, level: float | np.ndarray) -> np.ndarray:
        """`level` is a scalar or one value per node and column."""
        size = int(state["size"])
        n_columns, _, n_nodes = state["scores"].shape
        levels = np.broadcast_to(np.asarray(level, dtype=np.float64), (n_nodes, n_columns))
        out = np.empty((n_nodes, n_columns), dtype=np.float32)
        for column in range(n_columns):
            column_level = levels[:, column]
            same = (column_level == column_level[0]).all()
            # Known rows come first: origins of one column become known in origin order.
            known = state["known"][:size, column]
            stop = size - int(np.argmax(known[::-1])) if known.any() else 0
            out[:, column] = retained_quantile(
                state["scores"][column, :stop],
                column_level[0] if same else column_level,
                self.window,
                self.groups,
            )
        return out


class ACI:
    """Adaptive conformal inference (Gibbs and Candès 2021), per node and column.

    After each known score, the working level moves by `gamma * (miss - (1 - level))`,
    where a miss is a score above the threshold that was issued for it. The base
    calibrator gives the threshold at the working level. A working level at or above
    one gives an infinite threshold, at or below zero an empty interval.
    """

    def __init__(self, base, gamma: float = 0.005) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.base = base
        self.gamma = gamma

    def init(self, n_nodes: int, n_columns: int) -> State:
        return {
            "base": self.base.init(n_nodes, n_columns),
            "level": np.full((n_nodes, n_columns), np.nan),
        }

    def update(self, state: State, feedback: Feedback, level: float) -> State:
        working = np.where(np.isnan(state["level"]), level, state["level"])
        misses, known = _misses(feedback, working.shape[1])
        working = working + self.gamma * (misses - (1 - level) * known)
        return {"base": self.base.update(state["base"], feedback, level), "level": working}

    def threshold(self, state: State, level: float) -> np.ndarray:
        working = np.where(np.isnan(state["level"]), level, state["level"])
        inside = np.clip(working, 1e-6, 1 - 1e-6)
        out = self.base.threshold(state["base"], inside)
        out = np.where(working >= 1, np.inf, out)
        return np.where(working <= 0, -np.inf, out).astype(np.float32)


class QuantileTracker:
    """Online quantile tracking, the P term of conformal PID (Angelopoulos et al. 2023).

    The threshold moves by `lr * (miss - (1 - level))` after each known score. It needs
    no stored scores, so its state is one value per node and column. The integral and
    scorecaster terms of conformal PID are not here.
    """

    def __init__(self, lr: float, start: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = lr
        self.start = start

    def init(self, n_nodes: int, n_columns: int) -> State:
        return {"threshold": np.full((n_nodes, n_columns), self.start, dtype=np.float64)}

    def update(self, state: State, feedback: Feedback, level: float) -> State:
        misses, known = _misses(feedback, state["threshold"].shape[1])
        return {"threshold": state["threshold"] + self.lr * (misses - (1 - level) * known)}

    def threshold(self, state: State, level: float) -> np.ndarray:
        return state["threshold"].astype(np.float32)


def _misses(feedback: Feedback, n_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts `[N, C]` of misses (score above the issued threshold) and known scores."""
    known = np.isfinite(feedback.scores)
    miss = (feedback.scores > feedback.issued) & known
    columns = np.eye(n_columns)[feedback.column]  # [K, C]
    return miss.T @ columns, known.T @ columns


def _row(state: State, origin: int, capacity: int | None) -> tuple[State, int]:
    """Row of `origin`, adding it in origin order. Grows by doubling up to capacity."""
    size = int(state["size"])
    stored = state["origin"][:size]
    row = int(np.searchsorted(stored, origin))
    if row < size and stored[row] == origin:
        return state, row
    if capacity is not None and size == capacity:
        if row == 0:
            raise ValueError(f"origin {origin} is older than every stored origin")
        # Full: drop the oldest row in place.
        for name, axis in (("origin", 0), ("known", 0), ("scores", 1)):
            values = np.moveaxis(state[name], axis, 0)
            values[: size - 1] = values[1:size].copy()
        size, row = size - 1, row - 1
    if size == len(state["origin"]):
        allocated = max(1, 2 * size) if capacity is None else min(max(1, 2 * size), capacity)
        state = _grow(state, allocated)
    for name, axis in (("origin", 0), ("known", 0), ("scores", 1)):
        values = np.moveaxis(state[name], axis, 0)
        values[row + 1 : size + 1] = values[row:size].copy()
    state["origin"][row] = origin
    state["known"][row] = False
    state["scores"][:, row] = np.nan
    state["size"] = np.array(size + 1, dtype=np.int64)
    return state, row


def _grow(state: State, allocated: int) -> State:
    extra = allocated - len(state["origin"])
    n_columns, _, n_nodes = state["scores"].shape
    return {
        "size": state["size"],
        "origin": np.concatenate([state["origin"], np.full(extra, -1, dtype=np.int64)]),
        "known": np.concatenate([state["known"], np.zeros((extra, n_columns), dtype=bool)]),
        "scores": np.concatenate(
            [state["scores"], np.full((n_columns, extra, n_nodes), np.nan, dtype=np.float32)],
            axis=1,
        ),
    }
