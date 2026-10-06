"""Split conformal quantile of the known scores, with retention and pooling."""

import numpy as np

from calibre.conformal.calibrators.base import (
    Feedback,
    Level,
    QuantileCalibrator,
    State,
    check_level,
)
from calibre.conformal.calibrators.ranks import retained_quantile


class SplitQuantile(QuantileCalibrator):
    """Split conformal quantile of the known scores at `level`, per column.

    `window` keeps the last resolved origins per pool, `groups` pools nodes (one
    integer per node). `capacity` bounds the stored origins, so the saved state has a
    fixed size. With a window, keep `capacity` above the window plus the number of
    origins whose scores are not known yet. None keeps every origin.
    """

    def __init__(
        self,
        level: Level,
        window: int | None = None,
        groups: np.ndarray | None = None,
        capacity: int | None = None,
    ) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        if capacity is not None and window is not None and capacity < window:
            raise ValueError(f"capacity {capacity} is smaller than window {window}")
        self.level = check_level(level)
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

    def update(self, state: State, feedback: Feedback) -> State:
        for origin in np.unique(feedback.origin):
            state, _ = _row(state, int(origin), self.capacity)
        size = int(state["size"])
        rows = np.searchsorted(state["origin"][:size], feedback.origin)
        state["scores"][feedback.column, rows] = feedback.scores
        state["known"][rows, feedback.column] = True
        return state

    def threshold(self, state: State) -> np.ndarray:
        return self.threshold_at(state, self.level)

    def threshold_at(self, state: State, level: Level) -> np.ndarray:
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
