"""Empirical risk minimization over a grid of thresholds, for losses of any shape."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, State
from calibre.conformal.losses import Loss


class MinRisk(Calibrator):
    """The grid threshold with the least mean loss over the known targets, per node and
    column.

    `grid` is `[G]` or `[N, G]` increasing thresholds in the units of the score. Ties go
    to the smallest threshold. Every known target counts once, with no window. The state
    is a loss sum per grid threshold, so its size does not grow with the origins.

    With `Signed` and `Newsvendor(h, p)`, the minimizer is the empirical quantile of the
    scores at `p / (h + p)`, rank `ceil(n * p / (h + p))`, when the grid contains it.
    """

    def __init__(self, loss: Loss, grid: np.ndarray) -> None:
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim not in (1, 2) or grid.shape[-1] < 1:
            raise ValueError(f"grid must be [G] or [N, G], got shape {grid.shape}")
        if not np.isfinite(grid).all() or (np.diff(grid, axis=-1) <= 0).any():
            raise ValueError("grid must be finite and strictly increasing")
        self.loss = loss
        self.grid = grid

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        if self.grid.ndim == 2 and len(self.grid) != n_nodes:
            raise ValueError(f"grid has {len(self.grid)} rows for {n_nodes} nodes")
        size = self.grid.shape[-1]
        return {
            "risk": np.zeros((n_nodes, n_columns, size)),
            "count": np.zeros((n_nodes, n_columns), dtype=np.int64),
        }

    def update(self, state: State, feedback: Feedback) -> State:
        lower, upper = feedback.bounds(self.grid)
        losses = self.loss.loss(
            lower, upper, feedback.target[..., None], feedback.censored[..., None]
        )  # [K, N, G]
        known = np.isfinite(feedback.target)  # [K, N]
        losses = np.where(known[..., None], losses, 0)
        for column in np.unique(feedback.column):
            rows = feedback.column == column
            state["risk"][:, column] += losses[rows].sum(axis=0)
            state["count"][:, column] += known[rows].sum(axis=0)
        return state

    def threshold(self, state: State) -> np.ndarray:
        best = np.argmin(state["risk"], axis=-1)  # [N, C]; equal counts per grid value
        grid = np.broadcast_to(self.grid, (len(best), self.grid.shape[-1]))
        out = np.take_along_axis(grid, best, axis=1)
        return np.where(state["count"] > 0, out, np.inf).astype(np.float32)
