"""Conformal risk control: the smallest grid threshold whose corrected mean loss is low."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, State
from calibre.conformal.losses import Loss


class RiskControl(Calibrator):
    """Conformal risk control (Angelopoulos et al. 2024, 2026): the smallest grid threshold
    whose corrected mean loss is at most `alpha`, per node and column.

    The grid is `[G]` or `[N, G]` increasing thresholds in the units of the score. With n
    known targets and a loss of at most B, a threshold is feasible when
    `(loss sum + B) / (n + 1) <= alpha`. No feasible grid threshold gives inf. The state
    is a loss sum `[N, C, G]` per grid threshold and a count `[N, C]` of known targets, so
    its size does not grow with the origins. Every known target counts once, with no
    window.

    With `Miss`, the loss is nonincreasing in the threshold, and the result is the
    `SplitQuantile(1 - alpha)` threshold, rounded up to the grid.
    """

    def __init__(self, loss: Loss, grid: np.ndarray, alpha: float) -> None:
        if not 0 < alpha < loss.maximum:
            raise ValueError(f"alpha must be between zero and {loss.maximum}, got {alpha}")
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim not in (1, 2) or grid.shape[-1] < 1:
            raise ValueError(f"grid must be [G] or [N, G], got shape {grid.shape}")
        if not np.isfinite(grid).all() or (np.diff(grid, axis=-1) <= 0).any():
            raise ValueError("grid must be finite and strictly increasing")
        self.loss = loss
        self.grid = grid
        self.alpha = alpha

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        if self.grid.ndim == 2 and len(self.grid) != n_nodes:
            raise ValueError(f"grid has {len(self.grid)} rows for {n_nodes} nodes")
        return {
            "loss_sum": np.zeros((n_nodes, n_columns, self.grid.shape[-1])),
            "count": np.zeros((n_nodes, n_columns), dtype=np.int64),
        }

    def update(self, state: State, feedback: Feedback) -> State:
        """Add the loss that each grid threshold would have had on the known targets."""
        lower, upper = feedback.score.bound(feedback.point[..., None], self.grid)
        known = np.isfinite(feedback.target)  # [K, N]
        losses = self.loss.loss(lower, upper, feedback.target[..., None])
        losses = np.where(known[..., None], losses, 0)  # [K, N, G]
        columns = np.eye(state["count"].shape[1])[feedback.column]  # [K, C]
        state["loss_sum"] += np.einsum("kng,kc->ncg", losses, columns)
        state["count"] += (known.T @ columns).astype(np.int64)
        return state

    def threshold(self, state: State) -> np.ndarray:
        count = state["count"][..., None]
        feasible = state["loss_sum"] + self.loss.maximum <= self.alpha * (count + 1)
        rows = np.broadcast_to(self.grid, (len(feasible), self.grid.shape[-1]))
        first = np.take_along_axis(rows, np.argmax(feasible, axis=-1), axis=1)
        return np.where(feasible.any(axis=-1), first, np.inf).astype(np.float32)
