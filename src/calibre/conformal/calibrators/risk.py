"""Risk calibrators: thresholds from the mean loss of each candidate on a declared grid.

They keep the same state, a loss sum `[N, C, G]` per grid threshold and a count `[N, C]`
of known targets, so its size does not grow with the origins. They differ only in how
they choose a threshold from it. The grid is `[G]` or `[N, G]` increasing thresholds in
the units of the score. Every known target counts once, with no window.
"""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, State
from calibre.conformal.losses import Loss


class MinRisk(Calibrator):
    """The grid threshold with the least mean loss over the known targets, per node and
    column.

    Ties go to the smallest threshold. A result at either end of the grid can mean that
    the minimizer is outside it. With `Signed` and `Newsvendor(h, p)`, the minimizer is
    the empirical quantile of the scores at `p / (h + p)`, rank `ceil(n * p / (h + p))`,
    when the grid contains it.
    """

    def __init__(self, loss: Loss, grid: np.ndarray) -> None:
        self.loss = loss
        self.grid = _check_grid(grid)

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        return _initial_sums(self.grid, n_nodes, n_columns)

    def update(self, state: State, feedback: Feedback) -> State:
        return _add_losses(state, feedback, self.loss, self.grid)

    def threshold(self, state: State) -> np.ndarray:
        # The count is the same for every grid value, so the least sum is the least mean.
        best = _grid_value(self.grid, np.argmin(state["loss_sum"], axis=-1))
        return np.where(state["count"] > 0, best, np.inf).astype(np.float32)


class RiskControl(Calibrator):
    """Conformal risk control (Angelopoulos et al. 2024, 2026): the smallest grid threshold
    whose corrected mean loss is at most `alpha`, per node and column.

    With n known targets and a loss of at most B, a threshold is feasible when
    `(loss sum + B) / (n + 1) <= alpha`. No feasible grid threshold gives inf. The loss
    must have a finite `maximum`.

    With `Miss`, the loss is nonincreasing in the threshold, and the result is the
    `SplitQuantile(1 - alpha)` threshold, rounded up to the grid.
    """

    def __init__(self, loss: Loss, grid: np.ndarray, alpha: float) -> None:
        if not np.isfinite(loss.maximum):
            raise ValueError("risk control needs a loss with a finite maximum")
        if not 0 < alpha < loss.maximum:
            raise ValueError(f"alpha must be between zero and {loss.maximum}, got {alpha}")
        self.loss = loss
        self.grid = _check_grid(grid)
        self.alpha = alpha

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        return _initial_sums(self.grid, n_nodes, n_columns)

    def update(self, state: State, feedback: Feedback) -> State:
        return _add_losses(state, feedback, self.loss, self.grid)

    def threshold(self, state: State) -> np.ndarray:
        count = state["count"][..., None]
        feasible = state["loss_sum"] + self.loss.maximum <= self.alpha * (count + 1)
        first = _grid_value(self.grid, np.argmax(feasible, axis=-1))
        return np.where(feasible.any(axis=-1), first, np.inf).astype(np.float32)


def _check_grid(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim not in (1, 2) or grid.shape[-1] < 1:
        raise ValueError(f"grid must be [G] or [N, G], got shape {grid.shape}")
    if not np.isfinite(grid).all() or (np.diff(grid, axis=-1) <= 0).any():
        raise ValueError("grid must be finite and strictly increasing")
    return grid


def _initial_sums(grid: np.ndarray, n_nodes: int, n_columns: int) -> State:
    if grid.ndim == 2 and len(grid) != n_nodes:
        raise ValueError(f"grid has {len(grid)} rows for {n_nodes} nodes")
    return {
        "loss_sum": np.zeros((n_nodes, n_columns, grid.shape[-1])),
        "count": np.zeros((n_nodes, n_columns), dtype=np.int64),
    }


def _add_losses(state: State, feedback: Feedback, loss: Loss, grid: np.ndarray) -> State:
    """Add the loss that each grid threshold would have had on the known targets."""
    lower, upper = feedback.bounds(grid)
    losses = loss.loss(lower, upper, feedback.target[..., None], feedback.censored[..., None])
    known = np.isfinite(feedback.target)  # [K, N]
    losses = np.where(known[..., None], losses, 0)  # [K, N, G]
    columns = np.eye(state["count"].shape[1])[feedback.column]  # [K, C]
    state["loss_sum"] += np.einsum("kng,kc->ncg", losses, columns)
    state["count"] += (known.T @ columns).astype(np.int64)
    return state


def _grid_value(grid: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Grid thresholds `[N, C]` at grid positions `index` `[N, C]`."""
    rows = np.broadcast_to(grid, (len(index), grid.shape[-1]))
    return np.take_along_axis(rows, index, axis=1)
