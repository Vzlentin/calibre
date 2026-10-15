"""EnbPI residual calibration, separate from bootstrap point forecasting."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, Level, State, check_level
from calibre.conformal.calibrators.ranks import empirical_interval


class EnbPI(Calibrator):
    """Rolling asymmetric empirical intervals at level, per node and column.

    Use with the Asymmetric score. Seed with signed OOB residuals from a separately
    fitted bootstrap ensemble; subsequent errors use the archived issued center.
    window counts finite errors per node/column. Bootstrap learners stay outside
    this calibrator and its state. There is no exchangeable finite-sample claim.
    """

    def __init__(self, level: Level, window: int, beta_grid_size: int = 21) -> None:
        for name, value, minimum in (
            ("window", window, 1),
            ("beta_grid_size", beta_grid_size, 2),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        self.level = check_level(level)
        self.window = window
        self.beta_grid_size = beta_grid_size

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        np.broadcast_to(self.level, (n_nodes, n_columns))
        return {
            "residuals": np.full((n_nodes, n_columns, self.window), np.nan, dtype=np.float32),
            "count": np.zeros((n_nodes, n_columns), dtype=np.int64),
            "next": np.zeros((n_nodes, n_columns), dtype=np.int64),
        }

    def seed(self, state: State, residuals: np.ndarray) -> State:
        """Initialize an empty state with known signed errors [K, N, C] in time order.

        NaN is missing; infinity is rejected. The caller must use only errors known
        at the first issuing origin, from the same frozen point model.
        """
        residuals = np.asarray(residuals, dtype=np.float32)
        if residuals.ndim != 3 or residuals.shape[1:] != state["count"].shape:
            raise ValueError("seed residuals must be [K, N, C] matching the state")
        if np.isinf(residuals).any():
            raise ValueError("seed residuals must be finite or NaN")
        if state["count"].any():
            raise ValueError("seed requires an empty calibration state")
        for row in residuals:
            nodes, columns = np.nonzero(np.isfinite(row))
            slots = state["next"][nodes, columns]
            state["residuals"][nodes, columns, slots] = row[nodes, columns]
            state["next"][nodes, columns] = (slots + 1) % self.window
            state["count"][nodes, columns] = np.minimum(
                state["count"][nodes, columns] + 1, self.window
            )
        return state

    def update(self, state: State, feedback: Feedback) -> State:
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = feedback.target - feedback.point
        for row, column in enumerate(feedback.column):
            nodes = np.flatnonzero(np.isfinite(residuals[row]))
            slots = state["next"][nodes, column]
            state["residuals"][nodes, column, slots] = residuals[row, nodes]
            state["next"][nodes, column] = (slots + 1) % self.window
            state["count"][nodes, column] = np.minimum(
                state["count"][nodes, column] + 1, self.window
            )
        return state

    def threshold(self, state: State) -> np.ndarray:
        n_nodes, n_columns, window = state["residuals"].shape
        scores = state["residuals"].transpose(2, 0, 1).reshape(window, -1)
        levels = np.broadcast_to(self.level, (n_nodes, n_columns)).ravel()
        return empirical_interval(scores, levels, self.beta_grid_size).reshape(
            n_nodes, n_columns, 2
        )
