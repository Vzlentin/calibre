"""Split NexCP: geometric calibration weights and a unit test-point mass at infinity."""

import numpy as np

from calibre.conformal.calibrators.base import (
    Feedback,
    Level,
    QuantileCalibrator,
    State,
    check_level,
)
from calibre.conformal.calibrators.ranks import weighted_score_quantile


class NexCP(QuantileCalibrator):
    """Weighted split conformal at `level`, independently per node and column.

    In chronological retained-score order, weights are rho**n, ..., rho. The test
    point has weight 1 at +inf. `rho=1` recovers ordinary split conformal. `window`
    keeps the last finite scores per cell; None retains all. Missing scores and
    calendar gaps do not advance this clock. Use Absolute for symmetric intervals.

    The split claim requires a model trained separately from calibration data.
    Drift incurs the paper's unknown coverage-gap bound, not conditional coverage.
    Reference: Barber et al., https://arxiv.org/abs/2202.13415, equation (11).
    """

    def __init__(self, level: Level, rho: float = 0.99, window: int | None = None) -> None:
        decay = np.asarray(rho, dtype=np.float64)
        if (
            np.asarray(rho).dtype.kind == "b"
            or decay.ndim != 0
            or not np.isfinite(decay)
            or not 0 < decay <= 1
        ):
            raise ValueError("rho must be a scalar strictly above zero and at most one")
        if window is not None and (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
            or window < 1
        ):
            raise ValueError("window must be a positive integer or None")
        self.level = check_level(level)
        self.rho = float(decay)
        self.window = None if window is None else int(window)

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        np.broadcast_to(self.level, (n_nodes, n_columns))
        return {
            "scores": np.empty((n_nodes, n_columns, 0), dtype=np.float32),
            "count": np.zeros((n_nodes, n_columns), dtype=np.int64),
            "next": np.zeros((n_nodes, n_columns), dtype=np.int64),
        }

    def seed(self, state: State, scores: np.ndarray) -> State:
        """Seed an empty state with held-out scores [K, N, C] in time order.

        NaN is missing; infinity is rejected. Scores must already be known at the
        first issuance, from the same separately trained model and score function.
        """
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 3 or scores.shape[1:] != state["count"].shape:
            raise ValueError("seed scores must be [K, N, C] matching the state")
        if np.isinf(scores).any():
            raise ValueError("seed scores must be finite or NaN")
        if state["count"].any():
            raise ValueError("seed requires an empty calibration state")
        for row in scores:
            nodes, columns = np.nonzero(np.isfinite(row))
            self._append(state, nodes, columns, row[nodes, columns])
        return state

    def update(self, state: State, feedback: Feedback) -> State:
        if feedback.score.parameter_shape():
            raise ValueError("NexCP requires scalar scores, normally Absolute or Signed")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = feedback.scores
        for row, column in enumerate(feedback.column):
            nodes = np.flatnonzero(np.isfinite(scores[row]))
            self._append(state, nodes, column, scores[row, nodes])
        return state

    def threshold(self, state: State) -> np.ndarray:
        return self.threshold_at(state, self.level)

    def threshold_at(self, state: State, level: Level) -> np.ndarray:
        shape = state["count"].shape
        levels = np.broadcast_to(check_level(level), shape)
        length = state["scores"].shape[-1]
        if length == 0 or not (levels < self.rho).any():
            return np.full(shape, np.inf, dtype=np.float32)
        offsets = np.arange(-length, 0)
        slots = (state["next"][..., None] + offsets) % length
        scores = np.take_along_axis(state["scores"], slots, axis=-1)
        scores = np.where(offsets >= -state["count"][..., None], scores, np.nan)
        # Right alignment gives the newest score weight rho, not 1.
        weights = np.power(self.rho, -offsets)
        out = weighted_score_quantile(
            scores.reshape(-1, length).T, weights, levels.ravel()
        ).reshape(shape)
        # Every finite geometric history has finite-mass CDF strictly below rho.
        return np.where(levels < self.rho, out, np.inf).astype(np.float32)

    def _append(
        self,
        state: State,
        nodes: np.ndarray,
        columns: int | np.ndarray,
        values: np.ndarray,
    ) -> None:
        if len(nodes) == 0:
            return
        slots = state["next"][nodes, columns]
        length = state["scores"].shape[-1]
        required = int(slots.max()) + 1
        if required > length:
            allocated = max(required, max(1, 2 * length))
            if self.window is not None:
                allocated = min(allocated, self.window)
            extra = np.full(state["count"].shape + (allocated - length,), np.nan, dtype=np.float32)
            state["scores"] = np.concatenate([state["scores"], extra], axis=-1)
        state["scores"][nodes, columns, slots] = values
        state["next"][nodes, columns] = slots + 1
        state["count"][nodes, columns] += 1
        if self.window is not None:
            state["next"][nodes, columns] %= self.window
            state["count"][nodes, columns] = np.minimum(state["count"][nodes, columns], self.window)
