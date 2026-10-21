"""Conformal PID with a tangent integrator and causal rolling AR(1) scorecasting."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, Level, State, check_level
from calibre.conformal.calibrators.ranks import empirical_quantile


class ConformalPID(Calibrator):
    """Track nonnegative scalar scores, normally from Absolute, per node and column.

    Seed explicitly with revealed calibration scores before learning from feedback.
    P tracks the quantile, I integrates coverage error, and anticipation forecasts
    scores with AR(1), not the paper's Theta model. Rates use a fixed seed scale.
    Set ahead to each target column's last covered step: 1..H for Step, or the
    lead time for LeadTime. The default is one step for every column.
    Negative thresholds and signed-infinity saturation are intentional outputs.
    """

    def __init__(
        self,
        level: Level,
        learning_rate: float = 0.05,
        integral_gain: float = 0.1,
        saturation_scale: float = 5.0,
        window: int = 256,
        scorecast_window: int = 64,
        scorecast: bool = True,
        *,
        ahead: int | np.ndarray = 1,
    ) -> None:
        self.level = check_level(level)
        for name, value in (
            ("learning_rate", learning_rate),
            ("integral_gain", integral_gain),
            ("saturation_scale", saturation_scale),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 0 or not np.isfinite(array):
                raise ValueError(f"{name} must be a finite scalar")
            if float(array) < 0 or (name != "integral_gain" and float(array) == 0):
                raise ValueError(
                    f"{name} must be {'nonnegative' if name == 'integral_gain' else 'positive'}"
                )
        for name, value in (("window", window), ("scorecast_window", scorecast_window)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(scorecast, (bool, np.bool_)):
            raise ValueError("scorecast must be a boolean")
        steps_ahead = np.asarray(ahead)
        if (
            steps_ahead.ndim not in (0, 1)
            or steps_ahead.size == 0
            or steps_ahead.dtype.kind not in "iu"
            or (steps_ahead < 1).any()
            or (steps_ahead > np.iinfo(np.int64).max).any()
        ):
            raise ValueError("ahead must be a positive int64 scalar or [C] array")
        self.ahead = steps_ahead.astype(np.int64, copy=True)
        self.learning_rate = float(learning_rate)
        self.integral_gain = float(integral_gain)
        self.saturation_scale = float(saturation_scale)
        self.window = int(window)
        self.scorecast_window = int(scorecast_window)
        self.scorecast = bool(scorecast)

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        np.broadcast_to(self.level, (n_nodes, n_columns))
        np.broadcast_to(self.ahead, (n_columns,))
        return {
            "scores": np.full((n_nodes, n_columns, self.window), np.nan, dtype=np.float64),
            "count": np.zeros((n_nodes, n_columns), dtype=np.int64),
            "next": np.zeros((n_nodes, n_columns), dtype=np.int64),
            "scale": np.zeros((n_nodes, n_columns), dtype=np.float64),
            "proportional": np.zeros((n_nodes, n_columns), dtype=np.float64),
            "cumulative_error": np.zeros((n_nodes, n_columns), dtype=np.float64),
            "n_updates": np.zeros((n_nodes, n_columns), dtype=np.int64),
        }

    def seed(self, state: State, scores: np.ndarray) -> State:
        """Seed an empty state with nonnegative scores [K, N, C] in time order.

        NaN holes are skipped; every cell needs at least one finite score. Use only
        scores known at the first issuing origin. No past misses are invented.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 3 or scores.shape[1:] != state["count"].shape:
            raise ValueError("seed scores must be [K, N, C] matching the state")
        if np.isinf(scores).any() or (scores < 0).any():
            raise ValueError("seed scores must be nonnegative and finite or NaN")
        if not np.isfinite(scores).any(axis=0).all():
            raise ValueError("seed needs at least one finite score per node and column")
        if state["count"].any():
            raise ValueError("seed requires an empty calibration state")
        for row in scores:
            nodes, columns = np.nonzero(np.isfinite(row))
            slots = state["next"][nodes, columns]
            state["scores"][nodes, columns, slots] = row[nodes, columns]
            state["next"][nodes, columns] = (slots + 1) % self.window
            state["count"][nodes, columns] = np.minimum(
                state["count"][nodes, columns] + 1, self.window
            )
        retained = state["scores"]
        state["scale"] = np.maximum(
            np.maximum(np.nanstd(retained, axis=-1), 0.1 * np.nanmean(retained, axis=-1)), 1e-8
        )
        shape = state["count"].shape
        quantile = empirical_quantile(
            retained.reshape(-1, self.window).T, np.broadcast_to(self.level, shape).ravel()
        ).reshape(shape)
        state["proportional"] = quantile - self._scorecast(state)
        return state

    def update(self, state: State, feedback: Feedback) -> State:
        if feedback.score.parameter_shape():
            raise ValueError("ConformalPID requires nonnegative scalar scores, normally Absolute")
        if not (state["count"] > 0).all():
            raise ValueError("ConformalPID requires seed before feedback")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = feedback.scores
        if (scores[np.isfinite(scores)] < 0).any():
            raise ValueError("ConformalPID requires nonnegative scalar scores, normally Absolute")
        levels = np.broadcast_to(self.level, state["count"].shape)
        for row, column in enumerate(feedback.column):
            nodes = np.flatnonzero(np.isfinite(scores[row]))
            error = (scores[row, nodes] > feedback.issued[row, nodes]) - (1 - levels[nodes, column])
            state["proportional"][nodes, column] += (
                self.learning_rate * state["scale"][nodes, column] * error
            )
            state["cumulative_error"][nodes, column] += error
            state["n_updates"][nodes, column] += 1
            slots = state["next"][nodes, column]
            state["scores"][nodes, column, slots] = scores[row, nodes]
            state["next"][nodes, column] = (slots + 1) % self.window
            state["count"][nodes, column] = np.minimum(
                state["count"][nodes, column] + 1, self.window
            )
        return state

    def threshold(self, state: State) -> np.ndarray:
        threshold = state["proportional"] + self._integrator(state) + self._scorecast(state)
        return np.where(state["count"] > 0, threshold, np.inf).astype(np.float32)

    def _scorecast(self, state: State) -> np.ndarray:
        if not self.scorecast:
            return np.zeros_like(state["scale"])
        length = min(self.window, self.scorecast_window)
        counts = np.minimum(state["count"], length)
        # Right-align chronological history, including after the ring wraps.
        offsets = np.arange(-length, 0)
        slots = (state["next"][..., None] + offsets) % self.window
        scores = np.take_along_axis(state["scores"], slots, axis=-1)
        valid = offsets >= -counts[..., None]
        scores = np.where(valid, scores, 0.0)
        mean = scores.sum(axis=-1) / np.maximum(counts, 1)
        pairs = np.maximum(counts - 1, 1)
        paired = valid[..., :-1] & valid[..., 1:]
        x, y = scores[..., :-1], scores[..., 1:]
        mean_x = np.where(paired, x, 0).sum(axis=-1) / pairs
        mean_y = np.where(paired, y, 0).sum(axis=-1) / pairs
        centered_x = np.where(paired, x - mean_x[..., None], 0)
        centered_y = np.where(paired, y - mean_y[..., None], 0)
        denominator = np.sum(centered_x**2, axis=-1)
        ridge = np.maximum(state["scale"] ** 2 * 1e-6, 1e-12)
        slope = np.sum(centered_x * centered_y, axis=-1) / (denominator + ridge)
        slope = np.clip(slope, -0.99, 0.99)
        # Mean reversion must propagate the fitted intercept over all ahead steps.
        equilibrium = (mean_y - slope * mean_x) / (1 - slope)
        forecast = np.maximum(
            0.0, equilibrium + slope**self.ahead * (scores[..., -1] - equilibrium)
        )
        return np.where(counts < 3, mean, forecast)

    def _integrator(self, state: State) -> np.ndarray:
        if self.integral_gain == 0:
            return np.zeros_like(state["scale"])
        time = state["n_updates"].astype(np.float64) + 1
        argument = state["cumulative_error"] * np.log(time) / (self.saturation_scale * time)
        out = np.empty_like(argument)
        upper = argument >= np.pi / 2
        lower = argument <= -np.pi / 2
        inside = ~(upper | lower)
        out[upper], out[lower] = np.inf, -np.inf
        out[inside] = self.integral_gain * state["scale"][inside] * np.tan(argument[inside])
        return out
