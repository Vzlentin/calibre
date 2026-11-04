"""Sequential predictive conformal intervals from conditional residual quantiles."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from calibre.conformal.calibrators.base import (
    Calibrator,
    Feedback,
    Level,
    State,
    check_ahead,
    check_integer,
    check_level,
)
from calibre.conformal.calibrators.ranks import weighted_interval


class SPCI(Calibrator):
    """Fit a quantile forest per node/column using causally aligned signed errors.

    Use Asymmetric and set ahead to the last covered step of each target column:
    1..H for Step, or the lead time for LeadTime. Origins are calendar indexes,
    not retained-score positions. Each historical context contains only errors
    revealed by that forecast's origin. No fitted forest is retained.
    Requires the bootstrap extra (scikit-learn).
    """

    def __init__(
        self,
        level: Level,
        window: int,
        n_lags: int,
        *,
        ahead: int | np.ndarray,
        min_train_samples: int = 20,
        beta_grid_size: int = 21,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_leaf: int = 5,
        bootstrap: bool = True,
        random_state: int = 0,
        n_jobs: int | None = 1,
    ) -> None:
        self.level = check_level(level)
        self.window = check_integer("window", window, 2)
        self.n_lags = check_integer("n_lags", n_lags, 1)
        self.min_train_samples = check_integer("min_train_samples", min_train_samples, 1)
        self.beta_grid_size = check_integer("beta_grid_size", beta_grid_size, 2)
        self.n_estimators = check_integer("n_estimators", n_estimators, 1)
        self.min_samples_leaf = check_integer("min_samples_leaf", min_samples_leaf, 1)
        if self.n_lags + self.min_train_samples > self.window:
            raise ValueError("window must be at least n_lags + min_train_samples")
        self.max_depth = None if max_depth is None else check_integer("max_depth", max_depth, 1)
        if not isinstance(bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be boolean")
        self.bootstrap = bool(bootstrap)
        self.random_state = check_integer("random_state", random_state, 0, 2**32 - 1)
        self.n_jobs = None if n_jobs is None else check_integer("n_jobs", n_jobs)
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer or None")
        self.ahead = check_ahead(ahead)

    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        np.broadcast_to(self.level, (n_nodes, n_columns))
        np.broadcast_to(self.ahead, (n_columns,))
        shape = (n_nodes, n_columns)
        return {
            "residuals": np.full((*shape, self.window), np.nan, dtype=np.float32),
            "origins": np.full((*shape, self.window), -1, dtype=np.int64),
            "count": np.zeros(shape, dtype=np.int64),
            "next": np.zeros(shape, dtype=np.int64),
            "last_origin": np.full(shape, -1, dtype=np.int64),
        }

    def seed(self, state: State, residuals: np.ndarray, *, origins: np.ndarray) -> State:
        """Seed an empty state with signed errors [K, N, C] and issuing origins [K].

        Origins must be nonnegative and strictly increasing. NaN errors are
        missing; infinity is rejected. All targets must already be known at the
        first issuing origin. Seeds must come from appropriate historical forecasts,
        not in-sample fitted values. No past bounds or feedback misses are invented.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = np.asarray(residuals, dtype=np.float32)
        dates = np.asarray(origins)
        if residuals.ndim != 3 or residuals.shape[1:] != state["count"].shape:
            raise ValueError("seed residuals must be [K, N, C] matching the state")
        if np.isinf(residuals).any():
            raise ValueError("seed residuals must be finite float32 values or NaN")
        if (
            dates.shape != (len(residuals),)
            or dates.dtype.kind not in "iu"
            or (dates < 0).any()
            or (dates > np.iinfo(np.int64).max).any()
            or (dates[1:] <= dates[:-1]).any()
        ):
            raise ValueError("seed origins must be strictly increasing nonnegative int64 [K]")
        if (state["last_origin"] >= 0).any():
            raise ValueError("seed requires an empty calibration state")
        for row, origin in zip(residuals, dates, strict=True):
            for column in range(row.shape[1]):
                self._append(state, int(origin), column, row[:, column])
        return state

    def update(self, state: State, feedback: Feedback) -> State:
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = feedback.target - feedback.point
        for row, (origin, column) in enumerate(zip(feedback.origin, feedback.column, strict=True)):
            self._append(state, int(origin), int(column), residuals[row])
        return state

    def threshold(self, state: State) -> np.ndarray:
        shape = state["count"].shape
        levels = np.broadcast_to(self.level, shape)
        ahead = np.broadcast_to(self.ahead, (shape[1],))
        out = np.broadcast_to([-np.inf, np.inf], (*shape, 2)).copy().astype(np.float32)
        offsets = np.arange(-self.window, 0)
        nodes, columns = np.nonzero(state["count"] >= self.n_lags + self.min_train_samples)
        for node, column in zip(nodes, columns, strict=True):
            count = int(state["count"][node, column])
            slots = (state["next"][node, column] + offsets[-count:]) % self.window
            history = state["residuals"][node, column, slots]
            dates = state["origins"][node, column, slots]
            features, targets = self._training_pairs(history, dates, int(ahead[column]))
            if len(targets) >= self.min_train_samples:
                out[node, column] = self._interval(
                    features, targets, history[-self.n_lags :], float(levels[node, column])
                )
        return out

    def _append(self, state: State, origin: int, column: int, residuals: np.ndarray) -> None:
        if origin < 0 or np.any(state["last_origin"][:, column] >= origin):
            raise ValueError("feedback origins must increase strictly within each column")
        nodes = np.flatnonzero(np.isfinite(residuals))
        slots = state["next"][nodes, column]
        state["residuals"][nodes, column, slots] = residuals[nodes]
        state["origins"][nodes, column, slots] = origin
        state["next"][nodes, column] = (slots + 1) % self.window
        state["count"][nodes, column] = np.minimum(state["count"][nodes, column] + 1, self.window)
        state["last_origin"][:, column] = origin

    def _training_pairs(
        self, history: np.ndarray, origins: np.ndarray, ahead: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # For target error e[o,h], only errors e[u,h] with u+h <= o were known.
        ends = np.searchsorted(origins, origins - ahead, side="right")
        ready = ends >= self.n_lags
        indices = ends[ready, None] + np.arange(-self.n_lags, 0)
        return history[indices], history[ready]

    def _interval(
        self, features: np.ndarray, targets: np.ndarray, context: np.ndarray, level: float
    ) -> np.ndarray:
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(features, targets)
        leaves = forest.apply(features) == forest.apply(context[None, :])[0]
        # QRF: each tree gives unit total mass, shared equally by its leaf's rows.
        weights = (leaves / leaves.sum(axis=0)).mean(axis=1)
        return weighted_interval(targets, weights, level, self.beta_grid_size)
