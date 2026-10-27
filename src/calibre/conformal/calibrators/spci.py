"""Sequential predictive conformal intervals from conditional residual quantiles."""

import numpy as np

from calibre.conformal.calibrators.base import Calibrator, Feedback, Level, State, check_level


class SPCI(Calibrator):
    """Fit a quantile forest per node/column using causally aligned signed errors.

    Use Asymmetric and set ahead to the last covered step of each target column:
    1..H for Step, or the lead time for LeadTime. Origins are calendar indexes,
    not retained-score positions. Each historical context contains only errors
    revealed by that forecast's origin. No fitted forest is retained.
    Requires the spci extra (scikit-learn) when a cell becomes ready.
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
        for name, value, minimum in (
            ("window", window, 2),
            ("n_lags", n_lags, 1),
            ("min_train_samples", min_train_samples, 1),
            ("beta_grid_size", beta_grid_size, 2),
            ("n_estimators", n_estimators, 1),
            ("min_samples_leaf", min_samples_leaf, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if n_lags + min_train_samples > window:
            raise ValueError("window must be at least n_lags + min_train_samples")
        if max_depth is not None and (
            isinstance(max_depth, (bool, np.bool_))
            or not isinstance(max_depth, (int, np.integer))
            or max_depth < 1
        ):
            raise ValueError("max_depth must be a positive integer or None")
        if not isinstance(bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be boolean")
        if (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or not 0 <= random_state <= np.iinfo(np.uint32).max
        ):
            raise ValueError("random_state must be an integer in [0, 2**32 - 1]")
        if n_jobs is not None and (
            isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer))
            or n_jobs == 0
        ):
            raise ValueError("n_jobs must be a nonzero integer or None")
        steps = np.asarray(ahead)
        if (
            steps.ndim not in (0, 1)
            or steps.size == 0
            or steps.dtype.kind not in "iu"
            or (steps < 1).any()
            or (steps > np.iinfo(np.int64).max).any()
        ):
            raise ValueError("ahead must be a positive int64 scalar or [C] array")
        self.ahead = steps.astype(np.int64, copy=True)
        self.window = int(window)
        self.n_lags = int(n_lags)
        self.min_train_samples = int(min_train_samples)
        self.beta_grid_size = int(beta_grid_size)
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self.n_jobs = None if n_jobs is None else int(n_jobs)

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
        if feedback.score.parameter_shape() != (2,):
            raise ValueError("SPCI requires paired corrections, normally Asymmetric")
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
        from sklearn.ensemble import RandomForestRegressor

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
        support = np.flatnonzero(weights > 0)
        order = support[np.argsort(targets[support], kind="stable")]
        cumulative = np.cumsum(weights[order])
        cumulative /= cumulative[-1]
        beta = np.linspace(0, 1 - level, self.beta_grid_size)
        quantiles = np.clip(np.concatenate([beta, level + beta]), 0, 1)
        ranks = np.minimum(np.searchsorted(cumulative, quantiles), len(order) - 1)
        bounds = targets[order[ranks]].astype(np.float64).reshape(2, self.beta_grid_size)
        best = np.argmin(bounds[1] - bounds[0])
        return bounds[:, best].astype(np.float32)
