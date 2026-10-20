"""Local lag-row bootstrap ensembles with OOB-weighted point forecasts."""

from typing import Any

import numpy as np
from sklearn.base import clone

from calibre.forecast.models.base import FitResiduals, FittedResiduals, Forecaster, Window


class BootstrapEnsemble(Forecaster):
    """One frozen bootstrap ensemble per series, with shared row resamples.

    Only one-step lag regression is supported; covariates are rejected. Cloning
    keeps preprocessing inside each bootstrap fit. OOB exclusion is not temporal
    independence: neighboring lag rows still overlap.
    """

    def __init__(
        self,
        estimator: Any,
        n_lags: int,
        n_estimators: int = 20,
        random_state: int = 0,
        max_estimators: int | None = None,
    ) -> None:
        for name, value in (("n_lags", n_lags), ("n_estimators", n_estimators)):
            _positive_integer(name, value)
        budget = max(4 * n_estimators, n_estimators + 20)
        if max_estimators is not None:
            _positive_integer("max_estimators", max_estimators)
            budget = max_estimators
        if budget < n_estimators:
            raise ValueError("max_estimators must be at least n_estimators")
        self.estimator = estimator
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_estimators = budget

    def fit(self, window: Window) -> "FittedBootstrap":
        _check_window(window, self.n_lags + 2)
        if not np.isfinite(window.y).all():
            raise ValueError("bootstrap history must be finite")
        rows = np.lib.stride_tricks.sliding_window_view(window.y, self.n_lags + 1, axis=1)
        x, y = rows[..., :-1], rows[..., -1]
        n_series, n_rows = y.shape
        rng = np.random.default_rng(self.random_state)
        estimators: list[list[Any]] = [[] for _ in range(n_series)]
        indices, masks = [], []
        sums = np.zeros((n_series, n_rows), dtype=np.float64)
        counts = np.zeros(n_rows, dtype=np.int64)
        for _ in range(self.max_estimators):
            sampled = np.sort(rng.integers(0, n_rows, size=n_rows))
            oob = np.ones(n_rows, dtype=bool)
            oob[sampled] = False
            for node in range(n_series):
                model = clone(self.estimator)
                if "random_state" in model.get_params(deep=False):
                    model.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
                model.fit(x[node, sampled], y[node, sampled])
                if oob.any():
                    sums[node, oob] += _predict(model, x[node, oob])
                estimators[node].append(model)
            counts += oob
            indices.append(sampled)
            masks.append(oob)
            if len(indices) >= self.n_estimators and (counts > 0).all():
                break
        if (counts == 0).any():
            raise RuntimeError("some training rows have no OOB model; increase max_estimators")
        oob_mask = np.stack(masks, axis=1)  # [K, M]
        oob_predictions = sums / counts
        weights = (oob_mask / counts[:, None]).mean(axis=0)
        residuals = FitResiduals(
            np.arange(self.n_lags - 1, window.y.shape[1] - 1),
            (y - oob_predictions).T[..., None],
        )
        return FittedBootstrap(
            tuple(tuple(local) for local in estimators),
            self.n_lags,
            window.y.shape[1] - 1,
            np.stack(indices),
            oob_mask,
            oob_predictions.T,
            weights,
            residuals,
        )


class FittedBootstrap(FittedResiduals):
    """Frozen learners, their OOB diagnostics, and initialization residuals."""

    def __init__(
        self,
        estimators: tuple[tuple[Any, ...], ...],
        n_lags: int,
        fit_origin: int,
        bootstrap_indices: np.ndarray,
        oob_mask: np.ndarray,
        oob_predictions: np.ndarray,
        ensemble_weights: np.ndarray,
        residuals: FitResiduals,
    ) -> None:
        self.estimators = estimators
        self.n_lags = n_lags
        self.fit_origin = fit_origin
        self.bootstrap_indices = _readonly(bootstrap_indices)
        self.oob_mask = _readonly(oob_mask)
        self.oob_predictions = _readonly(oob_predictions)
        self.ensemble_weights = _readonly(ensemble_weights)
        self._residuals = residuals

    def predict(self, window: Window) -> np.ndarray:
        _check_window(window, self.fit_origin + 1)
        if len(window.y) != len(self.estimators):
            raise ValueError("prediction series do not match the fitted bootstrap ensemble")
        x = window.y[:, -self.n_lags :]
        if not np.isfinite(x).all():
            raise ValueError("bootstrap lags must be finite")
        points = np.empty((len(x), 1), dtype=np.float32)
        for node, local in enumerate(self.estimators):
            predictions = np.array([_predict(model, x[node : node + 1])[0] for model in local])
            points[node, 0] = predictions @ self.ensemble_weights
        if not np.isfinite(points).all():
            raise ValueError("bootstrap ensemble returned nonfinite points")
        return points

    def residuals(self) -> FitResiduals:
        return self._residuals


def _check_window(window: Window, minimum: int) -> None:
    if window.horizon != 1:
        raise ValueError("BootstrapEnsemble supports horizon=1 only")
    if window.x:
        raise ValueError("BootstrapEnsemble supports lags only, not covariates")
    if window.y.ndim != 2 or len(window.y) == 0 or window.y.shape[1] < minimum:
        raise ValueError(f"bootstrap history needs at least {minimum} periods per series")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(x), dtype=np.float64)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError("bootstrap learner returned invalid predictions")
    return values


def _readonly(values: np.ndarray) -> np.ndarray:
    values.flags.writeable = False
    return values
