"""Rolling-origin point forecasts: fit, predict, and reconcile at each origin.

`rolling_forecasts` keeps no state between calls and stores nothing. Each window is a
read-only slice that ends at its origin, so a model sees only what was known there.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from calibre.data.hierarchy import Hierarchy
from calibre.data.panel import Panel
from calibre.forecast.models.base import (
    Covariate,
    FitResiduals,
    Fitted,
    FittedQuantiles,
    FittedResiduals,
    Forecaster,
    Window,
    check_quantiles,
)
from calibre.forecast.reconcile import Identity, Reconciler


@dataclass(frozen=True)
class Forecasts:
    """Points `[O, N, H]` and optional native intervals for increasing origins `[O]`.

    Native intervals add a final pair axis and require two `quantile_levels`.
    Crossed raw endpoints are ordered on a copy, never on the caller's array.
    """

    origins: np.ndarray
    points: np.ndarray
    initial_residuals: FitResiduals | None = None
    interval: np.ndarray | None = None
    quantile_levels: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.interval is None and self.quantile_levels is None:
            return
        if self.interval is None or self.quantile_levels is None:
            raise ValueError("native intervals and quantile_levels must be supplied together")
        object.__setattr__(self, "quantile_levels", check_quantiles(self.quantile_levels))
        if self.points.ndim != 3 or self.points.shape[0] != len(self.origins):
            raise ValueError("native forecast points must be [O, N, H] matching origins")
        if np.shape(self.interval) != self.points.shape + (2,):
            raise ValueError("native intervals must have shape points.shape + (2,)")
        interval = np.sort(np.asarray(self.interval, dtype=np.float32), axis=-1)
        if not np.isfinite(interval).all():
            raise ValueError("native intervals must be finite")
        object.__setattr__(self, "interval", interval)

    def residuals(self, actuals: np.ndarray) -> np.ndarray:
        """Return `actual - point` `[O, N, H]` from node actuals `[N, T]`.

        A target after the last period is NaN, which means not yet known.
        """
        n_periods = actuals.shape[1]
        targets = self.origins[:, None] + np.arange(1, self.points.shape[2] + 1)
        values = actuals[:, np.minimum(targets, n_periods - 1)].transpose(1, 0, 2)
        resid = values.astype(np.float32) - self.points
        return np.where((targets < n_periods)[:, None, :], resid, np.float32(np.nan))


def rolling_forecasts(
    panel: Panel,
    hierarchy: Hierarchy,
    model: Forecaster,
    reconciler: Reconciler,
    origins: np.ndarray,
    horizon: int,
    refit_every: int | None = None,
    covariates: Mapping[str, Covariate] | None = None,
    *,
    initial_residuals: bool = False,
) -> Forecasts:
    """Fit, predict, and reconcile at each origin, the index of the last observed period.

    The model forecasts the first S nodes, with S from the reconciler shape `(N, S)`. It
    is fitted at the first origin and again every `refit_every` origins. None fits once.
    initial_residuals collects fit-time OOB errors, reconciled to the node axis.
    It requires FittedResiduals and no refit, so seeds never mix model generations.
    FittedQuantiles also returns native intervals and requires Identity and no refit.
    """
    if initial_residuals and refit_every is not None:
        raise ValueError("initial residuals require a frozen model, without refit_every")
    origins = _check_schedule(origins, len(panel.periods), horizon, refit_every)
    covariates = dict(covariates or {})
    n_nodes, n_series = reconciler.shape
    if n_nodes != len(hierarchy.nodes) or n_series not in (hierarchy.n_bottom, n_nodes):
        raise ValueError(f"reconciler shape {reconciler.shape} does not fit the hierarchy")
    last = int(origins[-1])
    for name, covariate in covariates.items():
        _check_covariate(name, covariate, hierarchy.n_bottom, last, horizon)

    history = _series_rows(panel.values, hierarchy, n_series, "sum")
    inputs = {
        name: _series_rows(covariate.values, hierarchy, n_series, covariate.aggregate)
        for name, covariate in covariates.items()
    }
    reach = {name: _reach(covariate, horizon) for name, covariate in covariates.items()}
    periods = pd.date_range(panel.periods[0], periods=last + 1 + horizon, freq=panel.freq)

    points = np.empty((len(origins), n_nodes, horizon), dtype=np.float32)
    fitted: Fitted | None = None
    seed: FitResiduals | None = None
    intervals: np.ndarray | None = None
    quantile_levels: tuple[float, float] | None = None
    for index, origin in enumerate(origins):
        seen = int(origin) + 1
        window = Window(
            y=history[:, :seen],
            x={name: _visible(values, seen, reach[name]) for name, values in inputs.items()},
            periods=periods[: seen + horizon],
            horizon=horizon,
        )
        if fitted is None or (refit_every is not None and index % refit_every == 0):
            fitted = model.fit(window)
            if isinstance(fitted, FittedQuantiles):
                if refit_every is not None or initial_residuals:
                    raise ValueError(
                        "native quantiles require a frozen model without initial_residuals"
                    )
                if not isinstance(reconciler, Identity):
                    raise ValueError(
                        "native quantiles require Identity; quantiles cannot be reconciled"
                    )
            if initial_residuals:
                if not isinstance(fitted, FittedResiduals):
                    raise TypeError("model does not expose FittedResiduals")
                seed = _reconciled_residuals(fitted.residuals(), reconciler, int(origin), horizon)
        if isinstance(fitted, FittedQuantiles):
            prediction = fitted.predict_quantiles(window)
            base = np.asarray(prediction.point)
            levels = check_quantiles(prediction.levels)
            if intervals is None:
                intervals = np.empty(points.shape + (2,), dtype=np.float32)
                quantile_levels = levels
            if levels != quantile_levels:
                raise ValueError("native quantile levels changed between origins")
            if np.shape(prediction.interval) != (n_series, horizon, 2):
                raise ValueError("model native intervals must be [S, H, 2]")
            intervals[index] = prediction.interval
        else:
            base = np.asarray(fitted.predict(window))
        if base.shape != (n_series, horizon):
            raise ValueError(f"model returned shape {base.shape}, expected {(n_series, horizon)}")
        if not np.isfinite(base).all():
            raise ValueError(f"model returned nonfinite points at origin {origin}")
        points[index] = reconciler(base)
    return Forecasts(
        origins=origins,
        points=points,
        initial_residuals=seed,
        interval=intervals,
        quantile_levels=quantile_levels,
    )


def _reconciled_residuals(
    residuals: FitResiduals, reconciler: Reconciler, origin: int, horizon: int
) -> FitResiduals:
    """Apply the same linear reconciliation to each aligned OOB error vector."""
    values = residuals.values
    if values.shape[1:] != (reconciler.shape[1], horizon):
        raise ValueError("fit residual shape does not match the forecast series and horizon")
    if (residuals.origins + horizon > origin).any():
        raise ValueError("fit residual targets must be known at the first forecast origin")
    # Flatten time/step columns rather than adding another loop over origins.
    mapped = reconciler(values.transpose(1, 0, 2).reshape(reconciler.shape[1], -1))
    mapped = mapped.reshape(reconciler.shape[0], len(values), horizon).transpose(1, 0, 2)
    return FitResiduals(residuals.origins, mapped)


def _check_schedule(
    origins: np.ndarray, n_periods: int, horizon: int, refit_every: int | None
) -> np.ndarray:
    origins = np.asarray(origins)
    if origins.ndim != 1 or len(origins) == 0 or not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("origins must be a non-empty vector of period indexes")
    if (np.diff(origins) <= 0).any():
        raise ValueError("origins must be strictly increasing")
    if origins[0] < 0 or origins[-1] >= n_periods:
        raise ValueError(f"origins must be in [0, {n_periods - 1}]")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if refit_every is not None and refit_every < 1:
        raise ValueError(f"refit_every must be at least 1, got {refit_every}")
    return origins.astype(np.int64)


def _check_covariate(
    name: str, covariate: Covariate, n_bottom: int, last: int, horizon: int
) -> None:
    rows, length = covariate.values.shape
    if rows not in (1, n_bottom):
        raise ValueError(f"covariate {name!r} must have one row or one row per bottom series")
    needed = last + 1 + (horizon if covariate.known_ahead else 0)
    if length != 1 and length < needed:
        raise ValueError(f"covariate {name!r} has {length} periods, needs {needed}")


def _series_rows(
    values: np.ndarray, hierarchy: Hierarchy, n_series: int, aggregate: str
) -> np.ndarray:
    """Read-only rows for the first `n_series` nodes. A single row is shared by all."""
    if len(values) > 1 and n_series > hierarchy.n_bottom:
        nodes = hierarchy.summing[:n_series]
        values = nodes @ values
        if aggregate == "mean":
            values = values / nodes.sum(axis=1)[:, None]
        values = values.astype(np.float32)
    view = values.view()
    view.flags.writeable = False
    return view


def _reach(covariate: Covariate, horizon: int) -> int | None:
    """Periods visible after the origin. None for a static covariate."""
    if covariate.values.shape[1] == 1:
        return None
    return horizon if covariate.known_ahead else 0


def _visible(values: np.ndarray, seen: int, reach: int | None) -> np.ndarray:
    return values if reach is None else values[:, : seen + reach]
