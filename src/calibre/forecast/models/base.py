"""Forecast contracts: what a model can see at one origin, and what it returns.

A `Forecaster` is fitted on a `Window` and returns a `Fitted` model. A `Fitted` model
predicts from a later `Window`. Each window ends at its origin, so a model cannot read
values after the origin, except covariates that are known ahead.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Covariate:
    """One input to the model, aligned on the panel calendar from its first period.

    The shape of `values` gives the kind: `[B, 1]` static, `[1, L]` calendar, `[B, L]`
    dynamic. `known_ahead` values are visible through the last forecast step. Other
    values are visible only through the origin. `aggregate` gives the value of an
    aggregate node from its bottom series. Calendar rows are the same for every node.
    """

    values: np.ndarray
    known_ahead: bool
    aggregate: Literal["sum", "mean"]

    @abstractmethod
    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"covariate values must be 2D, got shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("covariate values must be finite; fill them before the call")
        if self.aggregate not in ("sum", "mean"):
            raise ValueError(f"covariate aggregate must be 'sum' or 'mean', got {self.aggregate!r}")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Window:
    """Everything a model can see at one origin. All arrays are read-only.

    `y` is `[S, t]`, the history of the forecast series through the origin. Each array
    in `x` has one row or S rows and 1, t, or t + horizon columns. `periods` has
    t + horizon dates, so adapters can make date features for the forecast steps.
    """

    y: np.ndarray
    x: dict[str, np.ndarray]
    periods: pd.DatetimeIndex
    horizon: int


class Fitted(ABC):
    @abstractmethod
    def predict(self, window: Window) -> np.ndarray:
        """Return point forecasts `[S, horizon]` for the window."""


@dataclass(frozen=True)
class FitResiduals:
    """Out-of-sample signed errors [K, S, H] and their issuing origin indexes.

    For OOB errors, origins index the lag-row predictions, not the time the model
    was fitted. These errors are initialization data, not chronological forecasts.
    """

    origins: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins)
        values = np.array(self.values, dtype=np.float32)
        if (
            origins.ndim != 1
            or not np.issubdtype(origins.dtype, np.integer)
            or len(origins) == 0
            or (origins < 0).any()
            or (np.diff(origins) <= 0).any()
        ):
            raise ValueError("residual origins must be nonempty, increasing period indexes")
        if values.ndim != 3 or values.shape[0] != len(origins) or min(values.shape[1:]) < 1:
            raise ValueError("fit residuals must be [K, S, H] matching the origins")
        if np.isinf(values).any():
            raise ValueError("fit residuals must be finite or NaN")
        origins = origins.astype(np.int64, copy=True)
        origins.flags.writeable = False
        values.flags.writeable = False
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "values", values)


class FittedResiduals(Fitted):
    """A fitted model that exposes out-of-sample initialization errors."""

    @abstractmethod
    def residuals(self) -> FitResiduals:
        """Return signed errors known by the fitting origin."""


class Forecaster(ABC):
    @abstractmethod
    def fit(self, window: Window) -> Fitted:
        """Return a fitted model. Do not change `self`."""
