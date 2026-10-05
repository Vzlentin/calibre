"""Seasonal naive: repeat the last season of every series over the horizon."""

import numpy as np

from calibre.forecast import Window


class SeasonalNaive:
    """A local model with nothing to fit. `fit` returns the model itself."""

    def __init__(self, season: int) -> None:
        if season < 1:
            raise ValueError(f"season must be at least 1, got {season}")
        self.season = season

    def fit(self, window: Window) -> "SeasonalNaive":
        return self

    def predict(self, window: Window) -> np.ndarray:
        if window.y.shape[1] < self.season:
            raise ValueError(f"history has {window.y.shape[1]} periods, season is {self.season}")
        last_season = window.y[:, -self.season :]
        repeats = -(-window.horizon // self.season)
        return np.tile(last_season, repeats)[:, : window.horizon]
