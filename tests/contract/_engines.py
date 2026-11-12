"""StatsForecast and MLForecast objects built from the same specs that `references.py` reads,
and a local Forecaster of the tests."""

from collections.abc import Sequence

import mlforecast.lag_transforms
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import statsforecast.models
from mlforecast import MLForecast
from statsforecast import StatsForecast

from calibre import Forecaster, Panel, Window


def model(spec):
    if spec is None:
        return None
    name, kwargs = spec
    return getattr(statsforecast.models, name)(**kwargs)


def engine(models, freq: str, fallback=None, n_jobs: int = 1) -> StatsForecast:
    return StatsForecast(
        models=[model(spec) for spec in models],
        freq=freq,
        fallback_model=model(fallback),
        n_jobs=n_jobs,
    )


SKLEARN = {"ensemble": sklearn.ensemble, "linear_model": sklearn.linear_model}


def estimator(path: str, kwargs: dict):
    """A scikit-learn estimator from its path, such as `linear_model.LinearRegression`."""
    module, name = path.split(".")
    return getattr(SKLEARN[module], name)(**kwargs)


def ml_engine(spec: dict) -> MLForecast:
    """An MLForecast from a JSON spec: `models` as [alias, path, kwargs] triples, `freq`, and
    optional `lags`, `lag_transforms` as {lag: [[name, kwargs]]}, and `date_features`."""
    transforms = {
        int(lag): [getattr(mlforecast.lag_transforms, name)(**kw) for name, kw in specs]
        for lag, specs in spec.get("lag_transforms", {}).items()
    }
    return MLForecast(
        models={alias: estimator(path, kwargs) for alias, path, kwargs in spec["models"]},
        freq=spec["freq"],
        lags=spec.get("lags"),
        lag_transforms=transforms or None,
        date_features=spec.get("date_features"),
    )


class LastMeanForecaster(Forecaster):
    """A local Forecaster: the float64 mean of each window, at every step."""

    local = True

    def __init__(self, alias: str = "last_mean") -> None:
        self._alias = alias

    @property
    def models(self) -> list[str]:
        return [self._alias]

    def forecast(self, panel: Panel, windows: Sequence[Window], horizon: int) -> np.ndarray:
        out = np.full((1, len(panel.series), horizon, len(windows)), np.nan, np.float32)
        for o, win in enumerate(windows):
            for s in range(len(panel.series)):
                lo, hi = int(win.start[s]), int(win.stop[s])
                if lo < hi:
                    out[0, s, :, o] = np.float32(panel.data[lo:hi].mean(dtype=np.float64))
        return out
