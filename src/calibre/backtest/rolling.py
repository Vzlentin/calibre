"""rolling_forecasts: the points of each engine at each origin, from its window only."""

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calibre.data.panel import Panel
from calibre.forecast.forecasts import Forecasts
from calibre.forecast.window import Forecaster, Window

if TYPE_CHECKING:
    from mlforecast import MLForecast
    from statsforecast import StatsForecast


_ENGINES = {("statsforecast", "StatsForecast"), ("mlforecast", "MLForecast")}


def _engine_kind(engine: object) -> tuple[str, str]:
    kind = type(engine)
    return kind.__module__.split(".")[0], kind.__name__


def _wrap(engine: object, freq: str) -> Forecaster:
    if isinstance(engine, Forecaster):
        return engine
    # The engine modules load only when their engine is passed, so `import calibre` stays light.
    kind = _engine_kind(engine)
    if kind == ("statsforecast", "StatsForecast"):
        module = importlib.import_module("calibre.forecast.engines.statsforecast")
        return module.StatsForecastEngine(engine, freq)
    if kind == ("mlforecast", "MLForecast"):
        module = importlib.import_module("calibre.forecast.engines.mlforecast")
        return module.MLForecastEngine(engine, freq)
    raise TypeError(f"unknown engine {type(engine).__name__}")


def _check_forecaster(engine: Forecaster) -> list[str]:
    if not isinstance(engine.local, bool):
        raise TypeError("Forecaster.local must be a bool")
    models = engine.models
    if isinstance(models, str) or not isinstance(models, Sequence):
        raise TypeError("Forecaster.models must be a sequence of str")
    names = list(models)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("Forecaster.models needs at least one non-empty str")
    if len(set(names)) != len(names):
        raise ValueError("two models have the same alias")
    return names


def rolling_forecasts(
    panel: Panel,
    engines: Sequence["StatsForecast | MLForecast | Forecaster"],
    origins: pd.DatetimeIndex,
    horizon: int,
    *,
    lookback: int | None = None,
) -> Forecasts:
    """Run each engine at each origin, on the values of each series through the origin."""
    if not isinstance(panel, Panel):
        raise TypeError("panel must be a Panel")
    if isinstance(engines, str) or not isinstance(engines, Sequence):
        raise TypeError("engines must be a sequence of engine objects")
    if not isinstance(origins, pd.DatetimeIndex):
        raise TypeError("origins must be a pandas DatetimeIndex")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError("horizon must be an int")
    if lookback is not None and (not isinstance(lookback, int) or isinstance(lookback, bool)):
        raise TypeError("lookback must be an int or None")
    if len(engines) == 0:
        raise ValueError("engines needs at least one engine")
    for engine in engines:
        if not isinstance(engine, Forecaster) and _engine_kind(engine) not in _ENGINES:
            raise TypeError(
                "an engine must be a StatsForecast, MLForecast or Forecaster object, "
                f"got {type(engine).__name__}"
            )
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if lookback is not None and lookback < 1:
        raise ValueError("lookback must be at least 1")
    if len(origins) == 0:
        raise ValueError("origins needs at least one origin")
    if origins.hasnans or not (origins[1:] > origins[:-1]).all():
        raise ValueError("origins must be strictly increasing")
    position = panel.periods.get_indexer(origins)
    if (position < 0).any():
        raise ValueError("each origin must be a period of the panel")
    forecasters = [_wrap(engine, panel.freq) for engine in engines]
    names: list[str] = []
    for forecaster in forecasters:
        names.extend(_check_forecaster(forecaster))
    if len(set(names)) != len(names):
        raise ValueError("two models have the same alias")
    windows = [Window.at(panel, int(index), lookback) for index in position]
    blocks = []
    for forecaster in forecasters:
        block = forecaster.forecast(panel, windows, horizon)
        if not isinstance(block, np.ndarray) or block.dtype.kind not in "fiu":
            raise TypeError("Forecaster.forecast must return a numeric numpy array")
        expected = (len(forecaster.models), len(panel.series), horizon, len(windows))
        if block.shape != expected:
            raise ValueError(
                f"Forecaster.forecast returned shape {block.shape}, expected {expected}"
            )
        blocks.append(np.asarray(block, dtype=np.float32))
    points = np.concatenate(blocks, axis=0)
    return Forecasts(points, np.array(names, dtype=str), panel.series, origins)
