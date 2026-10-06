"""Long frames for frame-based model libraries, built from a window.

Frames hold only the last periods a model reads: a frame of the full M5 history is
1 GB at each origin, the last 84 periods are 44 MB. Series ids are row numbers.
Columns are copies, because torch warns on the read-only arrays of a window.
"""

import numpy as np
import pandas as pd

from calibre.forecast.models.base import Window


def split_features(window: Window, names: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Sort covariates into static, historical, and known-ahead by their window shape."""
    seen = window.y.shape[1]
    static, historical, future = [], [], []
    for name in names:
        if name not in window.x:
            raise ValueError(f"feature {name!r} is not a covariate")
        columns = window.x[name].shape[1]
        kind = static if columns == 1 else future if columns > seen else historical
        kind.append(name)
    return static, historical, future


def freq(window: Window) -> str:
    if window.periods.freqstr is None:
        raise ValueError("window periods have no frequency")
    return window.periods.freqstr


def history(window: Window, periods: int | None, names: list[str]) -> pd.DataFrame:
    """unique_id, ds, y, and the named covariates over the last `periods` periods."""
    n_series, seen = window.y.shape
    start = 0 if periods is None else max(seen - periods, 0)
    length = seen - start
    columns = {
        "unique_id": np.repeat(np.arange(n_series), length),
        "ds": np.tile(window.periods[start:seen].to_numpy(), n_series),
        "y": window.y[:, start:].flatten(),
    }
    for name in names:
        values = window.x[name]
        values = values if values.shape[1] == 1 else values[:, start:seen]
        columns[name] = np.broadcast_to(values, (n_series, length)).flatten()
    return pd.DataFrame(columns)


def future(window: Window, names: list[str]) -> pd.DataFrame:
    """unique_id, ds, and the named known-ahead covariates over the forecast steps."""
    n_series, seen = window.y.shape
    columns = {
        "unique_id": np.repeat(np.arange(n_series), window.horizon),
        "ds": np.tile(window.periods[seen:].to_numpy(), n_series),
    }
    for name in names:
        values = window.x[name][:, seen : seen + window.horizon]
        columns[name] = np.broadcast_to(values, (n_series, window.horizon)).flatten()
    return pd.DataFrame(columns)


def static(window: Window, names: list[str]) -> pd.DataFrame:
    """unique_id and one column per static covariate."""
    n_series = window.y.shape[0]
    columns = {"unique_id": np.arange(n_series)}
    for name in names:
        columns[name] = np.broadcast_to(window.x[name], (n_series, 1)).flatten()
    return pd.DataFrame(columns)


def points(out: pd.DataFrame, window: Window) -> np.ndarray:
    """Forecast frame with one value column back to `[S, H]` float32, in series order.

    The column is found by elimination: neuralforecast renames model aliases.
    """
    shape = (window.y.shape[0], window.horizon)
    out = out.reset_index(drop=out.index.name is None)
    values = [column for column in out.columns if column not in ("unique_id", "ds")]
    if len(values) != 1 or len(out) != shape[0] * shape[1]:
        raise ValueError(f"model returned columns {values} and {len(out)} rows for shape {shape}")
    out = out.sort_values(["unique_id", "ds"])
    return out[values[0]].to_numpy(np.float32).reshape(shape)
