"""Dense loader output: causal calendar-gap handling and the panel of a [B, T] array."""

import numpy as np
import pandas as pd

from calibre.data import Panel


def forward_fill(wide: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Reindex, forward-fill and trim until all series have a past observation.

    Return the filled frame, original-target mask [B, T], and trimmed period count.
    Never back-fill or interpret an absent observation as zero.
    """
    wide = wide.sort_index().sort_index(axis=1)
    calendar = pd.date_range(wide.index[0], wide.index[-1], freq=freq)
    wide = wide.reindex(calendar)
    observed = wide.notna()
    wide = wide.ffill()
    valid = np.flatnonzero(wide.notna().to_numpy().all(axis=1))
    if not len(valid):
        raise ValueError("no period has a causal observation for every series")
    first = int(valid[0])
    return wide.iloc[first:], observed.iloc[first:].to_numpy(dtype=bool).T, first


def dense_panel(
    series: np.ndarray,
    periods: pd.DatetimeIndex,
    values: np.ndarray,
    freq: str,
    censored: np.ndarray | None = None,
) -> Panel:
    """The panel of `values` [B, T]: each series covers the whole calendar.

    `censored`, when given, is a [B, T] mask of the values that are lower limits of demand.
    """
    n_series, n_periods = values.shape
    return Panel(
        series=series,
        periods=periods,
        freq=freq,
        data=values.ravel(),
        indptr=np.arange(0, n_series * n_periods + 1, n_periods, dtype=np.int64),
        starts=np.zeros(n_series, dtype=np.int64),
        censored=None if censored is None else censored.ravel(),
    )
