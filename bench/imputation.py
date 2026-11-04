"""Causal calendar-gap handling for Beijing, Bike and Favorita."""

import numpy as np
import pandas as pd


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
    valid = np.flatnonzero(wide.notna().all(axis=1).to_numpy())
    if not len(valid):
        raise ValueError("no period has a causal observation for every series")
    first = int(valid[0])
    return wide.iloc[first:], observed.iloc[first:].to_numpy(dtype=bool).T, first
