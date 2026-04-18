"""Time-decay sample weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID
from calibre.features._helpers import _sort_panel


def add_time_weights(
    df: pd.DataFrame,
    half_life_weeks: int = 52,
) -> pd.DataFrame:
    """Add exponential time-decayed observation weights per series.

    More recent observations receive higher weight during training. Uses
    exponential decay with configurable half-life (default: 52 weeks).
    """
    df = _sort_panel(df)

    max_ds = df.groupby(UNIQUE_ID, sort=False)[DS].transform("max")
    weeks_ago = (max_ds - df[DS]).dt.days / 7.0

    decay_rate = np.log(2) / half_life_weeks
    df["sample_weight"] = np.exp(-decay_rate * weeks_ago)

    return df
