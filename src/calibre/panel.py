"""Panel: bottom series values on a complete calendar, validated once."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    """Bottom series values `[B, T]` as read-only float32. Downstream code trusts them."""

    series: np.ndarray
    periods: pd.DatetimeIndex
    values: np.ndarray
    freq: str

    def __post_init__(self) -> None:
        series = np.array(self.series, dtype=str)
        periods = pd.DatetimeIndex(self.periods)
        values = np.array(self.values, dtype=np.float32)
        if len(series) == 0 or len(periods) == 0:
            raise ValueError("panel needs at least one series and one period")
        if values.shape != (len(series), len(periods)):
            raise ValueError(f"panel values shape {values.shape} != {(len(series), len(periods))}")
        if len(np.unique(series)) != len(series):
            raise ValueError("panel series ids are not unique")
        if not np.isfinite(values).all():
            raise ValueError("panel values must be finite (no NaN or infinity)")
        if not periods.equals(pd.date_range(periods[0], periods[-1], freq=self.freq)):
            raise ValueError(f"panel periods are not a complete {self.freq} range")
        series.flags.writeable = False
        values.flags.writeable = False
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_long(cls, frame: pd.DataFrame, freq: str) -> "Panel":
        """Pivot a long frame with columns series, period, value."""
        wide = frame.pivot(index="series", columns="period", values="value")
        wide = wide.sort_index(axis=0).sort_index(axis=1)
        if wide.isna().to_numpy().any():
            raise ValueError("panel is not rectangular: some (series, period) cells are missing")
        return cls(wide.index.to_numpy(), pd.DatetimeIndex(wide.columns), wide.to_numpy(), freq)
