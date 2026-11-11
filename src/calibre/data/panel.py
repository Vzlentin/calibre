"""Panel: the ragged grouped array of the series on one calendar, validated once."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from utilsforecast import processing as ufp


def _offset(freq: object) -> pd.offsets.BaseOffset:
    if not isinstance(freq, str):
        raise TypeError(f"freq must be a str, got {type(freq).__name__}")
    try:
        return pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError) as error:
        raise ValueError(f"freq {freq!r} is not a pandas offset alias") from error


def _array(value: object, name: str, kinds: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if value.dtype.kind not in kinds:
        raise TypeError(f"{name} has dtype {value.dtype}, which is not allowed")
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {value.shape}")
    return value


def _labels(value: object, name: str) -> np.ndarray:
    array = _array(value, name, "UO")
    if array.dtype.kind == "O" and not all(isinstance(item, str) for item in array):
        raise TypeError(f"{name} must hold str values")
    return np.array(array, dtype=str)


@dataclass(frozen=True)
class Panel:
    """Series values on one complete calendar: series s holds `data[indptr[s]:indptr[s + 1]]`.

    The values of series s start at `periods[starts[s]]`, one per period with no gap.
    `censored`, when present, flags each value that is a lower limit of demand.
    """

    series: np.ndarray
    periods: pd.DatetimeIndex
    freq: str
    data: np.ndarray
    indptr: np.ndarray
    starts: np.ndarray
    censored: np.ndarray | None = None

    def __post_init__(self) -> None:
        series = _labels(self.series, "series")
        if not isinstance(self.periods, pd.DatetimeIndex):
            raise TypeError("periods must be a pandas DatetimeIndex")
        offset = _offset(self.freq)
        data = np.array(_array(self.data, "data", "fiu"), dtype=np.float32)
        indptr = np.array(_array(self.indptr, "indptr", "iu"), dtype=np.int64)
        starts = np.array(_array(self.starts, "starts", "iu"), dtype=np.int64)
        censored = None
        if self.censored is not None:
            censored = np.array(_array(self.censored, "censored", "b"), dtype=bool)
        periods = self.periods.copy(deep=True)
        periods.freq = None
        n_series, n_periods = len(series), len(periods)
        if n_series == 0:
            raise ValueError("panel needs at least one series")
        if len(np.unique(series)) != n_series:
            raise ValueError("panel series ids are not unique")
        if n_periods == 0:
            raise ValueError("panel needs at least one period")
        expected = pd.date_range(periods[0], periods[-1], freq=offset)
        if not periods.equals(expected):
            raise ValueError(f"panel periods are not a complete calendar at {self.freq}")
        if len(indptr) != n_series + 1 or indptr[0] != 0 or indptr[-1] != len(data):
            raise ValueError("indptr must have S + 1 values, from 0 to the length of data")
        lengths = np.diff(indptr)
        if (lengths <= 0).any():
            raise ValueError("indptr must increase: each series needs at least one value")
        if len(starts) != n_series:
            raise ValueError("starts must have one value per series")
        lasts = starts + lengths - 1
        if (starts < 0).any() or (lasts >= n_periods).any():
            raise ValueError("a series runs outside the calendar")
        if not np.isfinite(data).all():
            raise ValueError("panel values must be finite")
        if censored is not None and len(censored) != len(data):
            raise ValueError("censored must have the length of data")
        for array in (series, data, indptr, starts, censored):
            if array is not None:
                array.flags.writeable = False
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "censored", censored)

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            object.__setattr__(self, name, value)

    def take(self, index: np.ndarray) -> "Panel":
        """The series at the strictly increasing positions `index`, on the same calendar."""
        if not isinstance(index, np.ndarray):
            raise TypeError("index must be a numpy array")
        if index.dtype.kind not in "iu":
            raise ValueError("index must hold integers")
        if index.ndim != 1 or len(index) == 0:
            raise ValueError("index must be a non-empty 1-d array")
        if (index < 0).any() or (index >= len(self.series)).any():
            raise ValueError("index must hold positions of series of the panel")
        if (np.diff(index) <= 0).any():
            raise ValueError("index must be strictly increasing")
        first = self.indptr[index]
        lengths = self.indptr[index + 1] - first
        indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        rows = np.repeat(first - indptr[:-1], lengths) + np.arange(indptr[-1])
        return Panel(
            series=self.series[index],
            periods=self.periods,
            freq=self.freq,
            data=self.data[rows],
            indptr=indptr,
            starts=self.starts[index],
            censored=None if self.censored is None else self.censored[rows],
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        freq: str,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        censored_col: str | None = None,
    ) -> "Panel":
        """Read a long frame: one row per series and period. Sorts the series by id."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        offset = _offset(freq)
        for name in (id_col, time_col, target_col):
            if not isinstance(name, str):
                raise TypeError("column names must be str")
        if censored_col is not None and not isinstance(censored_col, str):
            raise TypeError("censored_col must be a str or None")
        columns = [id_col, time_col, target_col]
        if censored_col is not None:
            columns.append(censored_col)
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"frame has no column {missing}")
        if len(frame) == 0:
            raise ValueError("frame has no rows")
        if not pd.api.types.is_datetime64_any_dtype(frame[time_col]):
            raise ValueError(f"column {time_col!r} must hold datetimes")
        if not pd.api.types.is_numeric_dtype(frame[target_col]) or pd.api.types.is_bool_dtype(
            frame[target_col]
        ):
            raise ValueError(f"column {target_col!r} must hold numbers")
        if censored_col is not None and not pd.api.types.is_bool_dtype(frame[censored_col]):
            raise ValueError(f"column {censored_col!r} must be boolean")
        if frame[[id_col, time_col]].isna().to_numpy().any():
            raise ValueError("ids and times must not be missing")
        if frame.duplicated([id_col, time_col]).any():
            raise ValueError("frame has a duplicate id and time")
        values = frame[target_col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("frame values must be finite")
        long = frame[[id_col, time_col, target_col]]
        processed = ufp.process_df(long, id_col, time_col, target_col)
        order = processed.sort_idxs
        if order is None:
            order = np.arange(len(frame))
        times = pd.DatetimeIndex(frame[time_col].to_numpy()[order])
        periods = pd.date_range(times.min(), times.max(), freq=offset)
        position = periods.get_indexer(times)
        if (position < 0).any():
            raise ValueError(f"a time of the frame is off the calendar at {freq}")
        indptr = np.asarray(processed.indptr, dtype=np.int64)
        lengths = np.diff(indptr)
        first = position[indptr[:-1]]
        expected = (
            np.repeat(first, lengths) + np.arange(len(position)) - np.repeat(indptr[:-1], lengths)
        )
        if not np.array_equal(position, expected):
            raise ValueError("a series has a gap")
        censored = None
        if censored_col is not None:
            censored = frame[censored_col].to_numpy(dtype=bool)[order]
        return cls(
            series=np.asarray(processed.uids).astype(str),
            periods=periods,
            freq=freq,
            data=values[order],
            indptr=indptr,
            starts=first.astype(np.int64),
            censored=censored,
        )
