"""Window: what one origin lets a model read, and the `Forecaster` contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from calibre.data.panel import Panel


@dataclass(frozen=True)
class Window:
    """Series s reads `data[start[s]:stop[s]]` at origin `origin`, a calendar index.

    A series whose first period is after the origin, or whose last period is before it,
    has `start == stop`: it reads nothing and gets no point.
    """

    origin: int
    start: np.ndarray  # [S] int64
    stop: np.ndarray  # [S] int64

    def __post_init__(self) -> None:
        if isinstance(self.origin, bool) or not isinstance(self.origin, (int, np.integer)):
            raise TypeError("origin must be an int")
        origin = int(self.origin)
        if origin < 0:
            raise ValueError("origin must be non-negative")
        for name in ("start", "stop"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu":
                raise TypeError(f"{name} must be an integer numpy array")
            if value.ndim != 1 or len(value) == 0:
                raise ValueError(f"{name} must be a non-empty 1-D array")
        start = np.array(self.start, dtype=np.int64)
        stop = np.array(self.stop, dtype=np.int64)
        if len(start) != len(stop):
            raise ValueError("start and stop must have the same length")
        if (start < 0).any() or (stop < start).any():
            raise ValueError("0 <= start <= stop must hold for each series")
        start.flags.writeable = False
        stop.flags.writeable = False
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def at(cls, panel: Panel, origin: int, lookback: int | None = None) -> "Window":
        """The values of each series through `origin`, at most the last `lookback`."""
        if not isinstance(panel, Panel):
            raise TypeError("panel must be a Panel")
        if isinstance(origin, bool) or not isinstance(origin, (int, np.integer)):
            raise TypeError("origin must be an int")
        origin = int(origin)
        if origin < 0 or origin >= len(panel.periods):
            raise ValueError("origin must be a period index of the panel")
        if lookback is not None:
            if isinstance(lookback, bool) or not isinstance(lookback, int):
                raise TypeError("lookback must be an int or None")
            if lookback < 1:
                raise ValueError("lookback must be at least 1")
        lengths = np.diff(panel.indptr)
        seen = origin - panel.starts + 1
        active = (seen >= 1) & (seen <= lengths)
        stop = np.where(active, panel.indptr[:-1] + seen, panel.indptr[:-1])
        size = seen if lookback is None else np.minimum(seen, lookback)
        start = np.where(active, stop - size, stop)
        return cls(origin, start.astype(np.int64), stop.astype(np.int64))


class Forecaster(ABC):
    """Contract of a model engine passed to `rolling_forecasts`.

    `local` is True when the forecast of each series reads only that series' own values,
    and False when the engine fits or predicts across series at once.
    """

    local: bool = True

    @property
    @abstractmethod
    def models(self) -> Sequence[str]:
        """The model names of this engine, at least one."""

    @abstractmethod
    def forecast(self, panel: Panel, windows: Sequence[Window], horizon: int) -> np.ndarray:
        """Points `[M, S, H, O]` float32 of `models` at each window."""
