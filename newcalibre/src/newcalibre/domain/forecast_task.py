"""Define forecast tasks with construction-time temporal hygiene."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd

from newcalibre.domain._calendar import CalendarFrequencyError, calendar_offset

HISTORY_TIMESTAMP: Final = "timestamp"


class ForecastTaskError(ValueError):
    """Report an invalid forecast task at its construction boundary."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ForecastTask:
    """Carry one fit-and-predict request across the adapter seam.

    Construction snapshots mutable inputs and public access returns defensive
    copies, so accepted history cannot later be changed to violate TSK-2.
    """

    _history: pd.DataFrame = field(repr=False)
    _horizon: int
    _origin: pd.Timestamp
    _calendar_frequency: str
    _model_config: dict[str, object] = field(repr=False)

    def __init__(
        self,
        *,
        history: pd.DataFrame,
        horizon: int,
        origin: pd.Timestamp,
        calendar_frequency: str,
        model_config: Mapping[str, object],
    ) -> None:
        if not isinstance(history, pd.DataFrame):
            raise ForecastTaskError("history must be a pandas DataFrame")
        if history.columns.has_duplicates:
            raise ForecastTaskError("history has duplicate column labels")
        if HISTORY_TIMESTAMP not in history.columns:
            raise ForecastTaskError(f"history requires column {HISTORY_TIMESTAMP!r}")
        timestamp_dtype = history[HISTORY_TIMESTAMP].dtype
        if not isinstance(timestamp_dtype, np.dtype) or timestamp_dtype.kind != "M":
            raise ForecastTaskError(
                "history timestamp must have a timezone-naive numpy datetime64 dtype"
            )
        if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
            raise ForecastTaskError("horizon must be a positive integer")
        if not isinstance(origin, pd.Timestamp) or pd.isna(origin):
            raise ForecastTaskError("origin must be a non-missing pandas Timestamp")
        if origin.tz is not None:
            raise ForecastTaskError("origin must be timezone-naive for the provisional calendar")
        if not isinstance(model_config, Mapping):
            raise ForecastTaskError("model configuration must be a mapping")

        try:
            offset = calendar_offset(calendar_frequency)
        except CalendarFrequencyError as error:
            raise ForecastTaskError(str(error)) from error
        timestamps = history[HISTORY_TIMESTAMP]
        if not timestamps.lt(origin).all():
            raise ForecastTaskError("every history timestamp must be strictly before origin")

        object.__setattr__(self, "_history", history.copy(deep=True))
        object.__setattr__(self, "_horizon", int(horizon))
        object.__setattr__(self, "_origin", origin)
        object.__setattr__(self, "_calendar_frequency", offset.freqstr)
        object.__setattr__(self, "_model_config", deepcopy(dict(model_config)))

    @property
    def history(self) -> pd.DataFrame:
        """Return a defensive copy of the validated pre-origin history."""
        return self._history.copy(deep=True)

    @property
    def horizon(self) -> int:
        """Return the positive forecast horizon."""
        return self._horizon

    @property
    def origin(self) -> pd.Timestamp:
        """Return the timezone-naive forecast origin."""
        return self._origin

    @property
    def calendar_frequency(self) -> str:
        """Return the normalized provisional pandas frequency."""
        return self._calendar_frequency

    @property
    def model_config(self) -> Mapping[str, object]:
        """Return a defensive copy of the model configuration."""
        return deepcopy(self._model_config)
