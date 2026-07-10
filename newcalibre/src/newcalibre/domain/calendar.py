"""Represent a dataset-declared pandas calendar as a domain value."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Integral

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

_BARE_WEEKLY_FREQUENCY = re.compile(r"^(?:[+-]?[0-9]+)?W$", re.IGNORECASE)


class CalendarError(ValueError):
    """Report an invalid calendar declaration or calendar operation."""


@dataclass(frozen=True, slots=True, init=False)
class Calendar:
    """Own one normalized, positive, explicitly anchored pandas frequency."""

    _frequency: str
    _offset: BaseOffset = field(repr=False, compare=False)

    def __init__(self, frequency: str) -> None:
        if not isinstance(frequency, str) or not frequency:
            raise CalendarError("calendar frequency must be a non-empty string")
        if _BARE_WEEKLY_FREQUENCY.fullmatch(frequency):
            raise CalendarError(
                "weekly calendar frequency requires an explicit anchor such as 'W-MON'"
            )
        try:
            offset = to_offset(frequency)
        except (TypeError, ValueError) as error:
            raise CalendarError(f"invalid pandas calendar frequency: {frequency!r}") from error
        if offset is None:
            raise CalendarError(f"invalid pandas calendar frequency: {frequency!r}")
        if offset.n <= 0:
            raise CalendarError("calendar frequency must advance by a positive period")
        object.__setattr__(self, "_frequency", offset.freqstr)
        object.__setattr__(self, "_offset", offset)

    @property
    def frequency(self) -> str:
        """Return the normalized dataset declaration."""
        return self._frequency

    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Return whether a timezone-naive timestamp lies on this calendar."""
        self._require_timestamp(timestamp)
        return self._offset.is_on_offset(timestamp)

    def require_member(self, timestamp: pd.Timestamp, *, name: str = "timestamp") -> None:
        """Reject a timestamp that is not a member of this calendar."""
        self._require_timestamp(timestamp, name=name)
        if not self._offset.is_on_offset(timestamp):
            raise CalendarError(
                f"{name} {timestamp!s} does not lie on calendar {self._frequency!r}"
            )

    def advance(self, timestamp: pd.Timestamp, periods: int) -> pd.Timestamp:
        """Advance a calendar member by a non-negative whole number of periods."""
        self.require_member(timestamp)
        if not isinstance(periods, Integral) or isinstance(periods, bool) or periods < 0:
            raise CalendarError("calendar periods must be a non-negative integer")
        if periods == 0:
            return timestamp
        return timestamp + int(periods) * self._offset

    def retreat(self, timestamp: pd.Timestamp, periods: int) -> pd.Timestamp:
        """Retreat a calendar member by a non-negative whole number of periods."""
        self.require_member(timestamp)
        if not isinstance(periods, Integral) or isinstance(periods, bool) or periods < 0:
            raise CalendarError("calendar periods must be a non-negative integer")
        if periods == 0:
            return timestamp
        return timestamp - int(periods) * self._offset

    @staticmethod
    def _require_timestamp(timestamp: pd.Timestamp, *, name: str = "timestamp") -> None:
        if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
            raise CalendarError(f"{name} must be a non-missing pandas Timestamp")
        if timestamp.tz is not None:
            raise CalendarError(f"{name} must be timezone-naive")
