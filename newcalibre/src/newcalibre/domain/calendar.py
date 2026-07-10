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
    """Own one normalized, positive, explicitly anchored pandas frequency.

    Frequencies multiplied by more than one retain a phase timestamp. An
    unbound declaration is useful while ingesting a panel; the panel binds it
    to its earliest timestamp and every later observation and task origin is
    checked against that same stride.
    """

    _frequency: str
    _offset: BaseOffset = field(repr=False, compare=False)
    _phase: pd.Timestamp | None

    def __init__(self, frequency: str, *, phase: pd.Timestamp | None = None) -> None:
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
        if phase is not None:
            self._require_timestamp(phase, name="calendar phase")
            if offset.n == 1:
                raise CalendarError("a single-period calendar does not carry a phase")
            if not offset.is_on_offset(phase):
                raise CalendarError(
                    f"calendar phase {phase!s} does not lie on calendar {offset.freqstr!r}"
                )
        object.__setattr__(self, "_frequency", offset.freqstr)
        object.__setattr__(self, "_offset", offset)
        object.__setattr__(self, "_phase", phase)

    @property
    def frequency(self) -> str:
        """Return the normalized dataset declaration."""
        return self._frequency

    @property
    def phase(self) -> pd.Timestamp | None:
        """Return the retained stride phase, or ``None`` when none is needed or bound."""
        return self._phase

    @property
    def requires_phase(self) -> bool:
        """Return whether this multiplied declaration needs a retained phase."""
        return self._offset.n > 1

    def bind(self, phase: pd.Timestamp) -> Calendar:
        """Return this calendar bound to a dataset phase timestamp."""
        self._require_timestamp(phase, name="calendar phase")
        if self._offset.n == 1:
            return self
        if self._phase is not None:
            if phase != self._phase:
                raise CalendarError("calendar phase is already bound to a different timestamp")
            return self
        return Calendar(self._frequency, phase=phase)

    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Return whether a timezone-naive timestamp lies on this calendar."""
        self._require_timestamp(timestamp)
        if not self._offset.is_on_offset(timestamp):
            return False
        if self._offset.n == 1:
            return True
        if self._phase is None:
            raise CalendarError("multiplied calendar must be bound to a dataset phase")

        start, end = sorted((self._phase, timestamp))
        grid = pd.date_range(start=start, end=end, freq=self._offset)
        return bool(len(grid) and grid[-1] == end)

    def require_member(self, timestamp: pd.Timestamp, *, name: str = "timestamp") -> None:
        """Reject a timestamp that is not a member of this calendar."""
        self._require_timestamp(timestamp, name=name)
        if not self.contains(timestamp):
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
