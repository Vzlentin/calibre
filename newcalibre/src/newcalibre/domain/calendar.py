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

    An unbound declaration is useful while ingesting a panel; the panel binds
    every frequency to its earliest timestamp. Every later observation and
    task origin is checked against that same stride and clock phase.
    """

    _frequency: str
    _offset: BaseOffset = field(repr=False, compare=False)
    _fixed_nanos: int | None = field(repr=False, compare=False)
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
        try:
            fixed_nanos: int | None = offset.nanos
        except ValueError:
            fixed_nanos = None
        if phase is not None:
            self._require_timestamp(phase, name="calendar phase")
            if not offset.is_on_offset(phase):
                raise CalendarError(
                    f"calendar phase {phase!s} does not lie on calendar {offset.freqstr!r}"
                )
        object.__setattr__(self, "_frequency", offset.freqstr)
        object.__setattr__(self, "_offset", offset)
        object.__setattr__(self, "_fixed_nanos", fixed_nanos)
        object.__setattr__(self, "_phase", phase)

    @property
    def frequency(self) -> str:
        """Return the normalized dataset declaration."""
        return self._frequency

    @property
    def phase(self) -> pd.Timestamp | None:
        """Return the retained dataset phase, or ``None`` while unbound."""
        return self._phase

    def bind(self, phase: pd.Timestamp) -> Calendar:
        """Return this calendar bound to a dataset phase timestamp."""
        self._require_timestamp(phase, name="calendar phase")
        if self._phase is not None:
            if phase != self._phase:
                raise CalendarError("calendar phase is already bound to a different timestamp")
            return self
        return Calendar(self._frequency, phase=phase)

    def shares_grid_with(self, other: Calendar) -> bool:
        """Return whether another bound calendar names the same timestamp grid."""
        if not isinstance(other, Calendar) or self._frequency != other._frequency:
            return False
        if self._phase is None or other._phase is None:
            return False
        return self.contains(other._phase) and other.contains(self._phase)

    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Return whether a timezone-naive timestamp lies on this calendar."""
        self._require_timestamp(timestamp)
        if self._phase is None:
            raise CalendarError("calendar must be bound to a dataset phase")
        return self._index_of(timestamp) is not None

    def _index_of(self, timestamp: pd.Timestamp) -> int | None:
        phase = self._phase
        if phase is None:
            raise CalendarError("calendar must be bound to a dataset phase")
        if self._fixed_nanos is not None:
            delta = self._epoch_nanoseconds(timestamp) - self._epoch_nanoseconds(phase)
            quotient, remainder = divmod(delta, self._fixed_nanos)
            return quotient if remainder == 0 else None
        return self._non_fixed_index(timestamp)

    def _non_fixed_index(self, timestamp: pd.Timestamp) -> int | None:
        phase = self._phase
        if phase is None:
            raise CalendarError("calendar must be bound to a dataset phase")
        if timestamp == phase:
            return 0
        forward = timestamp > phase

        def relation(periods: int) -> int:
            signed_periods = periods if forward else -periods
            try:
                candidate = phase + signed_periods * self._offset
            except (OverflowError, ValueError):
                return 1
            if (forward and candidate <= phase) or (not forward and candidate >= phase):
                raise CalendarError("calendar offset failed to advance monotonically")
            if candidate == timestamp:
                return 0
            before_target = candidate < timestamp if forward else candidate > timestamp
            return -1 if before_target else 1

        low = 0
        high = 1
        for _ in range(128):
            position = relation(high)
            if position == 0:
                return high if forward else -high
            if position > 0:
                break
            low = high
            high *= 2
        else:
            raise CalendarError("calendar membership search exceeded timestamp bounds")

        while low + 1 < high:
            middle = (low + high) // 2
            position = relation(middle)
            if position == 0:
                return middle if forward else -middle
            if position < 0:
                low = middle
            else:
                high = middle
        return None

    def _at_index(self, index: int) -> pd.Timestamp:
        phase = self._phase
        if phase is None:
            raise CalendarError("calendar must be bound to a dataset phase")
        if index == 0:
            return phase
        try:
            return phase + index * self._offset
        except (OverflowError, ValueError) as error:
            raise CalendarError("calendar operation exceeds timestamp bounds") from error

    @staticmethod
    def _epoch_nanoseconds(timestamp: pd.Timestamp) -> int:
        factors = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
        try:
            factor = factors[timestamp.unit]
        except KeyError as error:
            raise CalendarError("timestamp resolution must be s, ms, us, or ns") from error
        return int(timestamp.asm8.astype("int64")) * factor

    def require_member(self, timestamp: pd.Timestamp, *, name: str = "timestamp") -> None:
        """Reject a timestamp that is not a member of this calendar."""
        self._require_timestamp(timestamp, name=name)
        if not self.contains(timestamp):
            raise CalendarError(
                f"{name} {timestamp!s} does not lie on calendar {self._frequency!r}"
            )

    def advance(self, timestamp: pd.Timestamp, periods: int) -> pd.Timestamp:
        """Advance a calendar member by a non-negative whole number of periods."""
        self._require_timestamp(timestamp)
        index = self._index_of(timestamp)
        if index is None:
            raise CalendarError(
                f"timestamp {timestamp!s} does not lie on calendar {self._frequency!r}"
            )
        if not isinstance(periods, Integral) or isinstance(periods, bool) or periods < 0:
            raise CalendarError("calendar periods must be a non-negative integer")
        if periods == 0:
            return timestamp
        return self._at_index(index + int(periods))

    def retreat(self, timestamp: pd.Timestamp, periods: int) -> pd.Timestamp:
        """Retreat a calendar member by a non-negative whole number of periods."""
        self._require_timestamp(timestamp)
        index = self._index_of(timestamp)
        if index is None:
            raise CalendarError(
                f"timestamp {timestamp!s} does not lie on calendar {self._frequency!r}"
            )
        if not isinstance(periods, Integral) or isinstance(periods, bool) or periods < 0:
            raise CalendarError("calendar periods must be a non-negative integer")
        if periods == 0:
            return timestamp
        return self._at_index(index - int(periods))

    @staticmethod
    def _require_timestamp(timestamp: pd.Timestamp, *, name: str = "timestamp") -> None:
        if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
            raise CalendarError(f"{name} must be a non-missing pandas Timestamp")
        if timestamp.tz is not None:
            raise CalendarError(f"{name} must be timezone-naive")
