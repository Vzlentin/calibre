"""Provide the provisional anchored-pandas calendar representation."""

from __future__ import annotations

import re

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

_BARE_WEEKLY_FREQUENCY = re.compile(r"^(?:[+-]?[0-9]+)?W$", re.IGNORECASE)


class CalendarFrequencyError(ValueError):
    """Report an invalid provisional pandas calendar frequency."""


def calendar_offset(frequency: str) -> BaseOffset:
    """Parse an explicit pandas frequency without accepting a silent weekly anchor."""
    if not isinstance(frequency, str) or not frequency:
        raise CalendarFrequencyError("calendar frequency must be a non-empty string")
    if _BARE_WEEKLY_FREQUENCY.fullmatch(frequency):
        raise CalendarFrequencyError(
            "weekly calendar frequency requires an explicit anchor such as 'W-MON'"
        )

    try:
        offset = to_offset(frequency)
    except (TypeError, ValueError) as error:
        raise CalendarFrequencyError(f"invalid pandas calendar frequency: {frequency!r}") from error
    if offset is None:
        raise CalendarFrequencyError(f"invalid pandas calendar frequency: {frequency!r}")
    if offset.n <= 0:
        raise CalendarFrequencyError("calendar frequency must advance by a positive period")
    return offset


def advance_timestamp(origin: pd.Timestamp, periods: int, offset: BaseOffset) -> pd.Timestamp:
    """Advance an origin by a non-negative number of periods."""
    if periods == 0:
        # Anchored pandas offsets roll an off-grid timestamp even when multiplied by zero.
        # FRA-1 instead defines horizon step 1 as the origin period itself.
        return origin
    return origin + periods * offset
