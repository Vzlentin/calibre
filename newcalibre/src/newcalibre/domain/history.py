"""Define opaque staged-history views, deltas, cursors, and cycle identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from numbers import Integral
from typing import Protocol

import pandas as pd

from newcalibre.domain.session import SessionIdentity


class HistoryError(ValueError):
    """Report invalid staged-history metadata or composition."""


class _HistoryStorage(Protocol):
    """Describe the private adapter materialization seam."""

    @property
    def identity(self) -> str: ...

    @property
    def series_keys(self) -> tuple[str, ...]: ...

    def materialize(
        self,
        *,
        series_start: int,
        series_stop: int,
        time_start: int,
        time_stop: int,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class HistoryCursor:
    """Identify one exclusive time bound over a contiguous staged series range."""

    panel_identity: str
    series_start: int
    series_stop: int
    time_bound: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.panel_identity, str)
            or len(self.panel_identity) != 64
            or self.panel_identity != self.panel_identity.lower()
            or any(character not in "0123456789abcdef" for character in self.panel_identity)
        ):
            raise HistoryError("history cursor panel identity must be a SHA-256 hex string")
        for name, value in (
            ("series_start", self.series_start),
            ("series_stop", self.series_stop),
            ("time_bound", self.time_bound),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise HistoryError(f"history cursor {name} must be an integer")
        if self.series_start < 0 or self.series_stop <= self.series_start:
            raise HistoryError("history cursor series range must be non-empty and non-negative")
        if self.time_bound < 0:
            raise HistoryError("history cursor time bound must be non-negative")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class HistoryView:
    """Expose an immutable O(1) view over strictly pre-origin staged history."""

    _storage: _HistoryStorage = field(repr=False)
    _cursor: HistoryCursor
    _series_keys: tuple[str, ...]
    _identity: str

    @classmethod
    def _from_storage(
        cls,
        storage: _HistoryStorage,
        *,
        cursor: HistoryCursor,
    ) -> HistoryView:
        _require_storage_cursor(storage, cursor)
        series_keys = storage.series_keys[cursor.series_start : cursor.series_stop]
        instance = object.__new__(cls)
        object.__setattr__(instance, "_storage", storage)
        object.__setattr__(instance, "_cursor", cursor)
        object.__setattr__(instance, "_series_keys", series_keys)
        object.__setattr__(instance, "_identity", _view_identity(cursor))
        return instance

    @property
    def identity(self) -> str:
        """Return the canonical semantic identity of this staged slice."""
        return self._identity

    @property
    def cursor(self) -> HistoryCursor:
        """Return the exclusive cursor that closes this view."""
        return self._cursor

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return the canonical contiguous series enumeration."""
        return self._series_keys

    def materialize(self) -> pd.DataFrame:
        """Materialize an isolated adapter-owned frame for this view."""
        return self._storage.materialize(
            series_start=self._cursor.series_start,
            series_stop=self._cursor.series_stop,
            time_start=0,
            time_stop=self._cursor.time_bound,
        )

    def delta_since(self, cursor: HistoryCursor) -> HistoryDelta:
        """Return the contiguous newly visible interval after ``cursor``."""
        _require_storage_cursor(self._storage, cursor)
        _require_same_range(cursor, self._cursor)
        if cursor.time_bound > self._cursor.time_bound:
            raise HistoryError("history cursor is newer than the requested view")
        return HistoryDelta._from_storage(
            self._storage,
            start_cursor=cursor,
            end_cursor=self._cursor,
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class HistoryDelta:
    """Expose one immutable contiguous advancement between history cursors."""

    _storage: _HistoryStorage = field(repr=False)
    _start_cursor: HistoryCursor
    _end_cursor: HistoryCursor
    _identity: str

    @classmethod
    def _from_storage(
        cls,
        storage: _HistoryStorage,
        *,
        start_cursor: HistoryCursor,
        end_cursor: HistoryCursor,
    ) -> HistoryDelta:
        _require_storage_cursor(storage, start_cursor)
        _require_storage_cursor(storage, end_cursor)
        _require_same_range(start_cursor, end_cursor)
        if start_cursor.time_bound > end_cursor.time_bound:
            raise HistoryError("history delta cursors must advance monotonically")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_storage", storage)
        object.__setattr__(instance, "_start_cursor", start_cursor)
        object.__setattr__(instance, "_end_cursor", end_cursor)
        digest = hashlib.sha256()
        digest.update(b"newcalibre.history-delta/v1")
        digest.update(_cursor_bytes(start_cursor))
        digest.update(_cursor_bytes(end_cursor))
        object.__setattr__(instance, "_identity", digest.hexdigest())
        return instance

    @property
    def identity(self) -> str:
        """Return the canonical semantic identity of this delta."""
        return self._identity

    @property
    def start_cursor(self) -> HistoryCursor:
        """Return the inclusive starting cursor."""
        return self._start_cursor

    @property
    def end_cursor(self) -> HistoryCursor:
        """Return the exclusive ending cursor."""
        return self._end_cursor

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return the canonical contiguous series enumeration."""
        return self._storage.series_keys[
            self._end_cursor.series_start : self._end_cursor.series_stop
        ]

    def materialize(self) -> pd.DataFrame:
        """Materialize an isolated adapter-owned frame for this delta."""
        return self._storage.materialize(
            series_start=self._end_cursor.series_start,
            series_stop=self._end_cursor.series_stop,
            time_start=self._start_cursor.time_bound,
            time_stop=self._end_cursor.time_bound,
        )


@dataclass(frozen=True, slots=True)
class CycleToken:
    """Bind an intermediate engine value to one unique opened store cycle."""

    session: SessionIdentity
    origin: pd.Timestamp
    revision: int
    attempt: int

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("cycle token session must be a SessionIdentity")
        if not isinstance(self.origin, pd.Timestamp) or pd.isna(self.origin):
            raise TypeError("cycle token origin must be a non-missing pandas Timestamp")
        if self.origin.tz is not None:
            raise HistoryError("cycle token origin must be timezone-naive")
        if (
            not isinstance(self.revision, Integral)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise HistoryError("cycle token revision must be a positive integer")
        if (
            not isinstance(self.attempt, Integral)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise HistoryError("cycle token attempt must be a positive integer")
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "attempt", int(self.attempt))


def _require_storage_cursor(storage: _HistoryStorage, cursor: HistoryCursor) -> None:
    if not isinstance(cursor, HistoryCursor):
        raise TypeError("history cursor must be a HistoryCursor")
    if cursor.panel_identity != storage.identity:
        raise HistoryError("history cursor belongs to another staged panel")
    if cursor.series_stop > len(storage.series_keys):
        raise HistoryError("history cursor series range exceeds the staged panel")


def _require_same_range(first: HistoryCursor, second: HistoryCursor) -> None:
    if (
        first.panel_identity != second.panel_identity
        or first.series_start != second.series_start
        or first.series_stop != second.series_stop
    ):
        raise HistoryError("history cursors do not identify the same staged series range")


def _cursor_bytes(cursor: HistoryCursor) -> bytes:
    return (
        f"{cursor.panel_identity}:{cursor.series_start}:{cursor.series_stop}:{cursor.time_bound}"
    ).encode()


def _view_identity(cursor: HistoryCursor) -> str:
    digest = hashlib.sha256()
    digest.update(b"newcalibre.history-view/v1")
    digest.update(_cursor_bytes(cursor))
    return digest.hexdigest()


__all__ = [
    "CycleToken",
    "HistoryCursor",
    "HistoryDelta",
    "HistoryError",
    "HistoryView",
]
