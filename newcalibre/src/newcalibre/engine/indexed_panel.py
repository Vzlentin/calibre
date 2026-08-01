"""Stage one canonical panel and derive deterministic O(1) history tasks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd

from newcalibre.domain import (
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    ForecastTask,
    HistoryCursor,
    HistoryError,
    HistoryView,
    Panel,
    Scope,
    TargetSupport,
)
from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.domain.panel import _canonicalize_future_exogenous


class IndexedPanelError(ValueError):
    """Report invalid staged-panel task construction or cursor reuse."""


@dataclass(frozen=True, slots=True)
class _StagedHistory:
    """Keep the sole private series-major observation staging plane."""

    _frame: pd.DataFrame = field(repr=False, compare=False)
    _series_ordinals: np.ndarray = field(repr=False, compare=False)
    _time_offsets: np.ndarray = field(repr=False, compare=False)
    identity: str
    series_keys: tuple[str, ...]

    def materialize(
        self,
        *,
        series_start: int,
        series_stop: int,
        time_start: int,
        time_stop: int,
    ) -> pd.DataFrame:
        """Build one isolated adapter frame from opaque bounds."""
        selected = (
            (self._series_ordinals >= series_start)
            & (self._series_ordinals < series_stop)
            & (self._time_offsets >= time_start)
            & (self._time_offsets < time_stop)
        )
        return self._frame.loc[selected].reset_index(drop=True).copy(deep=True)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class IndexedPanel:
    """Own one immutable staged representation of a canonical panel."""

    _storage: _StagedHistory = field(repr=False)
    _calendar: Calendar
    _target_support: TargetSupport

    @classmethod
    def from_panel(cls, panel: Panel) -> IndexedPanel:
        """Stage canonical labels, values, presence, and calendar offsets once."""
        if not isinstance(panel, Panel):
            raise TypeError("indexed panel source must be a Panel")
        frame = panel.frame
        series_by_key = {key: ordinal for ordinal, key in enumerate(panel.series_keys)}
        series_ordinals = np.fromiter(
            (series_by_key[str(key)] for key in frame[SERIES_KEY]),
            dtype=np.int64,
            count=len(frame),
        )
        time_offsets_values: list[int] = []
        for raw_timestamp in frame[TIMESTAMP]:
            offset = panel.calendar._index_of(pd.Timestamp(raw_timestamp))
            if offset is None or offset < 0:
                raise IndexedPanelError("panel timestamp has no non-negative calendar offset")
            time_offsets_values.append(offset)
        time_offsets = np.asarray(time_offsets_values, dtype=np.int64)
        series_ordinals.flags.writeable = False
        time_offsets.flags.writeable = False
        identity = _panel_identity(
            frame,
            calendar=panel.calendar,
            target_support=panel.target_support,
        )
        storage = _StagedHistory(
            frame,
            series_ordinals,
            time_offsets,
            identity,
            panel.series_keys,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_storage", storage)
        object.__setattr__(instance, "_calendar", panel.calendar)
        object.__setattr__(instance, "_target_support", panel.target_support)
        return instance

    @property
    def identity(self) -> str:
        """Return the stable canonical staged-panel identity."""
        return self._storage.identity

    @property
    def calendar(self) -> Calendar:
        """Return the panel-wide bound calendar."""
        return self._calendar

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return canonical series keys in UTF-8 byte order."""
        return self._storage.series_keys

    @property
    def target_support(self) -> TargetSupport:
        """Return the declared mathematical target support."""
        return self._target_support

    def tasks(
        self,
        *,
        origin: pd.Timestamp,
        horizon: int,
        scope: Scope,
        model_config: Mapping[str, object],
        future_exogenous: pd.DataFrame | None = None,
        series_chunk_size: int | None = None,
        previous_cursors: Mapping[tuple[str, ...], HistoryCursor] | None = None,
    ) -> tuple[ForecastTask, ...]:
        """Build deterministic tasks with views and newly admissible deltas."""
        if not isinstance(scope, Scope):
            raise IndexedPanelError("scope must be Scope.LOCAL or Scope.GLOBAL")
        ForecastTask._require_horizon(horizon)
        try:
            self._calendar.require_member(origin, name="origin")
        except ValueError as error:
            raise IndexedPanelError(str(error)) from error
        if series_chunk_size is not None and (
            not isinstance(series_chunk_size, Integral)
            or isinstance(series_chunk_size, bool)
            or series_chunk_size < 1
        ):
            raise IndexedPanelError("series chunk size must be a positive integer")
        origin_bound = self._calendar._index_of(origin)
        if origin_bound is None or origin_bound < 0:
            raise IndexedPanelError("origin must not precede the staged panel phase")
        try:
            future = _canonicalize_future_exogenous(
                future_exogenous,
                calendar=self._calendar,
                origin=origin,
                horizon=int(horizon),
                series_keys=self.series_keys,
            )
        except ValueError as error:
            raise IndexedPanelError(str(error)) from error
        prior = {} if previous_cursors is None else dict(previous_cursors)
        partitions = self._partitions(scope=scope, series_chunk_size=series_chunk_size)
        expected_keys = {keys for _start, _stop, keys in partitions}
        unknown = set(prior) - expected_keys
        if unknown:
            raise IndexedPanelError("previous cursors name an unknown task series range")

        tasks: list[ForecastTask] = []
        for series_start, series_stop, series_keys in partitions:
            cursor = HistoryCursor(
                self.identity,
                series_start,
                series_stop,
                origin_bound,
            )
            start_cursor = prior.get(
                series_keys,
                HistoryCursor(self.identity, series_start, series_stop, 0),
            )
            try:
                view = HistoryView._from_storage(self._storage, cursor=cursor)
                delta = view.delta_since(start_cursor)
            except HistoryError as error:
                raise IndexedPanelError(str(error)) from error
            local_future = None
            if future is not None:
                local_future = future[future[SERIES_KEY].isin(series_keys)].reset_index(drop=True)
            tasks.append(
                ForecastTask._from_components(
                    history=view,
                    delta=delta,
                    cursor=cursor,
                    future_exogenous=local_future,
                    horizon=int(horizon),
                    origin=origin,
                    calendar=self._calendar,
                    model_config=model_config,
                    scope=scope,
                    series_keys=series_keys,
                )
            )
        return tuple(tasks)

    def _partitions(
        self,
        *,
        scope: Scope,
        series_chunk_size: int | None,
    ) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
        if scope is Scope.GLOBAL:
            return ((0, len(self.series_keys), self.series_keys),)
        chunk_size = 1 if series_chunk_size is None else int(series_chunk_size)
        return tuple(
            (
                start,
                min(start + chunk_size, len(self.series_keys)),
                self.series_keys[start : start + chunk_size],
            )
            for start in range(0, len(self.series_keys), chunk_size)
        )


def _panel_identity(
    frame: pd.DataFrame,
    *,
    calendar: Calendar,
    target_support: TargetSupport,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"newcalibre.indexed-panel/v1\0")
    digest.update(
        canonical_json_bytes(
            {
                "calendar_frequency": calendar.frequency,
                "calendar_phase": (
                    None
                    if calendar.phase is None
                    else {
                        "unit": calendar.phase.unit,
                        "value": int(calendar.phase.asm8.astype("int64")),
                    }
                ),
                "columns": [
                    {"dtype": str(frame[column].dtype), "name": column} for column in frame.columns
                ],
                "target_support": target_support.value,
            },
            path="indexed panel identity",
        )
    )
    digest.update(
        pd.util.hash_pandas_object(frame, index=False, categorize=True)
        .to_numpy(copy=False)
        .tobytes()
    )
    return digest.hexdigest()


__all__ = ["IndexedPanel", "IndexedPanelError"]
