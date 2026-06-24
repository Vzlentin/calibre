"""Forecast task value types and their URI-backed serialized references."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID


@lru_cache(maxsize=1024)
def _read_parquet_cached(uri: str) -> pd.DataFrame:
    from calibre.core.io import read_parquet

    return read_parquet(uri)


@dataclass(frozen=True)
class ForecastTask:
    """One fit-and-predict unit: history plus the model config and horizon.

    Bundles the input history, the model configuration, and the forecast
    horizon for a single series (or global config) that an adapter fits.
    """

    history: pd.DataFrame
    horizon: int
    model_config: dict
    forecast_origin: pd.Timestamp | None = None
    future_x: pd.DataFrame | None = None
    task_group: str | None = None
    censoring: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if UNIQUE_ID not in self.history.columns:
            raise ValueError(
                f"ForecastTask.history must have a '{UNIQUE_ID}' column. "
                "Pass history=df where df includes unique_id."
            )

    @property
    def unique_id(self) -> str:
        return str(self.history[UNIQUE_ID].iloc[0])

    @property
    def model_name(self) -> str:
        return self.model_config.get("name", self.model_config["model"])

    def to_uri(self, base_uri: str) -> ForecastTaskRef:
        from calibre.core.io import join_uri, write_parquet

        history_uri = join_uri(base_uri, f"{self.unique_id}.parquet")
        write_parquet(self.history, history_uri)
        future_x_uri = None
        if self.future_x is not None:
            future_x_uri = join_uri(base_uri, f"{self.unique_id}.future_x.parquet")
            write_parquet(self.future_x, future_x_uri)
        censoring_uri = None
        if self.censoring is not None:
            censoring_uri = join_uri(base_uri, f"{self.unique_id}.censoring.parquet")
            write_parquet(self.censoring, censoring_uri)
        return ForecastTaskRef(
            unique_id=self.unique_id,
            model_config=dict(self.model_config),
            horizon=self.horizon,
            forecast_origin=self.forecast_origin,
            history_uri=history_uri,
            future_x_uri=future_x_uri,
            task_group=self.task_group,
            censoring_uri=censoring_uri,
        )


@dataclass(frozen=True)
class TaskGroups:
    """Pre-partitioned forecast tasks split by dispatch scope.

    Scope is resolved exactly once, in ``build_tasks``. The engine consumes
    this partition directly and never re-interprets ``get_scope``. ``local``
    holds one task per ``(unique_id, config)``; ``global_`` holds one task per
    global config (deduplicated across series).
    """

    local: list[ForecastTask] = field(default_factory=list)
    global_: list[ForecastTask] = field(default_factory=list)

    @property
    def tasks(self) -> list[ForecastTask]:
        """Flat view of every task, local first then global."""
        return [*self.local, *self.global_]

    def __len__(self) -> int:
        return len(self.local) + len(self.global_)

    def __iter__(self) -> Iterator[ForecastTask]:
        return iter(self.tasks)


@dataclass(frozen=True)
class ForecastTaskRef:
    """A lightweight, serializable handle to a :class:`ForecastTask`.

    Stores the task's history (and optional future regressors) as parquet URIs
    so tasks can cross process boundaries; :meth:`materialize` reloads them.
    """

    unique_id: str
    model_config: dict[str, Any]
    horizon: int
    forecast_origin: pd.Timestamp | None
    history_uri: str
    future_x_uri: str | None = None
    task_group: str | None = None
    censoring_uri: str | None = None

    def materialize(self) -> ForecastTask:
        history = _read_parquet_cached(self.history_uri).copy()
        future_x = (
            _read_parquet_cached(self.future_x_uri).copy()
            if self.future_x_uri is not None
            else None
        )
        censoring = (
            _read_parquet_cached(self.censoring_uri).copy()
            if self.censoring_uri is not None
            else None
        )
        return ForecastTask(
            history=history,
            horizon=self.horizon,
            model_config=dict(self.model_config),
            forecast_origin=self.forecast_origin,
            future_x=future_x,
            task_group=self.task_group,
            censoring=censoring,
        )


@dataclass(frozen=True)
class ChunkTaskRef:
    """One staged artifact pair for a bounded group of same-config local series.

    A chunk stages one panel-history parquet (the concatenated per-series
    histories) and, when any series carries exogenous features, one future_x
    parquet filtered to the chunk's uids. The shared ``horizon``/``model_config``
    apply to every series in the chunk; the worker iterates ``unique_ids`` and
    fits each series independently, so per-series local semantics are exact for
    every backend. The chunk identity is staging-local only — it never reaches
    forecast/ledger rows, which carry the real per-series uids.
    """

    unique_ids: tuple[str, ...]
    model_config: dict[str, Any]
    horizon: int
    history_uri: str
    future_x_uri: str | None = None
    task_group: str | None = None
    censoring_uri: str | None = None


def stage_local_chunk(
    tasks: list[ForecastTask],
    base_uri: str,
    *,
    horizon: int,
    model_config: dict[str, Any],
    task_group: str | None,
) -> ChunkTaskRef:
    """Stage one chunk: concat the per-series histories + filtered future_x/censoring.

    Every task in ``tasks`` shares ``horizon``/``model_config`` (they were
    grouped by resolved config upstream). The chunk's future_x and censoring
    panels are the union of the per-series frames, each filtered to the chunk's
    uids; mixed presence is harmless because the worker re-slices per uid (and
    passes ``None`` when a series has no rows). The M5 dataset carries no
    censoring, so the censoring panel is absent for that path.
    """
    from calibre.core.io import join_uri, write_parquet

    for task in tasks:
        n_uids = task.history[UNIQUE_ID].nunique()
        if n_uids > 1:
            raise ValueError(
                f"local task for {task.unique_id!r} carries {n_uids} series; "
                "local scope stages one series per task"
            )
    unique_ids = tuple(task.unique_id for task in tasks)
    history = pd.concat([task.history for task in tasks], ignore_index=True)
    history_uri = join_uri(base_uri, "history.parquet")
    write_parquet(history, history_uri)

    future_frames = [task.future_x for task in tasks if task.future_x is not None]
    future_x_uri: str | None = None
    if future_frames:
        future_x = pd.concat(future_frames, ignore_index=True).drop_duplicates([UNIQUE_ID, DS])
        # Filter on stringified identity: ``unique_ids`` are str (via
        # ForecastTask.unique_id) while the column may carry a non-string dtype.
        future_x = future_x[future_x[UNIQUE_ID].astype(str).isin(unique_ids)]
        future_x_uri = join_uri(base_uri, "future_x.parquet")
        write_parquet(future_x, future_x_uri)

    censoring_frames = [task.censoring for task in tasks if task.censoring is not None]
    censoring_uri: str | None = None
    if censoring_frames:
        censoring = pd.concat(censoring_frames, ignore_index=True).drop_duplicates([UNIQUE_ID, DS])
        censoring = censoring[censoring[UNIQUE_ID].astype(str).isin(unique_ids)]
        censoring_uri = join_uri(base_uri, "censoring.parquet")
        write_parquet(censoring, censoring_uri)

    return ChunkTaskRef(
        unique_ids=unique_ids,
        model_config=dict(model_config),
        horizon=horizon,
        history_uri=history_uri,
        future_x_uri=future_x_uri,
        task_group=task_group,
        censoring_uri=censoring_uri,
    )
