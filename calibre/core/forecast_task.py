from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID


@lru_cache(maxsize=1024)
def _read_parquet_cached(uri: str) -> pd.DataFrame:
    from calibre.core.io import read_parquet

    return read_parquet(uri)


@dataclass(frozen=True)
class ForecastTask:
    history: pd.DataFrame
    horizon: int
    model_config: dict
    forecast_origin: pd.Timestamp | None = None
    future_x: pd.DataFrame | None = None
    task_group: str | None = None

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
        return ForecastTaskRef(
            unique_id=self.unique_id,
            model_config=dict(self.model_config),
            horizon=self.horizon,
            forecast_origin=self.forecast_origin,
            history_uri=history_uri,
            future_x_uri=future_x_uri,
            task_group=self.task_group,
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
    unique_id: str
    model_config: dict[str, Any]
    horizon: int
    forecast_origin: pd.Timestamp | None
    history_uri: str
    future_x_uri: str | None = None
    task_group: str | None = None

    def materialize(self) -> ForecastTask:
        history = _read_parquet_cached(self.history_uri).copy()
        future_x = (
            _read_parquet_cached(self.future_x_uri).copy()
            if self.future_x_uri is not None
            else None
        )
        return ForecastTask(
            history=history,
            horizon=self.horizon,
            model_config=dict(self.model_config),
            forecast_origin=self.forecast_origin,
            future_x=future_x,
            task_group=self.task_group,
        )
