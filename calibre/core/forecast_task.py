from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec  # type: ignore[import-untyped]
import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID

if TYPE_CHECKING:
    from calibre.core.forecast_task_io import ForecastTaskRef


def _join_uri(base: str, filename: str) -> str:
    if "://" not in base:
        return str(Path(base) / filename)
    return f"{base.rstrip('/')}/{filename}"


def _ensure_parent(uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = str(Path(path).parent).replace("\\", "/")
    if parent and parent != ".":
        fs.mkdirs(parent, exist_ok=True)


@dataclass(frozen=True)
class ForecastTask:
    """A forecast task carrying one or more series.

    ``history`` always includes a ``unique_id`` column so that adapters can
    work uniformly with both single-series (one unique value) and multi-series
    (many unique values) history.
    """

    history: pd.DataFrame
    horizon: int
    model_config: dict
    forecast_origin: pd.Timestamp | None = None
    future_x: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if UNIQUE_ID not in self.history.columns:
            raise ValueError(
                f"ForecastTask.history must have a '{UNIQUE_ID}' column. "
                "Pass history=df where df includes unique_id."
            )

    @property
    def unique_id(self) -> str:
        """The unique_id of the first (or only) series in history."""
        return str(self.history[UNIQUE_ID].iloc[0])

    @property
    def model_name(self) -> str:
        return self.model_config.get("name", self.model_config["model"])

    def to_uri(self, base_uri: str) -> ForecastTaskRef:
        from calibre.core.forecast_task_io import ForecastTaskRef

        history_uri = _join_uri(base_uri, f"{self.unique_id}.parquet")
        _ensure_parent(history_uri)
        self.history.to_parquet(history_uri, index=False)
        future_x_uri = None
        if self.future_x is not None:
            future_x_uri = _join_uri(base_uri, f"{self.unique_id}.future_x.parquet")
            _ensure_parent(future_x_uri)
            self.future_x.to_parquet(future_x_uri, index=False)
        return ForecastTaskRef(
            unique_id=self.unique_id,
            model_config=dict(self.model_config),
            horizon=self.horizon,
            forecast_origin=self.forecast_origin,
            history_uri=history_uri,
            future_x_uri=future_x_uri,
        )
