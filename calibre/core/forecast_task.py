from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID

if TYPE_CHECKING:
    from calibre.core.forecast_task_io import ForecastTaskRef


@dataclass(frozen=True)
class ForecastTask:
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
        return str(self.history[UNIQUE_ID].iloc[0])

    @property
    def model_name(self) -> str:
        return self.model_config.get("name", self.model_config["model"])

    def to_uri(self, base_uri: str) -> ForecastTaskRef:
        from calibre.core.forecast_task_io import ForecastTaskRef
        from calibre.execution.io import ensure_parent_dir, join_uri

        history_uri = join_uri(base_uri, f"{self.unique_id}.parquet")
        ensure_parent_dir(history_uri)
        self.history.to_parquet(history_uri, index=False)
        future_x_uri = None
        if self.future_x is not None:
            future_x_uri = join_uri(base_uri, f"{self.unique_id}.future_x.parquet")
            ensure_parent_dir(future_x_uri)
            self.future_x.to_parquet(future_x_uri, index=False)
        return ForecastTaskRef(
            unique_id=self.unique_id,
            model_config=dict(self.model_config),
            horizon=self.horizon,
            forecast_origin=self.forecast_origin,
            history_uri=history_uri,
            future_x_uri=future_x_uri,
        )
