from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from calibre.core.forecast_task import ForecastTask


@lru_cache(maxsize=1024)
def _read_parquet_cached(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri)


@dataclass(frozen=True)
class ForecastTaskRef:
    unique_id: str
    model_config: dict[str, Any]
    horizon: int
    forecast_origin: pd.Timestamp | None
    history_uri: str
    future_x_uri: str | None = None

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
        )
