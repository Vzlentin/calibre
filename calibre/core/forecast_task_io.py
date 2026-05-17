from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from calibre.core.forecast_task import ForecastTask


@lru_cache(maxsize=1024)
def _read_history_cached(unique_id: str, history_uri: str) -> pd.DataFrame:
    del unique_id
    return pd.read_parquet(history_uri)


@lru_cache(maxsize=1024)
def _read_future_x_cached(unique_id: str, future_x_uri: str) -> pd.DataFrame:
    del unique_id
    return pd.read_parquet(future_x_uri)


@dataclass(frozen=True)
class ForecastTaskRef:
    unique_id: str
    model_config: dict[str, Any]
    horizon: int
    forecast_origin: pd.Timestamp | None
    history_uri: str
    future_x_uri: str | None = None

    def materialize(self) -> ForecastTask:
        history = _read_history_cached(self.unique_id, self.history_uri).copy()
        future_x = (
            _read_future_x_cached(self.unique_id, self.future_x_uri).copy()
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
