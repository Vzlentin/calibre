from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastTask:
    unique_id: str
    history: pd.DataFrame
    horizon: int
    model_config: dict
    forecast_origin: pd.Timestamp | None = None
    future_x: pd.DataFrame | None = None

    @property
    def model_name(self) -> str:
        return self.model_config.get("name", self.model_config["model"])
