from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from calibre.contracts.forecast_frame import UNIQUE_ID


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
