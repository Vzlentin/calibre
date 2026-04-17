from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y_HAT, H
from calibre.tasks.forecast_task import ForecastTask


def _build_predict_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Nixtla-format predict result to [unique_id, ds, y_hat, h]."""
    model_col = next(c for c in raw.columns if c not in (UNIQUE_ID, DS))
    out = raw[[UNIQUE_ID, DS]].reset_index(drop=True)
    out[Y_HAT] = raw[model_col].astype("float64").values
    out[H] = out.groupby(UNIQUE_ID).cumcount() + 1
    return out


class ModelAdapter(ABC):
    @abstractmethod
    def __init__(self, model_config: dict) -> None: ...

    @abstractmethod
    def fit(self, task: ForecastTask) -> None: ...

    @abstractmethod
    def predict(self, task: ForecastTask) -> pd.DataFrame: ...
