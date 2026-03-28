from __future__ import annotations

from typing import Protocol

import pandas as pd

from calibre.tasks.forecast_task import ForecastTask


class ModelAdapter(Protocol):
    def fit(self, task: ForecastTask) -> None: ...
    def predict(self, task: ForecastTask) -> pd.DataFrame: ...
