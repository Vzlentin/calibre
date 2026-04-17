from __future__ import annotations

import pandas as pd
import statsforecast.models
from statsforecast import StatsForecast

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y
from calibre.models.base import ModelAdapter, _build_predict_frame
from calibre.tasks.forecast_task import ForecastTask


class StatsForecastAdapter(ModelAdapter):
    PARALLEL_BY_UID = True

    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._sf: StatsForecast | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = getattr(statsforecast.models, model_name, None)
        if model_cls is None:
            raise ValueError(f"Unknown statsforecast model: {model_name!r}")
        params = {
            k: v for k, v in self._config.items() if k not in ("model", "name", "freq", "backend")
        }
        model = model_cls(**params)

        freq = self._config.get("freq", "W")
        sf_df = task.history[[UNIQUE_ID, DS, Y]].copy()
        sf_df[Y] = sf_df[Y].astype("float32")

        self._sf = StatsForecast(models=[model], freq=freq)
        self._sf.fit(sf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() before predict()")
        return _build_predict_frame(self._sf.predict(h=task.horizon))
