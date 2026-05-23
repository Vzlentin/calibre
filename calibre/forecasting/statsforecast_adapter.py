from __future__ import annotations

from typing import Any

import pandas as pd
import statsforecast.models
from statsforecast import StatsForecast

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y, exogenous_columns
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import CacheableAdapter, ModelAdapter, _build_predict_frame

_RESERVED_KEYS = frozenset({"model", "name", "freq", "backend", "scope"})


class StatsForecastAdapter(CacheableAdapter, ModelAdapter):
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._sf: StatsForecast | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = getattr(statsforecast.models, model_name, None)
        if model_cls is None:
            raise ValueError(f"Unknown statsforecast model: {model_name!r}")
        params = {k: v for k, v in self._config.items() if k not in _RESERVED_KEYS}
        model = model_cls(**params)

        freq = self._config.get("freq", "W")
        sf_df = task.history[[UNIQUE_ID, DS, Y, *exogenous_columns(task.history)]].copy()
        sf_df[Y] = sf_df[Y].astype("float32")

        self._sf = StatsForecast(models=[model], freq=freq)
        self._sf.fit(sf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() before predict()")
        predict_kwargs: dict[str, Any] = {"h": task.horizon}
        if task.future_x is not None and not task.future_x.empty:
            predict_kwargs["X_df"] = task.future_x
        return _build_predict_frame(self._sf.predict(**predict_kwargs))
