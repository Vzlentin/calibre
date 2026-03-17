from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, MFLES, SeasonalNaive

from calibre.tasks.forecast_task import ForecastTask

_NIXTLA_MODELS: dict[str, type] = {
    "SeasonalNaive": SeasonalNaive,
    "AutoARIMA": AutoARIMA,
    "AutoETS": AutoETS,
    "MFLES": MFLES,
}


class StatsForecastAdapter:
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._sf: StatsForecast | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = _NIXTLA_MODELS[model_name]
        params = {k: v for k, v in self._config.items() if k not in ("model", "name", "freq")}
        model = model_cls(**params)

        freq = self._config.get("freq", "W")
        sf_df = pd.DataFrame(
            {
                "unique_id": task.unique_id,
                "ds": task.history["ds"].values,
                "y": task.history["y"].values.astype("float32"),
            }
        )

        self._sf = StatsForecast(models=[model], freq=freq)
        self._sf.fit(sf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() before predict()")

        result = self._sf.predict(h=task.horizon)
        model_cols = [c for c in result.columns if c not in ("unique_id", "ds")]

        return pd.DataFrame(
            {
                "ds": result["ds"],
                "y_hat": result[model_cols[0]].astype("float64"),
                "h": range(1, task.horizon + 1),
            }
        )
