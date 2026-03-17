from __future__ import annotations

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from mlforecast import MLForecast

from calibre.tasks.forecast_task import ForecastTask

_ML_MODELS: dict[str, type] = {
    "LightGBM": lgb.LGBMRegressor,
    "XGBoost": xgb.XGBRegressor,
}

_RESERVED_KEYS = frozenset({"model", "name", "freq", "lags", "lag_transforms", "target_transforms"})


class MLForecastAdapter:
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._mlf: MLForecast | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = _ML_MODELS[model_name]
        params = {k: v for k, v in self._config.items() if k not in _RESERVED_KEYS}
        model = model_cls(**params)

        freq = self._config.get("freq", "W")
        lags = self._config.get("lags", list(range(1, task.horizon + 1)))
        lag_transforms = self._config.get("lag_transforms")
        target_transforms = self._config.get("target_transforms")

        mlf_kwargs: dict = {"models": [model], "freq": freq, "lags": lags}
        if lag_transforms is not None:
            mlf_kwargs["lag_transforms"] = lag_transforms
        if target_transforms is not None:
            mlf_kwargs["target_transforms"] = target_transforms

        mlf_df = pd.DataFrame(
            {
                "unique_id": task.unique_id,
                "ds": task.history["ds"].values,
                "y": task.history["y"].values.astype("float32"),
            }
        )

        self._mlf = MLForecast(**mlf_kwargs)
        self._mlf.fit(mlf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._mlf is None:
            raise RuntimeError("Call fit() before predict()")

        result = self._mlf.predict(h=task.horizon)
        model_cols = [c for c in result.columns if c not in ("unique_id", "ds")]

        return pd.DataFrame(
            {
                "ds": result["ds"],
                "y_hat": result[model_cols[0]].astype("float64"),
                "h": range(1, task.horizon + 1),
            }
        )
