from __future__ import annotations

import pandas as pd

from calibre.tasks.forecast_task import ForecastTask


class NeuralForecastAdapter:
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._nf: object | None = None

    def fit(self, task: ForecastTask) -> None:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, PatchTST, TiDE

        neural_models: dict[str, type] = {
            "NHiTS": NHITS,
            "TiDE": TiDE,
            "PatchTST": PatchTST,
        }
        reserved_keys = frozenset({"model", "name", "freq", "input_size", "max_steps"})

        model_name = self._config["model"]
        model_cls = neural_models[model_name]

        input_size = self._config.get("input_size", 2 * task.horizon)
        max_steps = self._config.get("max_steps", 100)
        freq = self._config.get("freq", "W")

        params = {k: v for k, v in self._config.items() if k not in reserved_keys}
        model = model_cls(
            h=task.horizon,
            input_size=input_size,
            max_steps=max_steps,
            enable_progress_bar=False,
            **params,
        )

        sf_df = pd.DataFrame(
            {
                "unique_id": task.unique_id,
                "ds": task.history["ds"].values,
                "y": task.history["y"].values.astype("float32"),
            }
        )

        self._nf = NeuralForecast(models=[model], freq=freq)
        self._nf.fit(df=sf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("Call fit() before predict()")

        result = self._nf.predict()

        # Extract point forecast column using positional approach
        model_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        point_col = model_cols[0]

        return pd.DataFrame(
            {
                "ds": result["ds"].values,
                "y_hat": result[point_col].astype("float64").values,
                "h": range(1, task.horizon + 1),
            }
        )
