from __future__ import annotations

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, PatchTST, TiDE

from calibre.tasks.forecast_task import ForecastTask

_NEURAL_MODELS: dict[str, type] = {
    "NHiTS": NHITS,
    "TiDE": TiDE,
    "PatchTST": PatchTST,
}

_RESERVED_KEYS = frozenset({"model", "name", "freq", "input_size", "max_steps"})


class NeuralForecastAdapter:
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._nf: NeuralForecast | None = None
        self._model_name: str | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = _NEURAL_MODELS[model_name]

        input_size = self._config.get("input_size", 2 * task.horizon)
        max_steps = self._config.get("max_steps", 100)
        freq = self._config.get("freq", "W")

        params = {k: v for k, v in self._config.items() if k not in _RESERVED_KEYS}
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
        self._model_name = model_name

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._nf is None or self._model_name is None:
            raise RuntimeError("Call fit() before predict()")

        result = self._nf.predict()

        # Extract point forecast column: model class name only (not -median/-lo/-hi variants)
        point_col = next(
            c
            for c in result.columns
            if c not in ("unique_id", "ds")
            and not any(c.endswith(suffix) for suffix in ("-median", "-lo", "-hi"))
            and c.startswith(self._model_name)
        )

        return pd.DataFrame(
            {
                "ds": result["ds"].values,
                "y_hat": result[point_col].astype("float64").values,
                "h": range(1, task.horizon + 1),
            }
        )
