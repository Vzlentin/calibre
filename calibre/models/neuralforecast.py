from __future__ import annotations

import neuralforecast.models
import pandas as pd
from neuralforecast import NeuralForecast

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y
from calibre.models.base import ModelAdapter, _build_predict_frame
from calibre.tasks.forecast_task import ForecastTask

_RESERVED_KEYS = frozenset({"model", "name", "freq", "input_size", "max_steps", "backend", "scope"})


class NeuralForecastAdapter(ModelAdapter):
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._nf: NeuralForecast | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = getattr(neuralforecast.models, model_name, None)
        if model_cls is None:
            raise ValueError(f"Unknown neuralforecast model: {model_name!r}")

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

        nf_df = task.history[[UNIQUE_ID, DS, Y]].copy()
        nf_df[Y] = nf_df[Y].astype("float32")

        self._nf = NeuralForecast(models=[model], freq=freq)
        self._nf.fit(df=nf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("Call fit() before predict()")
        return _build_predict_frame(self._nf.predict())
