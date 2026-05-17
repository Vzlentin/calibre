from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import neuralforecast.models
    from neuralforecast import NeuralForecast

    _NEURALFORECAST_AVAILABLE = True
except ImportError:  # pragma: no cover
    neuralforecast = None  # type: ignore[assignment]
    NeuralForecast = None  # type: ignore[misc, assignment]
    _NEURALFORECAST_AVAILABLE = False

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y, exogenous_columns
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import ModelAdapter, _build_predict_frame

_RESERVED_KEYS = frozenset({"model", "name", "freq", "input_size", "max_steps", "backend", "scope"})


class NeuralForecastAdapter(ModelAdapter):
    def __init__(self, model_config: dict) -> None:
        if not _NEURALFORECAST_AVAILABLE:
            raise RuntimeError(
                "neuralforecast is not installed. Install calibre with the 'neural' extra: "
                "pip install calibre[neural]"
            )
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

        nf_df = task.history[[UNIQUE_ID, DS, Y, *exogenous_columns(task.history)]].copy()
        nf_df[Y] = nf_df[Y].astype("float32")

        self._nf = NeuralForecast(models=[model], freq=freq)
        self._nf.fit(df=nf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        """Forwards ``task.future_x`` as ``futr_df`` when non-empty.

        The model must declare matching regressor names via
        ``futr_exog_list=[<col>]`` in ``model_config``; the library raises if
        the lists are inconsistent.
        """
        if self._nf is None:
            raise RuntimeError("Call fit() before predict()")
        predict_kwargs: dict[str, Any] = {}
        if task.future_x is not None and not task.future_x.empty:
            predict_kwargs["futr_df"] = task.future_x
        return _build_predict_frame(self._nf.predict(**predict_kwargs))
