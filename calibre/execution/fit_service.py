"""Fit-time config validation (roadmap P0.3).

The ``/fit`` lifecycle job used to flip a record to SUCCEEDED without fitting
anything, so a config incompatible with the data (unknown model, bad freq,
missing regressors, empty history for a SKU) only blew up lazily at ``/predict``.

This service eagerly fits the configured model against the supplied history to
prove it can produce a forecast, raising a descriptive error otherwise. It lives
in the execution layer (FIX #3) so the API route stays thin: create the record,
hand off here, serialize the result. Artifact persistence / reuse is a separate
follow-up (the real adapters do not implement state serialization yet), so this
validation pass does not persist anything.
"""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_registry import get_scope, resolve_adapter


def validate_fit_config(
    *,
    forecaster_config: dict,
    history: pd.DataFrame,
    future_x: pd.DataFrame | None,
    horizon: int,
    freq: str,
    sku_set: list[str],
) -> None:
    """Fit the config against ``history`` to confirm it yields a forecast.

    Raises a descriptive exception when the model/config cannot be fit to the
    data. Does not persist any fitted state.
    """
    config = {**forecaster_config, "freq": freq}
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset is None:
        raise ValueError(f"invalid freq: {freq!r}")
    origin = pd.Timestamp(history[DS].max()) + offset

    if get_scope(forecaster_config) == "global":
        _validate_one(
            ForecastTask(
                history=history,
                horizon=horizon,
                model_config=config,
                forecast_origin=origin,
                future_x=future_x,
            )
        )
        return

    for uid in sku_set:
        uid_history = history[history[UNIQUE_ID] == uid]
        if uid_history.empty:
            raise ValueError(f"no history rows for sku {uid!r}")
        uid_future = None if future_x is None else future_x[future_x[UNIQUE_ID] == uid]
        _validate_one(
            ForecastTask(
                history=uid_history.reset_index(drop=True),
                horizon=horizon,
                model_config=config,
                forecast_origin=origin,
                future_x=uid_future,
            )
        )


def _validate_one(task: ForecastTask) -> None:
    adapter = resolve_adapter(task.model_config)
    adapter.fit(task)
    preds = adapter.predict(task)
    if preds is None or preds.empty:
        raise ValueError(
            f"fit produced no forecast for sku {task.unique_id!r}; "
            "config is incompatible with the supplied history"
        )
