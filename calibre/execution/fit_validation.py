"""Fit-time config validation.

The ``/fit`` lifecycle job used to flip a record to SUCCEEDED without fitting
anything, so a config incompatible with the data (unknown model/backend, bad
freq, a SKU with no history) only blew up lazily at ``/predict``.

``validate_fit_config`` is a **compatibility gate**: it mirrors ``/predict``'s
single multi-SKU panel fit to prove the configured model can fit the supplied
history, and cheaply checks every requested SKU has history. When supplied with
a model artifact cache it also persists the fitted adapter for the canonical
serving origin.

Scope/contract notes:
- It proves *fittability on the provided history*, NOT that an arbitrary future
  ``/predict`` origin will have enough rows after history is sliced to
  ``ds < origin`` — that is request-dependent and cannot be known at fit time.
- The persisted artifact is keyed by the exact fit-time task. A ``/predict``
  request with a different history slice will miss that artifact and fit/cache
  its own state instead.
"""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.core.forecast_task import ForecastTask
from calibre.execution.prediction import fit_predict_task
from calibre.forecasting.cache import ModelArtifactCache


def validate_fit_config(
    *,
    forecaster_config: dict,
    history: pd.DataFrame,
    future_x: pd.DataFrame | None,
    horizon: int,
    freq: str,
    sku_set: list[str],
    cache: ModelArtifactCache | None = None,
) -> str | None:
    """Fit the config against ``history`` to confirm it yields a forecast.

    Raises a descriptive exception when a requested SKU has no history or the
    model/config cannot be fit to the data. Returns the artifact cache key when
    a cache was supplied.
    """
    missing = sorted({uid for uid in sku_set if history[history[UNIQUE_ID] == uid].empty})
    if missing:
        raise ValueError(f"no history rows for sku(s): {missing}")

    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"invalid freq: {freq!r}") from exc

    origin = pd.Timestamp(history[DS].max()) + offset
    # One panel fit over the full history — the same path /predict takes, so the
    # gate can't be stricter than production.
    task = ForecastTask(
        history=history,
        horizon=horizon,
        model_config={**forecaster_config, "freq": freq},
        forecast_origin=origin,
        future_x=future_x,
    )
    prediction = fit_predict_task(task, cache=cache)
    if prediction.forecast.empty:
        raise ValueError(
            "fit produced no forecast; config is incompatible with the supplied history"
        )
    return prediction.artifact_key
