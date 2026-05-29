"""Fit-time config validation (roadmap P0.3).

The ``/fit`` lifecycle job used to flip a record to SUCCEEDED without fitting
anything, so a config incompatible with the data (unknown model/backend, bad
freq, a SKU with no history) only blew up lazily at ``/predict``.

``validate_fit_config`` is a **compatibility gate**: it mirrors ``/predict``'s
single multi-SKU panel fit (via ``fit_predict_task``) to prove the configured
model can fit the supplied history, and cheaply checks every requested SKU has
history. It lives in the execution layer (FIX #3) so the API route stays thin.

Scope/contract notes:
- It proves *fittability on the provided history*, NOT that an arbitrary future
  ``/predict`` origin will have enough rows after history is sliced to
  ``ds < origin`` — that is request-dependent and cannot be known at fit time.
- It does not persist the fitted model; ``/predict`` still fits. Artifact
  persistence + reuse is a separate follow-up (the adapters don't implement
  state serialization yet).
"""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import fit_predict_task


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

    Raises a descriptive exception when a requested SKU has no history or the
    model/config cannot be fit to the data. Does not persist any fitted state.
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
    preds = fit_predict_task(task)
    if preds is None or preds.empty:
        raise ValueError(
            "fit produced no forecast; config is incompatible with the supplied history"
        )
