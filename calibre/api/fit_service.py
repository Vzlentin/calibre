"""Fit-lifecycle service: validate, train, and re-load fit artifacts.

Pure functions taking ``FitRecord`` plus the ``ModelArtifactCache`` as
dependencies. The HTTP layer (``calibre/api/main.py``) is responsible for
sourcing those dependencies; this module never touches module-level state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from calibre.api.lifecycle import FitRecord
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import (
    coerce_forecast_frame_dtypes,
    finalize_preds,
    fit_adapter_for_task,
)
from calibre.forecasting.adapter_registry import get_scope
from calibre.forecasting.cache import ModelArtifactCache

AdapterResolver = Callable[[dict[str, Any]], Any]
"""Resolves an adapter from a model config (typically ``resolve_adapter``)."""


def model_config_for_fit(record: FitRecord) -> dict[str, Any]:
    return {**record.forecaster_config, "freq": record.freq}


def validate_fit_record(
    record: FitRecord,
    model_config: dict[str, Any],
    resolver: AdapterResolver,
) -> None:
    if int(record.horizon) < 1:
        raise ValueError("horizon must be at least 1")
    if Y not in record.history.columns:
        raise ValueError("history must include y")
    pd.tseries.frequencies.to_offset(record.freq)
    present = {str(uid) for uid in record.history[UNIQUE_ID].dropna().unique()}
    missing = sorted(set(record.sku_set) - present)
    if missing:
        raise ValueError(f"history missing sku(s): {missing}")
    if record.future_x is not None and not record.future_x.empty:
        required = {UNIQUE_ID, DS}
        missing_future_cols = sorted(required - set(record.future_x.columns))
        if missing_future_cols:
            raise ValueError(f"future_x missing columns: {missing_future_cols}")
    resolver(model_config)


def fit_tasks_for_record(
    record: FitRecord,
    model_config: dict[str, Any],
) -> list[tuple[str, ForecastTask]]:
    scope = get_scope(model_config)
    if scope == "global":
        return [
            (
                "__global__",
                ForecastTask(
                    history=record.history,
                    horizon=record.horizon,
                    model_config=model_config,
                    future_x=record.future_x,
                ),
            )
        ]

    tasks: list[tuple[str, ForecastTask]] = []
    for uid in record.sku_set:
        history = record.history[record.history[UNIQUE_ID] == uid].reset_index(drop=True)
        future_x = record.future_x
        if future_x is not None and not future_x.empty:
            future_x = future_x[future_x[UNIQUE_ID] == uid].reset_index(drop=True)
        tasks.append(
            (
                uid,
                ForecastTask(
                    history=history,
                    horizon=record.horizon,
                    model_config=model_config,
                    future_x=future_x,
                ),
            )
        )
    return tasks


def fit_model_artifacts(
    record: FitRecord,
    cache: ModelArtifactCache,
    resolver: AdapterResolver,
) -> dict[str, str]:
    """Validate the record, fit each label's adapter, and return artifact URIs."""
    model_config = model_config_for_fit(record)
    validate_fit_record(record, model_config, resolver)
    artifacts: dict[str, str] = {}
    for label, task in fit_tasks_for_record(record, model_config):
        adapter = resolver(task.model_config)
        fit_adapter_for_task(adapter, task, cache)
        cache_key = getattr(adapter, "cache_key", None)
        if callable(cache_key):
            artifacts[label] = cache.uri_for(str(cache_key(task)))
    return artifacts


def predict_from_artifacts(
    record: FitRecord,
    origin: pd.Timestamp,
    forecaster_config: dict[str, Any],
    future_x: pd.DataFrame | None,
    cache: ModelArtifactCache | None,
    resolver: AdapterResolver,
) -> pd.DataFrame:
    """Load each fit artifact through its persisted URI and predict.

    ``record.artifact_urls`` is the source of truth for which adapter state
    to load — cache keys are never recomputed from the task at predict time.
    Labels with no stored artifact fall back to fit-on-predict against the
    per-label history slice.

    Raises :class:`ValueError` when no label has any history before
    ``origin``; the HTTP layer translates that into a 400.
    """
    scope = get_scope(forecaster_config)
    labels = ["__global__"] if scope == "global" else list(record.sku_set)
    artifact_cache = cache if record.artifact_urls else None

    pred_frames: list[pd.DataFrame] = []
    for label in labels:
        if scope == "global":
            label_history = record.history
            label_future_x = future_x
        else:
            label_history = record.history[record.history[UNIQUE_ID] == label]
            if future_x is not None and not future_x.empty:
                label_future_x = future_x[future_x[UNIQUE_ID] == label].reset_index(drop=True)
            else:
                label_future_x = future_x

        label_history = label_history[label_history[DS] < origin].reset_index(drop=True)
        if label_history.empty:
            continue

        task = ForecastTask(
            history=label_history,
            horizon=record.horizon,
            model_config=forecaster_config,
            forecast_origin=origin,
            future_x=label_future_x,
        )
        adapter = resolver(task.model_config)
        artifact_uri = record.artifact_urls.get(label) if record.artifact_urls else None
        blob = (
            artifact_cache.load_by_uri(artifact_uri)
            if (artifact_cache is not None and artifact_uri)
            else None
        )
        if blob is not None and hasattr(adapter, "load_state"):
            adapter.load_state(blob)
        else:
            adapter.fit(task)
        preds = adapter.predict(task)
        pred_frames.append(finalize_preds(preds, origin, task.model_name))

    if not pred_frames:
        raise ValueError("history is empty before origin")
    return coerce_forecast_frame_dtypes(pd.concat(pred_frames, ignore_index=True))
