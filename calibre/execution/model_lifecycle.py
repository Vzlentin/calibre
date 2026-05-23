"""Execution-owned fit lifecycle helpers for API jobs.

The HTTP layer supplies request data and persistence dependencies; this
module owns adapter validation, eager artifact fitting, and artifact-backed
prediction so adapter execution stays behind an execution boundary.
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
from calibre.forecasting.adapter_base import CacheableAdapter
from calibre.forecasting.adapter_registry import get_scope
from calibre.forecasting.cache import ModelArtifactCache

AdapterResolver = Callable[[dict[str, Any]], Any]
"""Resolves an adapter from a model config (typically ``resolve_adapter``)."""


def model_config_for_fit(record: FitRecord) -> dict[str, Any]:
    return {**record.forecaster_config, "freq": record.freq}


def validate_fit_record(
    record: FitRecord,
    history: pd.DataFrame,
    future_x: pd.DataFrame | None,
    model_config: dict[str, Any],
    resolver: AdapterResolver,
) -> None:
    if int(record.horizon) < 1:
        raise ValueError("horizon must be at least 1")
    if Y not in history.columns:
        raise ValueError("history must include y")
    pd.tseries.frequencies.to_offset(record.freq)
    present = {str(uid) for uid in history[UNIQUE_ID].dropna().unique()}
    missing = sorted(set(record.sku_set) - present)
    if missing:
        raise ValueError(f"history missing sku(s): {missing}")
    if future_x is not None and not future_x.empty:
        required = {UNIQUE_ID, DS}
        missing_future_cols = sorted(required - set(future_x.columns))
        if missing_future_cols:
            raise ValueError(f"future_x missing columns: {missing_future_cols}")
    resolver(model_config)


def fit_tasks_for_record(
    record: FitRecord,
    history: pd.DataFrame,
    future_x: pd.DataFrame | None,
    model_config: dict[str, Any],
) -> list[tuple[str, ForecastTask]]:
    scope = get_scope(model_config)
    if scope == "global":
        return [
            (
                "__global__",
                ForecastTask(
                    history=history,
                    horizon=record.horizon,
                    model_config=model_config,
                    future_x=future_x,
                ),
            )
        ]

    tasks: list[tuple[str, ForecastTask]] = []
    for uid in record.sku_set:
        label_history = history[history[UNIQUE_ID] == uid].reset_index(drop=True)
        label_future_x = future_x
        if label_future_x is not None and not label_future_x.empty:
            label_future_x = label_future_x[label_future_x[UNIQUE_ID] == uid].reset_index(drop=True)
        tasks.append(
            (
                uid,
                ForecastTask(
                    history=label_history,
                    horizon=record.horizon,
                    model_config=model_config,
                    future_x=label_future_x,
                ),
            )
        )
    return tasks


def fit_model_artifacts(
    record: FitRecord,
    history: pd.DataFrame,
    future_x: pd.DataFrame | None,
    cache: ModelArtifactCache,
    resolver: AdapterResolver,
) -> dict[str, str]:
    """Validate the record, fit each label's adapter, and return artifact URIs."""
    model_config = model_config_for_fit(record)
    validate_fit_record(record, history, future_x, model_config, resolver)
    artifacts: dict[str, str] = {}
    for label, task in fit_tasks_for_record(record, history, future_x, model_config):
        adapter = resolver(task.model_config)
        if not isinstance(adapter, CacheableAdapter):
            fit_adapter_for_task(adapter, task, cache=None)
            continue
        fit_adapter_for_task(adapter, task, cache)
        cache_key_fn = getattr(adapter, "cache_key", None)
        if not callable(cache_key_fn):
            raise RuntimeError(f"Cacheable adapter for label {label!r} has no cache_key")
        cache_key = str(cache_key_fn(task))
        artifact_uri = cache.uri_for(cache_key)
        if cache.load_by_uri(artifact_uri) is None:
            raise RuntimeError(f"Adapter artifact was not written for label {label!r}")
        artifacts[label] = artifact_uri
    return artifacts


def predict_from_artifacts(
    record: FitRecord,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    forecaster_config: dict[str, Any],
    future_x: pd.DataFrame | None,
    cache: ModelArtifactCache | None,
    resolver: AdapterResolver,
) -> pd.DataFrame:
    """Load each fit artifact through its persisted URI and predict.

    ``record.artifact_urls`` is the source of truth for which adapter state
    to load. If a persisted URI exists but cannot be loaded, prediction fails
    instead of silently re-fitting. For eager artifacts fit on the full fit
    history, origins inside that fit history are rejected to avoid training
    on future actuals relative to the requested origin.
    """
    if record.artifact_urls:
        fit_cutoff = pd.Timestamp(history[DS].max())
        if pd.Timestamp(origin) <= fit_cutoff:
            raise ValueError(
                "origin must be after the fit history cutoff when using eager fit artifacts "
                f"(origin={pd.Timestamp(origin)}, fit_cutoff={fit_cutoff})"
            )

    scope = get_scope(forecaster_config)
    labels = ["__global__"] if scope == "global" else list(record.sku_set)
    artifact_cache = cache if record.artifact_urls else None

    pred_frames: list[pd.DataFrame] = []
    for label in labels:
        if scope == "global":
            label_history = history
            label_future_x = future_x
        else:
            label_history = history[history[UNIQUE_ID] == label]
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
        if artifact_uri:
            if artifact_cache is None:
                raise RuntimeError(f"Model artifact cache is required for label {label!r}")
            blob = artifact_cache.load_by_uri(artifact_uri)
            if blob is None:
                raise FileNotFoundError(
                    f"Persisted model artifact for label {label!r} was not found: {artifact_uri}"
                )
            if not hasattr(adapter, "load_state"):
                raise RuntimeError(f"Adapter for label {label!r} cannot load persisted state")
            adapter.load_state(blob)
        else:
            adapter.fit(task)
        preds = adapter.predict(task)
        pred_frames.append(finalize_preds(preds, origin, task.model_name))

    if not pred_frames:
        raise ValueError("history is empty before origin")
    return coerce_forecast_frame_dtypes(pd.concat(pred_frames, ignore_index=True))
