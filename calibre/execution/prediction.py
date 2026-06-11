from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    CALIBRATION_STATE_REF,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    CONFORMAL_PARTITION,
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    is_quantile_column,
)
from calibre.core.forecast_task import (
    ChunkTaskRef,
    ForecastTask,
    ForecastTaskRef,
    _read_parquet_cached,
)
from calibre.forecasting.adapter_base import PredictionResult
from calibre.forecasting.adapter_registry import resolve_adapter
from calibre.forecasting.cache import ModelArtifactCache

logger = logging.getLogger(__name__)


def _finalize_preds(preds: pd.DataFrame, origin: pd.Timestamp, model_name: str) -> pd.DataFrame:
    preds[FORECAST_ORIGIN] = origin
    preds[MODEL_NAME] = model_name
    preds[Y] = np.nan
    extras = [c for c in preds.columns if is_quantile_column(c) and c not in REQUIRED_COLUMNS]
    return preds[REQUIRED_COLUMNS + extras]


def fit_predict_task(
    task: ForecastTask,
    cache: ModelArtifactCache | None = None,
    *,
    collect_fitted_values: bool = False,
) -> PredictionResult:
    adapter = resolve_adapter(task.model_config)
    model_name = task.model_name
    uid = task.unique_id
    origin = task.forecast_origin

    fit_started = time.perf_counter()
    fit_ran, artifact_key = adapter.fit_with_cache(
        task,
        cache,
        collect_fitted_values=collect_fitted_values,
    )
    logger.info(
        "completed adapter fit" if fit_ran else "restored adapter from cache",
        extra={
            "origin": origin,
            "model_name": model_name,
            "unique_id": uid,
            "phase": "fit",
            "cache_hit": not fit_ran,
            "duration_ms": round((time.perf_counter() - fit_started) * 1000.0, 3),
        },
    )

    predict_started = time.perf_counter()
    preds = adapter.predict(task)
    logger.info(
        "completed adapter predict",
        extra={
            "origin": origin,
            "model_name": model_name,
            "unique_id": uid,
            "phase": "predict",
            "duration_ms": round((time.perf_counter() - predict_started) * 1000.0, 3),
        },
    )
    fitted_values = adapter.fitted_values(task) if collect_fitted_values else None
    return PredictionResult(
        forecast=preds,
        fitted_values=fitted_values,
        artifact_key=artifact_key,
    )


def _coerce_forecast_frame_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for col in (
        UNIQUE_ID,
        MODEL_NAME,
        CALIBRATION_STATE_REF,
        CONFORMAL_PARTITION,
        CONFORMAL_METHOD,
        CONFORMAL_MODE,
    ):
        if col in result.columns:
            result[col] = result[col].astype("object")
    for col in (DS, FORECAST_ORIGIN):
        if col in result.columns:
            result[col] = pd.to_datetime(result[col]).astype("datetime64[ns]")
    for col in (Y, Y_HAT, NONCONFORMITY_SCORE, CONFORMAL_ALPHA):
        if col in result.columns:
            result[col] = result[col].astype("float64")
    if H in result.columns:
        result[H] = result[H].astype("int64")
    return result


def _empty_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def _empty_prediction_result() -> PredictionResult:
    return PredictionResult(forecast=_empty_forecast_frame())


def _process_local_chunk(
    chunk_ref: ChunkTaskRef,
    origin: pd.Timestamp,
    collect_fitted_values: bool,
) -> PredictionResult:
    """Materialize one staged chunk and fit each of its series independently.

    The chunk panel is read once, then the worker iterates the chunk's uids and
    runs the SAME per-series fit/predict as the per-series path: own adapter
    instance (via ``fit_predict_task``), own history slice, own per-uid future_x
    slice (``None`` when the series has no future_x rows). Fitting per series
    inside the task keeps local semantics exact for every backend — pooled
    adapters (mlforecast/neuralforecast) never see more than one series.

    This function is intentionally conformal- and order-blind so all mutable
    conformal state stays on the driver. The chunk identity stays staging-local:
    every returned row carries its real per-series unique_id.
    """
    panel = _read_parquet_cached(chunk_ref.history_uri)
    future_panel = (
        _read_parquet_cached(chunk_ref.future_x_uri) if chunk_ref.future_x_uri is not None else None
    )
    model_config = dict(chunk_ref.model_config)

    results: list[PredictionResult] = []
    for uid in chunk_ref.unique_ids:
        history = panel[(panel[UNIQUE_ID] == uid) & (panel[DS] < origin)]
        if history.empty:
            continue

        future_x = None
        if future_panel is not None and not future_panel.empty:
            uid_future = future_panel[future_panel[UNIQUE_ID] == uid]
            future_x = uid_future if not uid_future.empty else None

        origin_task = ForecastTask(
            history=history.reset_index(drop=True),
            horizon=chunk_ref.horizon,
            model_config=dict(model_config),
            forecast_origin=origin,
            future_x=future_x.reset_index(drop=True) if future_x is not None else None,
            task_group=chunk_ref.task_group,
        )
        result = fit_predict_task(origin_task, collect_fitted_values=collect_fitted_values)
        results.append(
            PredictionResult(
                forecast=_finalize_preds(result.forecast, origin, origin_task.model_name),
                fitted_values=result.fitted_values,
                artifact_key=result.artifact_key,
            )
        )

    return _concat_prediction_results(results)


def _process_global_panel(
    refs: list[ForecastTaskRef],
    model_config: dict,
    origin: pd.Timestamp,
    collect_fitted_values: bool,
) -> PredictionResult:
    """Fit one global adapter for a config over the full multi-SKU panel."""
    histories: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    horizon: int | None = None
    task_group: str | None = None

    for ref in refs:
        task = ref.materialize()
        history = task.history[task.history[DS] < origin]
        if not history.empty:
            histories.append(history)
        if task.future_x is not None and not task.future_x.empty:
            future_frames.append(task.future_x)
        if horizon is None:
            horizon = task.horizon
        if task_group is None:
            task_group = task.task_group

    if not histories or horizon is None:
        return _empty_prediction_result()

    panel = pd.concat(histories, ignore_index=True).drop_duplicates([UNIQUE_ID, DS])
    future_x = (
        pd.concat(future_frames, ignore_index=True).drop_duplicates([UNIQUE_ID, DS])
        if future_frames
        else None
    )
    origin_task = ForecastTask(
        history=panel,
        horizon=horizon,
        model_config=dict(model_config),
        forecast_origin=origin,
        future_x=future_x,
        task_group=task_group,
    )
    result = fit_predict_task(origin_task, collect_fitted_values=collect_fitted_values)
    return PredictionResult(
        forecast=_finalize_preds(result.forecast, origin, origin_task.model_name),
        fitted_values=result.fitted_values,
        artifact_key=result.artifact_key,
    )


def _concat_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_forecast_frame()
    return _coerce_forecast_frame_dtypes(pd.concat(non_empty, ignore_index=True))


def _concat_prediction_results(results: list[PredictionResult]) -> PredictionResult:
    if not results:
        return _empty_prediction_result()
    forecasts = _concat_prediction_frames([result.forecast for result in results])
    fitted_parts = [
        result.fitted_values
        for result in results
        if result.fitted_values is not None and not result.fitted_values.empty
    ]
    fitted = pd.concat(fitted_parts, ignore_index=True) if fitted_parts else None
    return PredictionResult(forecast=forecasts, fitted_values=fitted)
