from __future__ import annotations

import logging
import os

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import sessionmaker

from calibre.api import tuning_service
from calibre.api.errors import format_error
from calibre.api.lifecycle import FitRecord, LifecycleStore, LifecycleStoreProtocol, TuneRecord
from calibre.api.run_store import MemoryRunStore, RunStore, SqlRunStore
from calibre.api.schemas import (
    BacktestRequest,
    CalibrateRequest,
    CalibrateResponse,
    FitHandle,
    FitRequest,
    ObserveRequest,
    ObserveResponse,
    OrderRequest,
    OrderResponse,
    PredictRequest,
    PredictResponse,
    RunResponse,
    SessionStateResponse,
    TuneCandidatePayload,
    TuneHandle,
    TuneRequest,
    TuneStudyResponse,
)
from calibre.cli.config import ConformalConfig
from calibre.conformal.runtime import (
    SymmetricIntervalRuntime,
    build_symmetric_interval_runtime,
)
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.io import join_uri, read_parquet
from calibre.core.run_status import RunStatus
from calibre.core.serialization import frame_from_records, json_safe_records
from calibre.execution.dataset import SalesAdapter, SnapshotSalesAdapter
from calibre.execution.decision_loop import observe_cumulative, observe_per_horizon
from calibre.execution.fit_validation import validate_fit_config
from calibre.execution.prediction import (
    _coerce_forecast_frame_dtypes,
    _finalize_preds,
    fit_predict_task,
)
from calibre.forecasting import get_scope
from calibre.forecasting.cache import ModelArtifactCache
from calibre.ordering.policy_config import (
    NewsvendorConfig,
    OrderPolicy,
    RsConfig,
    RssConfig,
    apply_order_policy,
)
from calibre.storage.lifecycle_repo import SqlLifecycleStore
from calibre.storage.objstore import artifact_base_uri, signed_url
from calibre.storage.postgres import (
    database_url,
    make_engine,
    make_session_factory,
)
from calibre.storage.sales_repo import SqlSalesAdapter
from calibre.storage.session import derive_session_id

app = FastAPI(title="Calibre", version="0.1.0")

logger = logging.getLogger(__name__)

_MEMORY_STORE = MemoryRunStore()
_DB_URL: str | None = None
_DB_FACTORY: sessionmaker | None = None
_SQL_STORE: SqlRunStore | None = None
_LIFECYCLE_STORE = LifecycleStore()
_SQL_LIFECYCLE_STORE: SqlLifecycleStore | None = None


def _db_session_factory() -> sessionmaker | None:
    global _DB_FACTORY, _DB_URL, _SQL_STORE, _SQL_LIFECYCLE_STORE
    url = database_url()
    if not url:
        return None
    if _DB_FACTORY is None or url != _DB_URL:
        _DB_URL = url
        _DB_FACTORY = make_session_factory(make_engine(url))
        _SQL_STORE = None
        _SQL_LIFECYCLE_STORE = None
    return _DB_FACTORY


def _run_store() -> RunStore:
    global _SQL_STORE
    factory = _db_session_factory()
    if factory is None:
        return _MEMORY_STORE
    if _SQL_STORE is None:
        _SQL_STORE = SqlRunStore(factory)
    return _SQL_STORE


def _lifecycle_store() -> LifecycleStoreProtocol:
    """Select the lifecycle store: SQL when LIFECYCLE_STORE=sql, else in-memory.

    The in-memory store is process-local (lost on restart, invisible across
    workers); set LIFECYCLE_STORE=sql with CALIBRE_DATABASE_URL for a
    deployment that survives both.
    """
    global _SQL_LIFECYCLE_STORE
    if os.environ.get("LIFECYCLE_STORE") != "sql":
        return _LIFECYCLE_STORE
    factory = _db_session_factory()
    if factory is None:
        raise RuntimeError("LIFECYCLE_STORE=sql requires CALIBRE_DATABASE_URL to be set")
    if _SQL_LIFECYCLE_STORE is None:
        _SQL_LIFECYCLE_STORE = SqlLifecycleStore(factory)
    return _SQL_LIFECYCLE_STORE


def _model_artifact_cache() -> ModelArtifactCache:
    return ModelArtifactCache(join_uri(artifact_base_uri(), "model-artifacts"))


def _read_parquet_uri(uri: str, label: str) -> pd.DataFrame:
    """Read a parquet URI, mapping a missing/unreadable file to a 400.

    Keeps URI-ingress reads (``future_x``, ``actuals``) returning a client error
    rather than a 500 when the URI doesn't resolve."""
    try:
        return read_parquet(uri)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"{label} not readable: {exc}") from exc


def _sales_adapter(uri: str) -> SalesAdapter:
    """Resolve a ``sales_uri`` to a SalesAdapter by scheme.

    ``sql://`` / ``db://`` reads the project's own Postgres ``sales`` table;
    anything else is treated as a parquet/fsspec snapshot URI."""
    if uri.startswith(("sql://", "db://")):
        factory = _db_session_factory()
        if factory is None:
            raise HTTPException(
                status_code=400,
                detail="sql:// sales_uri requires CALIBRE_DATABASE_URL to be set",
            )
        return SqlSalesAdapter(factory)
    return SnapshotSalesAdapter(uri)


def _resolve_as_of(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid as_of: {exc}") from exc


def _load_sales(sales_uri: str, sku_set: list[str], as_of: pd.Timestamp | None) -> pd.DataFrame:
    try:
        history = _sales_adapter(sales_uri).load_sales(sku_set, as_of)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"sales_uri not readable: {exc}") from exc
    if UNIQUE_ID not in history.columns or DS not in history.columns:
        raise HTTPException(status_code=400, detail="sales history must include unique_id and ds")
    return history


def _actuals_lookup(actuals: pd.DataFrame) -> pd.Series:
    """Build a ``(unique_id, ds) -> y`` Series for the decision-loop observe fns.

    Mirrors the lookup ``DecisionLoop.run`` constructs so the API observes
    through the same code path: ``(str, Timestamp)`` keys, dropping rows with no
    actual ``y`` to record.
    """
    cache = {
        (str(row[UNIQUE_ID]), pd.Timestamp(row[DS])): float(row[Y])
        for _, row in actuals.iterrows()
        if not pd.isna(row[Y])
    }
    lookup = pd.Series(cache, dtype=float)
    if not lookup.empty:
        lookup.index = pd.MultiIndex.from_tuples(lookup.index)
    return lookup


def _merge_future_x_override(
    base: pd.DataFrame | None,
    override: dict[str, list[dict]] | None,
) -> pd.DataFrame | None:
    if not override:
        return base.copy() if base is not None else None

    override_frames = []
    for uid, records in override.items():
        if not records:
            continue
        frame = pd.DataFrame(records)
        if DS not in frame.columns:
            raise ValueError(f"future_x_override for {uid!r} must include ds")
        frame[UNIQUE_ID] = uid
        frame[DS] = pd.to_datetime(frame[DS])
        override_frames.append(frame)

    if not override_frames:
        return base.copy() if base is not None else None

    override_frame = pd.concat(override_frames, ignore_index=True).drop_duplicates(
        [UNIQUE_ID, DS], keep="last"
    )

    if base is None or base.empty:
        base_frame = pd.DataFrame(columns=[UNIQUE_ID, DS])
    else:
        base_frame = base.copy()
    if UNIQUE_ID not in base_frame.columns or DS not in base_frame.columns:
        raise ValueError("future_x must include unique_id and ds to apply an override")
    base_frame[UNIQUE_ID] = base_frame[UNIQUE_ID].astype(str)
    base_frame[DS] = pd.to_datetime(base_frame[DS])

    merged = override_frame.set_index([UNIQUE_ID, DS]).combine_first(
        base_frame.set_index([UNIQUE_ID, DS])
    )
    return merged.reset_index()


def _conformal_config_from_dict(payload: dict) -> ConformalConfig:
    return ConformalConfig(
        method=payload["method"],
        coverage=float(payload.get("coverage", 0.9)),
        calibration_window=int(payload.get("calibration_window", 100)),
        gamma=float(payload.get("gamma", 0.05)),
        mode=payload.get("mode", "perhorizon"),
        protection_period=payload.get("protection_period"),
    )


def _runtime_for_session(
    record: FitRecord,
) -> SymmetricIntervalRuntime:
    assert record.conformal_config is not None
    runtime_config = _conformal_config_from_dict(record.conformal_config).to_runtime_config()
    saved = _lifecycle_store().get_conformal_state(record.session_id)
    if saved:
        return SymmetricIntervalRuntime.from_partition_states(runtime_config, saved)
    return build_symmetric_interval_runtime(runtime_config)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/backtests", response_model=RunResponse, status_code=202)
def backtests(
    req: BacktestRequest,
    bg: BackgroundTasks,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> RunResponse:
    store = _run_store()
    run = store.create(req.config, idempotency_key=idempotency_key)
    if run.status == RunStatus.FAILED:
        run = store.queue(run.id)
    if run.status in {RunStatus.QUEUED, RunStatus.FAILED}:
        bg.add_task(store.run_backtest_job, run.id)
    return run


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run_status(run_id: str) -> RunResponse:
    run = _run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post("/fit", response_model=FitHandle, status_code=202)
def fit(req: FitRequest, bg: BackgroundTasks) -> FitHandle:
    if not req.sku_set:
        raise HTTPException(status_code=400, detail="sku_set must not be empty")
    history = _load_sales(req.sales_uri, list(req.sku_set), _resolve_as_of(req.as_of))
    future_x = _read_parquet_uri(req.future_x_uri, "future_x_uri") if req.future_x_uri else None
    session_id = derive_session_id(
        req.tenant,
        req.sku_set,
        req.forecaster_config,
        req.conformal_config or {},
    )
    fit_id = LifecycleStore.new_fit_id()
    record = FitRecord(
        fit_id=fit_id,
        session_id=session_id,
        tenant=req.tenant,
        sku_set=list(req.sku_set),
        forecaster_config=dict(req.forecaster_config),
        horizon=int(req.horizon),
        freq=req.freq,
        history=history,
        future_x=future_x,
        conformal_config=dict(req.conformal_config) if req.conformal_config else None,
        status=RunStatus.QUEUED,
    )
    _lifecycle_store().put_fit(record)
    bg.add_task(_run_fit_job, fit_id)
    return FitHandle(
        fit_id=fit_id,
        session_id=session_id,
        status=RunStatus.QUEUED,
    )


def _run_fit_job(fit_id: str) -> None:
    store = _lifecycle_store()
    record = store.get_fit(fit_id)
    if record is None:
        return
    store.update_fit(fit_id, status=RunStatus.RUNNING)
    try:
        cache = _model_artifact_cache()
        # Eagerly fit to validate the config against the data, so an
        # incompatible config FAILS here rather than silently succeeding and
        # only blowing up at /predict.
        artifact_key = validate_fit_config(
            forecaster_config=record.forecaster_config,
            history=record.history,
            future_x=record.future_x,
            horizon=record.horizon,
            freq=record.freq,
            sku_set=record.sku_set,
            cache=cache,
        )
        artifact_urls = {"session_id": record.session_id}
        if artifact_key is not None:
            artifact_urls["model_artifact"] = signed_url(cache.uri_for_key(artifact_key))
        store.update_fit(
            fit_id,
            status=RunStatus.SUCCEEDED,
            artifact_urls=artifact_urls,
        )
    except Exception as exc:
        logger.exception("fit job failed", extra={"fit_id": fit_id})
        store.update_fit(fit_id, status=RunStatus.FAILED, error=format_error(exc))


@app.get("/fits/{fit_id}", response_model=FitHandle)
def get_fit(fit_id: str) -> FitHandle:
    record = _lifecycle_store().get_fit(fit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="fit not found")
    return FitHandle(
        fit_id=record.fit_id,
        session_id=record.session_id,
        status=record.status,
        artifact_urls=dict(record.artifact_urls),
        error=record.error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    record = _lifecycle_store().get_fit(req.fit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="fit not found")
    if record.status != RunStatus.SUCCEEDED:
        raise HTTPException(status_code=409, detail=f"fit not ready: status={record.status.value}")
    try:
        origin = pd.Timestamp(req.origin)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid origin: {exc}") from exc

    history = record.history[record.history[DS] < origin]
    if history.empty:
        raise HTTPException(status_code=400, detail="history is empty before origin")

    forecaster_config = {**record.forecaster_config, "freq": record.freq}
    try:
        future_x = _merge_future_x_override(record.future_x, req.future_x_override)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    task = ForecastTask(
        history=history,
        horizon=record.horizon,
        model_config=forecaster_config,
        forecast_origin=origin,
        future_x=future_x,
    )
    try:
        prediction = fit_predict_task(task, cache=_model_artifact_cache())
    except Exception as exc:
        logger.exception("predict failed", extra={"fit_id": req.fit_id})
        raise HTTPException(status_code=500, detail=format_error(exc)) from exc
    forecast_frame = _coerce_forecast_frame_dtypes(
        _finalize_preds(prediction.forecast, origin, task.model_name)
    )
    _lifecycle_store().update_fit(record.fit_id, last_forecast=forecast_frame)
    return PredictResponse(rows=len(forecast_frame), forecast=json_safe_records(forecast_frame))


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest) -> CalibrateResponse:
    store = _lifecycle_store()
    record = store.first_fit_for_session(req.session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    if record.conformal_config is None:
        raise HTTPException(status_code=400, detail="session has no conformal config")
    forecast_frame = _coerce_forecast_frame_dtypes(frame_from_records(req.forecast))
    runtime = _runtime_for_session(record)
    calibrated_frame = runtime.apply(forecast_frame)
    partition_states = runtime.get_partition_states()
    store.upsert_conformal_state(req.session_id, partition_states)
    store.update_fit(record.fit_id, last_calibrated=calibrated_frame)
    return CalibrateResponse(
        rows=len(calibrated_frame),
        calibrated=json_safe_records(calibrated_frame),
    )


def _build_order_policy(ordering: dict) -> OrderPolicy:
    """Map an ``/order`` ordering spec to the per-policy config.

    Rejects ``quantile`` for the rss and newsvendor policies (where it does not
    apply); unrecognized knobs are otherwise ignored.
    """
    policy = ordering["policy"]
    params = ordering["params"]
    params_frame = params if isinstance(params, pd.DataFrame) else pd.DataFrame(params)
    coverage = float(ordering.get("coverage", 0.9))
    if policy == "rs":
        quantile = ordering.get("quantile")
        return RsConfig(
            params=params_frame,
            coverage=coverage,
            quantile=None if quantile is None else float(quantile),
        )
    if policy == "rss":
        if ordering.get("quantile") is not None:
            raise ValueError("quantile is not a valid knob for the rss policy")
        return RssConfig(params=params_frame, coverage=coverage)
    if policy == "newsvendor":
        if ordering.get("quantile") is not None:
            raise ValueError("quantile is not a valid knob for the newsvendor policy")
        return NewsvendorConfig(
            params=params_frame,
            coverage=coverage,
            period=int(ordering.get("period", 1)),
        )
    raise ValueError(f"unknown order policy: {policy!r}")


@app.post("/order", response_model=OrderResponse)
def order(req: OrderRequest) -> OrderResponse:
    frame = _coerce_forecast_frame_dtypes(frame_from_records(req.calibrated))
    try:
        policy_config = _build_order_policy(req.ordering)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid ordering spec: {exc}") from exc
    try:
        orders_frame = apply_order_policy(frame, policy_config)
    except Exception as exc:
        logger.exception("order policy application failed")
        raise HTTPException(status_code=400, detail=format_error(exc)) from exc
    if req.session_id is not None:
        store = _lifecycle_store()
        record = store.first_fit_for_session(req.session_id)
        if record is not None:
            store.put_orders(record.tenant, req.session_id, orders_frame)
    return OrderResponse(rows=len(orders_frame), orders=json_safe_records(orders_frame))


@app.post("/observe", response_model=ObserveResponse, status_code=202)
def observe(req: ObserveRequest, bg: BackgroundTasks) -> ObserveResponse:
    store = _lifecycle_store()
    record = store.first_fit_for_session(req.session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    if record.conformal_config is None:
        raise HTTPException(status_code=400, detail="session has no conformal config")
    bg.add_task(_run_observe_job, req.session_id, req.actuals)
    return ObserveResponse(session_id=req.session_id, status=RunStatus.QUEUED)


def _run_observe_job(session_id: str, actual_records: list[dict]) -> None:
    store = _lifecycle_store()
    record = store.first_fit_for_session(session_id)
    if record is None:
        logger.warning("observe skipped: no fit for session", extra={"session_id": session_id})
        return
    if record.conformal_config is None:
        logger.warning(
            "observe skipped: session has no conformal config",
            extra={"session_id": session_id},
        )
        return
    if record.last_calibrated is None or record.last_calibrated.empty:
        logger.warning(
            "observe skipped: no calibrated frame on session (call /calibrate first)",
            extra={"session_id": session_id},
        )
        return

    actuals = frame_from_records(actual_records)
    if (
        actuals.empty
        or UNIQUE_ID not in actuals.columns
        or DS not in actuals.columns
        or Y not in actuals.columns
    ):
        logger.warning(
            "observe skipped: actuals empty or missing unique_id/ds/y",
            extra={"session_id": session_id, "rows": len(actuals)},
        )
        return

    runtime = _runtime_for_session(record)
    lower_col, upper_col = runtime.interval_columns
    calibrated = record.last_calibrated.copy()
    if lower_col not in calibrated.columns or upper_col not in calibrated.columns:
        logger.warning(
            "observe skipped: calibrated frame missing interval columns",
            extra={"session_id": session_id, "expected": [lower_col, upper_col]},
        )
        return
    if Y not in calibrated.columns:
        # last_calibrated normally carries y (a NaN placeholder) from the
        # predict output, but a hand-crafted /calibrate payload may omit it.
        # The observe dispatch fills actuals into this column, so ensure it
        # exists rather than letting _fill_actuals raise KeyError.
        calibrated[Y] = float("nan")

    actuals_lookup = _actuals_lookup(actuals)
    if actuals_lookup.empty:
        logger.warning(
            "observe skipped: no usable actuals (no non-null y rows)",
            extra={"session_id": session_id, "rows": len(actuals)},
        )
        return

    # Dispatch on the conformal mode rather than pre-filtering rows with NaN
    # bounds. Cumulative mode emits NaN bounds on a window's intermediate
    # horizons by construction, so dropping NaN-bound rows (the old behaviour)
    # discarded exactly the observations the cumulative runtime needs to
    # complete a window (lessons.md §40). decision_loop owns the per-horizon vs
    # cumulative readiness logic; route through it so the API cannot diverge.
    if runtime.mode == "cumulative":
        observe_cumulative(runtime, [calibrated], actuals_lookup)
    else:
        observe_per_horizon(runtime, [calibrated], actuals_lookup, lower_col, upper_col)
    store.upsert_conformal_state(session_id, runtime.get_partition_states())


@app.post("/tune", response_model=TuneHandle, status_code=202)
def tune(req: TuneRequest, bg: BackgroundTasks) -> TuneHandle:
    if not req.sku_set:
        raise HTTPException(status_code=400, detail="sku_set must not be empty")
    if not tuning_service.has_search_space(req.search_space_id):
        raise HTTPException(
            status_code=400, detail=f"unknown search_space_id: {req.search_space_id}"
        )
    if not tuning_service.has_objective(req.objective_id):
        raise HTTPException(status_code=400, detail=f"unknown objective_id: {req.objective_id}")
    try:
        model_scope = get_scope(req.base_model_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if req.hpo_scope != model_scope:
        raise HTTPException(
            status_code=422,
            detail=(
                f"hpo_scope={req.hpo_scope!r} does not match "
                f"base_model_config scope={model_scope!r}"
            ),
        )
    history = _load_sales(req.sales_uri, list(req.sku_set), _resolve_as_of(req.as_of))
    actuals = _read_parquet_uri(req.actuals_uri, "actuals_uri")
    if not req.origins:
        raise HTTPException(status_code=400, detail="origins must not be empty")
    try:
        origins = [pd.Timestamp(o) for o in req.origins]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid origins: {exc}") from exc
    session_id = derive_session_id(
        req.tenant,
        req.sku_set,
        req.base_model_config,
        req.conformal_config or {},
    )
    study_id = LifecycleStore.new_study_id()
    store = _lifecycle_store()
    store.put_study(
        TuneRecord(
            study_id=study_id,
            session_id=session_id,
            tenant=req.tenant,
            sku_set=list(req.sku_set),
            status=RunStatus.QUEUED,
        )
    )
    factory = _db_session_factory()
    bg.add_task(
        tuning_service.run_tune_job,
        study_id,
        req,
        history,
        actuals,
        origins,
        store=store,
        factory=factory,
    )
    return TuneHandle(study_id=study_id, session_id=session_id, status=RunStatus.QUEUED)


@app.get("/studies/{study_id}", response_model=TuneStudyResponse)
def get_study(study_id: str) -> TuneStudyResponse:
    record = _lifecycle_store().get_study(study_id)
    if record is None:
        raise HTTPException(status_code=404, detail="study not found")
    best_candidates = {
        uid: TuneCandidatePayload(
            model_config_values=payload.get("model_config", {}),
            conformal_config=payload.get("conformal_config", {}),
            ordering_config=payload.get("ordering_config", {}),
        )
        for uid, payload in record.best_candidates.items()
    }
    return TuneStudyResponse(
        study_id=record.study_id,
        session_id=record.session_id,
        tenant=record.tenant,
        sku_set=list(record.sku_set),
        status=record.status,
        best_candidates=best_candidates,
        error=record.error,
    )


def _maybe_json_records(frame: pd.DataFrame | None) -> list[dict] | None:
    if frame is None or frame.empty:
        return None
    return json_safe_records(frame)


@app.get("/sessions/{tenant}/{uid}", response_model=SessionStateResponse)
def session_state(tenant: str, uid: str) -> SessionStateResponse:
    store = _lifecycle_store()
    fits = store.fits_for_tenant_uid(tenant, uid)
    if not fits:
        raise HTTPException(status_code=404, detail="session not found")
    # fits_for_tenant_uid returns metadata only; load frames for the selected fit.
    record = store.get_fit(fits[-1].fit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionStateResponse(
        session_id=record.session_id,
        tenant=tenant,
        unique_id=uid,
        state=store.get_conformal_state(record.session_id),
        last_forecast=_maybe_json_records(record.last_forecast),
        open_orders=_maybe_json_records(store.orders_for_tenant_uid(tenant, uid)),
    )
