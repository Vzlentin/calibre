from __future__ import annotations

import logging
import os
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from calibre.api.lifecycle import (
    FitRecord,
    LifecycleStore,
    MemoryLifecycleStore,
    TuneRecord,
)
from calibre.api.observe_service import (
    run_observe,
    runtime_for_session,
)
from calibre.api.order_service import (
    apply_order_policy,
    build_policy_config,
    persist_orders,
)
from calibre.api.run_store import MemoryRunStore, RunStore, SqlRunStore
from calibre.api.schemas import (
    BacktestRequest,
    CalibrateRequest,
    CalibrateResponse,
    DataSourceSpec,
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
from calibre.api.tune_service import (
    objective_for_study,
    oracle_cost_for_request,
    run_tune_job,
)
from calibre.conformal.runtime import SymmetricIntervalRuntime
from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.core.run_status import RunStatus
from calibre.execution import backend as backend_module
from calibre.execution.backend import coerce_forecast_frame_dtypes
from calibre.execution.model_lifecycle import (
    fit_model_artifacts,
    predict_from_artifacts,
)
from calibre.forecasting.cache import ModelArtifactCache
from calibre.storage.adapters import OrderRepo, SqlSalesAdapter
from calibre.storage.lifecycle_repo import SqlLifecycleStore
from calibre.storage.models import Base
from calibre.storage.postgres import (
    database_url,
    make_engine,
    make_session_factory,
)
from calibre.storage.session import derive_session_id
from calibre.tuning import (
    TuningCandidate,
    TuningObjective,
    optimize_task_candidate,
)

app = FastAPI(title="Calibre", version="0.1.0")

logger = logging.getLogger(__name__)

_MEMORY_STORE = MemoryRunStore()
_DB_URL: str | None = None
_DB_FACTORY: sessionmaker | None = None
_SQL_STORE: SqlRunStore | None = None
_LIFECYCLE_STORE: LifecycleStore | None = MemoryLifecycleStore()
_LIFECYCLE_STORE_KEY: tuple[str, str | None] = ("memory", None)
_MODEL_CACHE: ModelArtifactCache | None = None
_MODEL_CACHE_URI: str | None = None

_SEARCH_SPACES: dict[str, Callable[[optuna.Trial], TuningCandidate]] = {}
_OBJECTIVES: dict[str, TuningObjective] = {}


def register_tuning_search_space(
    name: str, search_space: Callable[[optuna.Trial], TuningCandidate]
) -> None:
    _SEARCH_SPACES[name] = search_space


def register_tuning_objective(name: str, objective: TuningObjective) -> None:
    _OBJECTIVES[name] = objective


def _db_session_factory() -> sessionmaker | None:
    global _DB_FACTORY, _DB_URL, _SQL_STORE
    url = database_url()
    if not url:
        return None
    if _DB_FACTORY is None or url != _DB_URL:
        _DB_URL = url
        _DB_FACTORY = make_session_factory(make_engine(url))
        _SQL_STORE = None
    return _DB_FACTORY


def _run_store() -> RunStore:
    global _SQL_STORE
    factory = _db_session_factory()
    if factory is None:
        return _MEMORY_STORE
    if _SQL_STORE is None:
        _SQL_STORE = SqlRunStore(factory)
    return _SQL_STORE


def _lifecycle_store() -> LifecycleStore:
    global _LIFECYCLE_STORE, _LIFECYCLE_STORE_KEY
    mode = os.environ.get("LIFECYCLE_STORE", "").strip().lower()
    if mode not in {"", "memory", "sql"}:
        raise RuntimeError("LIFECYCLE_STORE must be 'memory' or 'sql'")

    url = database_url()
    desired = mode or ("sql" if url else "memory")
    if desired == "memory":
        key: tuple[str, str | None] = ("memory", None)
        if _LIFECYCLE_STORE is None or key != _LIFECYCLE_STORE_KEY:
            _LIFECYCLE_STORE = MemoryLifecycleStore()
            _LIFECYCLE_STORE_KEY = key
        return _LIFECYCLE_STORE

    sql_url = url or "sqlite+pysqlite:///:memory:"
    key = ("sql", sql_url)
    if _LIFECYCLE_STORE is None or key != _LIFECYCLE_STORE_KEY:
        if url is None:
            engine = make_engine(
                sql_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            Base.metadata.create_all(engine)
        else:
            engine = make_engine(sql_url)
        _LIFECYCLE_STORE = SqlLifecycleStore(make_session_factory(engine))
        _LIFECYCLE_STORE_KEY = key
    return _LIFECYCLE_STORE


def _model_artifact_cache() -> ModelArtifactCache:
    global _MODEL_CACHE, _MODEL_CACHE_URI
    uri = os.environ.get("CALIBRE_MODEL_CACHE_DIR") or str(
        Path(tempfile.gettempdir()) / "calibre-model-cache"
    )
    if _MODEL_CACHE is None or uri != _MODEL_CACHE_URI:
        _MODEL_CACHE = ModelArtifactCache(uri)
        _MODEL_CACHE_URI = uri
    return _MODEL_CACHE


def _json_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    clean = clean.astype(object)
    clean[pd.isna(clean)] = None
    return clean.to_dict(orient="records")


def _frame_from_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    if DS in frame.columns:
        frame[DS] = pd.to_datetime(frame[DS])
    if "forecast_origin" in frame.columns:
        frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    return frame


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


def _format_error(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _resolve_history_source(
    field_name: str,
    inline: list[dict] | None,
    source: DataSourceSpec | None,
    tenant: str,
    sku_set: list[str],
) -> pd.DataFrame:
    if (inline is None) == (source is None):
        raise HTTPException(
            status_code=400,
            detail=f"exactly one of {field_name!r} or {field_name}_source must be provided",
        )
    if inline is not None:
        return _frame_from_records(inline)

    assert source is not None  # narrowed by the XOR check above
    if source.kind == "parquet":
        if not source.uri:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}_source.uri is required when kind='parquet'",
            )
        adapter = SqlSalesAdapter(source=source.uri)
    else:
        factory = _db_session_factory()
        if factory is None:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}_source kind='sql' requires CALIBRE_DATABASE_URL",
            )
        adapter = SqlSalesAdapter(factory=factory, tenant=tenant)
    try:
        frame = adapter.load_history()
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if sku_set:
        frame = frame[frame[UNIQUE_ID].isin(sku_set)].reset_index(drop=True)
    return frame


def _runtime_for_session(record: FitRecord) -> SymmetricIntervalRuntime:
    return runtime_for_session(record, _lifecycle_store())


def _fit_model_artifacts(record: FitRecord) -> dict[str, str]:
    history = _lifecycle_store().get_fit_frame(record.fit_id, "history")
    if history is None:
        raise ValueError("fit record is missing history artifact")
    future_x = _lifecycle_store().get_fit_frame(record.fit_id, "future_x")
    return fit_model_artifacts(
        record,
        history=history,
        future_x=future_x,
        cache=_model_artifact_cache(),
        resolver=backend_module.resolve_adapter,
    )


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
    history = _resolve_history_source(
        "history", req.history, req.history_source, req.tenant, list(req.sku_set)
    )
    if UNIQUE_ID not in history.columns or DS not in history.columns:
        raise HTTPException(status_code=400, detail="history must include unique_id and ds")
    future_x = _frame_from_records(req.future_x) if req.future_x else None
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
        conformal_config=dict(req.conformal_config) if req.conformal_config else None,
        status=RunStatus.QUEUED,
    )
    store = _lifecycle_store()
    store.put_fit(record)
    store.put_fit_frame(fit_id, "history", history)
    if future_x is not None:
        store.put_fit_frame(fit_id, "future_x", future_x)
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
        artifact_urls = _fit_model_artifacts(record)
        store.update_fit(
            fit_id,
            status=RunStatus.SUCCEEDED,
            artifact_urls=artifact_urls,
        )
    except Exception as exc:  # pragma: no cover - background task safety net
        logger.exception("fit job failed", extra={"fit_id": fit_id})
        store.update_fit(fit_id, status=RunStatus.FAILED, error=_format_error(exc))


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

    forecaster_config = {**record.forecaster_config, "freq": record.freq}
    history = _lifecycle_store().get_fit_frame(record.fit_id, "history")
    if history is None:
        raise HTTPException(status_code=500, detail="fit record is missing history artifact")
    base_future_x = _lifecycle_store().get_fit_frame(record.fit_id, "future_x")
    try:
        future_x = _merge_future_x_override(base_future_x, req.future_x_override)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        forecast_frame = predict_from_artifacts(
            record,
            history,
            origin,
            forecaster_config,
            future_x,
            cache=_model_artifact_cache(),
            resolver=backend_module.resolve_adapter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict failed", extra={"fit_id": record.fit_id})
        raise HTTPException(status_code=500, detail=_format_error(exc)) from exc

    _lifecycle_store().update_fit(record.fit_id, last_forecast=forecast_frame)
    return PredictResponse(rows=len(forecast_frame), forecast=_json_records(forecast_frame))


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest) -> CalibrateResponse:
    store = _lifecycle_store()
    record = store.first_fit_for_session(req.session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    if record.conformal_config is None:
        raise HTTPException(status_code=400, detail="session has no conformal config")
    forecast_frame = coerce_forecast_frame_dtypes(_frame_from_records(req.forecast))
    runtime = _runtime_for_session(record)
    calibrated_frame = runtime.apply(forecast_frame)
    partition_states = runtime.get_partition_states()
    store.upsert_conformal_state(req.session_id, partition_states)
    store.update_fit(record.fit_id, last_calibrated=calibrated_frame)
    return CalibrateResponse(
        rows=len(calibrated_frame),
        calibrated=_json_records(calibrated_frame),
    )


@app.post("/order", response_model=OrderResponse)
def order(req: OrderRequest) -> OrderResponse:
    frame = coerce_forecast_frame_dtypes(_frame_from_records(req.calibrated))
    try:
        policy_config = build_policy_config(req.ordering)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid ordering spec: {exc}") from exc
    try:
        orders_frame = apply_order_policy(frame, policy_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=_format_error(exc)) from exc
    if req.session_id is not None:
        persist_orders(_lifecycle_store(), _db_session_factory(), req.session_id, orders_frame)
    return OrderResponse(rows=len(orders_frame), orders=_json_records(orders_frame))


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
    run_observe(
        _lifecycle_store(),
        session_id,
        actual_records,
        runtime_factory=_runtime_for_session,
    )


@app.post("/tune", response_model=TuneHandle, status_code=202)
def tune(req: TuneRequest, bg: BackgroundTasks) -> TuneHandle:
    if not req.sku_set:
        raise HTTPException(status_code=400, detail="sku_set must not be empty")
    if req.search_space_id not in _SEARCH_SPACES:
        raise HTTPException(
            status_code=400, detail=f"unknown search_space_id: {req.search_space_id}"
        )
    if req.objective_id not in _OBJECTIVES:
        raise HTTPException(status_code=400, detail=f"unknown objective_id: {req.objective_id}")
    history = _resolve_history_source(
        "history", req.history, req.history_source, req.tenant, list(req.sku_set)
    )
    if UNIQUE_ID not in history.columns or DS not in history.columns:
        raise HTTPException(status_code=400, detail="history must include unique_id and ds")
    actuals = _resolve_history_source(
        "actuals", req.actuals, req.actuals_source, req.tenant, list(req.sku_set)
    )
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
    try:
        oracle_cost = oracle_cost_for_request(req, _OBJECTIVES[req.objective_id], actuals, origins)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"could not compute regret oracle_cost: {exc}",
        ) from exc

    study_id = LifecycleStore.new_study_id()
    _lifecycle_store().put_study(
        TuneRecord(
            study_id=study_id,
            session_id=session_id,
            tenant=req.tenant,
            sku_set=list(req.sku_set),
            status=RunStatus.QUEUED,
            oracle_cost=oracle_cost,
        )
    )
    bg.add_task(_run_tune_job, study_id, req, history, actuals, origins)
    return TuneHandle(study_id=study_id, session_id=session_id, status=RunStatus.QUEUED)


def _run_tune_job(
    study_id: str,
    req: TuneRequest,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
) -> None:
    store = _lifecycle_store()
    record = store.get_study(study_id)
    if record is None:
        return
    try:
        objective = objective_for_study(_OBJECTIVES[req.objective_id], record.oracle_cost)
    except ValueError as exc:
        store.update_study(study_id, status=RunStatus.FAILED, error=_format_error(exc))
        return
    run_tune_job(
        store=store,
        factory=_db_session_factory(),
        study_id=study_id,
        req=req,
        history=history,
        actuals=actuals,
        origins=origins,
        objective=objective,
        search_space=_SEARCH_SPACES[req.search_space_id],
        optimizer=optimize_task_candidate,
    )


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
        oracle_cost=record.oracle_cost,
        error=record.error,
    )


def _maybe_json_records(frame: pd.DataFrame | None) -> list[dict] | None:
    if frame is None or frame.empty:
        return None
    return _json_records(frame)


@app.get("/sessions/{tenant}/{uid}", response_model=SessionStateResponse)
def session_state(tenant: str, uid: str) -> SessionStateResponse:
    store = _lifecycle_store()
    fits = store.fits_for_tenant_uid(tenant, uid)
    if not fits:
        raise HTTPException(status_code=404, detail="session not found")
    record = fits[-1]
    factory = _db_session_factory()
    open_orders = (
        OrderRepo(factory).list_for_session(record.session_id, tenant=tenant, unique_id=uid)
        if factory is not None
        else store.get_fit_frame(record.fit_id, "last_orders")
    )
    return SessionStateResponse(
        session_id=record.session_id,
        tenant=tenant,
        unique_id=uid,
        state=store.get_conformal_state(record.session_id),
        last_forecast=_maybe_json_records(store.get_fit_frame(record.fit_id, "last_forecast")),
        open_orders=_maybe_json_records(open_orders),
    )
