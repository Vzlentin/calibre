from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable

import optuna
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import sessionmaker

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
from calibre.core.run_status import RunStatus
from calibre.execution.backend import (
    _coerce_forecast_frame_dtypes,
    _finalize_preds,
    _fit_predict_task,
)
from calibre.execution.decision_loop import observe_cumulative, observe_per_horizon
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy
from calibre.storage.lifecycle_repo import SqlLifecycleStore
from calibre.storage.postgres import (
    TuningRunRepo,
    database_url,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.storage.session import derive_session_id
from calibre.tuning import (
    TuningCandidate,
    TuningObjective,
    TuningTask,
    optimize_task_candidate,
)

app = FastAPI(title="Calibre", version="0.1.0")

logger = logging.getLogger(__name__)

_MEMORY_STORE = MemoryRunStore()
_DB_URL: str | None = None
_DB_FACTORY: sessionmaker | None = None
_SQL_STORE: SqlRunStore | None = None
_LIFECYCLE_STORE = LifecycleStore()
_SQL_LIFECYCLE_STORE: SqlLifecycleStore | None = None

_SEARCH_SPACES: dict[str, Callable[[optuna.Trial], TuningCandidate]] = {}
_OBJECTIVES: dict[str, TuningObjective] = {}


def register_tuning_search_space(
    name: str, search_space: Callable[[optuna.Trial], TuningCandidate]
) -> None:
    _SEARCH_SPACES[name] = search_space


def register_tuning_objective(name: str, objective: TuningObjective) -> None:
    _OBJECTIVES[name] = objective


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


def _format_error(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


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
    history = _frame_from_records(req.history)
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
        store.update_fit(
            fit_id,
            status=RunStatus.SUCCEEDED,
            artifact_urls={"session_id": record.session_id},
        )
    except Exception as exc:  # pragma: no cover - background task safety net
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
        preds = _fit_predict_task(task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_format_error(exc)) from exc
    forecast_frame = _coerce_forecast_frame_dtypes(_finalize_preds(preds, origin, task.model_name))
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
    forecast_frame = _coerce_forecast_frame_dtypes(_frame_from_records(req.forecast))
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
    frame = _coerce_forecast_frame_dtypes(_frame_from_records(req.calibrated))
    try:
        params = req.ordering["params"]
        params_frame = params if isinstance(params, pd.DataFrame) else pd.DataFrame(params)
        policy_config = OrderPolicyConfig(
            policy=req.ordering["policy"],
            params=params_frame,
            coverage=float(req.ordering.get("coverage", 0.9)),
            period=int(req.ordering.get("period", 1)),
            quantile=req.ordering.get("quantile"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid ordering spec: {exc}") from exc
    try:
        orders_frame = apply_order_policy(frame, policy_config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_format_error(exc)) from exc
    if req.session_id is not None:
        store = _lifecycle_store()
        record = store.first_fit_for_session(req.session_id)
        if record is not None:
            store.update_fit(record.fit_id, last_orders=orders_frame)
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

    actuals = _frame_from_records(actual_records)
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
    if req.search_space_id not in _SEARCH_SPACES:
        raise HTTPException(
            status_code=400, detail=f"unknown search_space_id: {req.search_space_id}"
        )
    if req.objective_id not in _OBJECTIVES:
        raise HTTPException(status_code=400, detail=f"unknown objective_id: {req.objective_id}")
    history = _frame_from_records(req.history)
    if UNIQUE_ID not in history.columns or DS not in history.columns:
        raise HTTPException(status_code=400, detail="history must include unique_id and ds")
    actuals = _frame_from_records(req.actuals)
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
    _lifecycle_store().put_study(
        TuneRecord(
            study_id=study_id,
            session_id=session_id,
            tenant=req.tenant,
            sku_set=list(req.sku_set),
            status=RunStatus.QUEUED,
        )
    )
    bg.add_task(_run_tune_job, study_id, req, history, actuals, origins)
    return TuneHandle(study_id=study_id, session_id=session_id, status=RunStatus.QUEUED)


def _filter_uid(frame: pd.DataFrame, uid: str) -> pd.DataFrame:
    if frame.empty or UNIQUE_ID not in frame.columns:
        return frame
    return frame[frame[UNIQUE_ID] == uid].reset_index(drop=True)


def _candidate_to_payload(candidate: TuningCandidate) -> dict[str, dict]:
    return {
        "model_config": dict(candidate.model_config),
        "conformal_config": dict(candidate.conformal_config),
        "ordering_config": dict(candidate.ordering_config),
    }


def _load_existing_tuning_run(
    factory: sessionmaker | None, session_id: str, unique_id: str
) -> dict[str, dict] | None:
    if factory is None:
        return None
    with session_scope(factory) as session:
        row = TuningRunRepo(session).get(session_id, unique_id)
        if row is None:
            return None
        candidate = dict(row.candidate)
    return {
        "model_config": dict(candidate.get("model_config", {})),
        "conformal_config": dict(candidate.get("conformal_config", {})),
        "ordering_config": dict(candidate.get("ordering_config", {})),
    }


def _persist_tuning_run(
    factory: sessionmaker | None,
    session_id: str,
    unique_id: str,
    payload: dict[str, dict],
) -> None:
    if factory is None:
        return
    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            session_id,
            unique_id,
            candidate=payload,
            score=None,
        )


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
    store.update_study(study_id, status=RunStatus.RUNNING)
    session_id = record.session_id
    factory = _db_session_factory()
    candidates: dict[str, dict[str, dict]] = {}
    try:
        for uid in req.sku_set:
            existing = _load_existing_tuning_run(factory, session_id, uid)
            if existing is not None:
                candidates[uid] = existing
                continue
            task = TuningTask(
                unique_id=uid,
                history=_filter_uid(history, uid),
                horizon=int(req.horizon),
                base_model_config=dict(req.base_model_config),
                search_space=_SEARCH_SPACES[req.search_space_id],
                actuals=_filter_uid(actuals, uid),
                origins=origins,
                objective=_OBJECTIVES[req.objective_id],
                n_trials=int(req.n_trials),
                freq=req.freq,
            )
            candidate = optimize_task_candidate(task)
            payload = _candidate_to_payload(candidate)
            _persist_tuning_run(factory, session_id, uid, payload)
            candidates[uid] = payload
        store.update_study(
            study_id,
            status=RunStatus.SUCCEEDED,
            best_candidates=candidates,
        )
    except Exception as exc:  # pragma: no cover - background task safety net
        store.update_study(study_id, status=RunStatus.FAILED, error=_format_error(exc))


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
    return _json_records(frame)


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
        open_orders=_maybe_json_records(record.last_orders),
    )
