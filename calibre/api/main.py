from __future__ import annotations

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import sessionmaker

from calibre.api.run_store import MemoryRunStore, RunStore, SqlRunStore
from calibre.api.schemas import ForecastRequest, ForecastResponse, RunResponse
from calibre.cli.commands import run_config
from calibre.core.run_status import RunStatus
from calibre.storage.postgres import database_url, make_engine, make_session_factory

app = FastAPI(title="Calibre", version="0.1.0")
MAX_FORECAST_UNIQUE_IDS = 30

_MEMORY_STORE = MemoryRunStore()
_DB_URL: str | None = None
_DB_FACTORY: sessionmaker | None = None
_SQL_STORE: SqlRunStore | None = None


def _run_store() -> RunStore:
    global _DB_FACTORY, _DB_URL, _SQL_STORE
    url = database_url()
    if not url:
        return _MEMORY_STORE
    if _DB_FACTORY is None or _SQL_STORE is None or url != _DB_URL:
        _DB_URL = url
        _DB_FACTORY = make_session_factory(make_engine(url))
        _SQL_STORE = SqlRunStore(_DB_FACTORY)
    return _SQL_STORE


def _json_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    clean = clean.astype(object)
    clean[pd.isna(clean)] = None
    return clean.to_dict(orient="records")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forecasts", response_model=ForecastResponse)
def forecasts(req: ForecastRequest) -> ForecastResponse:
    try:
        config = req.as_backend_config()
        result = run_config(config, max_unique_ids=MAX_FORECAST_UNIQUE_IDS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    frame = result if isinstance(result, pd.DataFrame) else result.ledger.to_df()
    return ForecastResponse(rows=len(frame), forecasts=_json_records(frame))


@app.post("/backtests", response_model=RunResponse, status_code=202)
def backtests(
    req: ForecastRequest,
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
