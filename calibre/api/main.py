from __future__ import annotations

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from calibre.api.runs import create_run, get_run, run_backtest_job
from calibre.api.schemas import ForecastRequest, ForecastResponse, RunResponse
from calibre.cli.commands import run_config

app = FastAPI(title="Calibre", version="0.1.0")
MAX_FORECAST_UNIQUE_IDS = 30


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
    run = create_run(req.config, idempotency_key=idempotency_key)
    if run.status == "queued":
        bg.add_task(run_backtest_job, run.id)
    return run


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run_status(run_id: str) -> RunResponse:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run
