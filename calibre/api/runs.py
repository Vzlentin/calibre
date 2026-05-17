from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy.orm import sessionmaker

from calibre.api.schemas import RunResponse
from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.storage.objstore import artifact_pointer, read_initial_ledger, signed_url
from calibre.storage.postgres import (
    ConformalStateRepo,
    ForecastPointerRepo,
    RunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.storage.state import SqlConformalStateStore


@dataclass
class RunRecord:
    id: str
    config: dict
    status: str = "queued"
    artifact_urls: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def response(self) -> RunResponse:
        return RunResponse(
            id=self.id,
            status=self.status,  # type: ignore[arg-type]
            artifact_urls=dict(self.artifact_urls),
            error=self.error,
        )


_RUNS: dict[str, RunRecord] = {}
_IDEMPOTENCY: dict[str, str] = {}
_DB_URL: str | None = None
_DB_FACTORY: sessionmaker | None = None


def _database_url() -> str | None:
    return os.environ.get("CALIBRE_DATABASE_URL")


def _session_factory() -> sessionmaker | None:
    global _DB_FACTORY, _DB_URL
    url = _database_url()
    if not url:
        return None
    if _DB_FACTORY is None or url != _DB_URL:
        _DB_URL = url
        _DB_FACTORY = make_session_factory(make_engine(url))
    return _DB_FACTORY


def _db_response(run, pointer_repo: ForecastPointerRepo) -> RunResponse:
    artifact_urls = {
        pointer.kind: signed_url(pointer.uri) for pointer in pointer_repo.list_for_run(run.id)
    }
    if "_rows" in run.config:
        artifact_urls["rows"] = str(run.config["_rows"])
    return RunResponse(
        id=str(run.id),
        status=run.status,  # type: ignore[arg-type]
        artifact_urls=artifact_urls,
        error=run.error,
    )


def create_run(config: dict, *, idempotency_key: str | None = None) -> RunResponse:
    factory = _session_factory()
    if factory is not None:
        with session_scope(factory) as session:
            run = RunRepo(session).create(config=config, idempotency_key=idempotency_key)
            return _db_response(run, ForecastPointerRepo(session))

    if idempotency_key is not None and idempotency_key in _IDEMPOTENCY:
        return _RUNS[_IDEMPOTENCY[idempotency_key]].response()
    run_id = str(uuid4())
    record = RunRecord(id=run_id, config=config)
    _RUNS[run_id] = record
    if idempotency_key is not None:
        _IDEMPOTENCY[idempotency_key] = run_id
    return record.response()


def get_run(run_id: str) -> RunResponse | None:
    factory = _session_factory()
    if factory is not None:
        with session_scope(factory) as session:
            try:
                parsed_run_id = UUID(run_id)
            except ValueError:
                return None
            run = RunRepo(session).get(parsed_run_id)
            if run is None:
                return None
            return _db_response(run, ForecastPointerRepo(session))

    record = _RUNS.get(run_id)
    return record.response() if record is not None else None


def run_backtest_job(run_id: str) -> None:
    factory = _session_factory()
    if factory is not None:
        _run_backtest_job_db(factory, UUID(run_id))
        return

    record = _RUNS[run_id]
    record.status = "running"
    try:
        config = load_config_from_mapping(record.config)
        result = run_config(config)
        if isinstance(result, pd.DataFrame):
            rows = len(result)
        else:
            rows = len(result.ledger.to_df())
            if config.output.ledger_path is not None:
                record.artifact_urls["ledger"] = signed_url(config.output.ledger_path)
            if config.output.order_ledger_path is not None:
                record.artifact_urls["order_ledger"] = signed_url(config.output.order_ledger_path)
        record.artifact_urls["rows"] = str(rows)
        record.status = "succeeded"
    except Exception as exc:  # pragma: no cover - exercised through API test as status
        record.status = "failed"
        record.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _record_artifact_pointers(
    pointer_repo: ForecastPointerRepo,
    run_id: UUID,
    config,
) -> None:
    if config.output.ledger_path is not None:
        try:
            pointer = artifact_pointer(config.output.ledger_path)
        except FileNotFoundError:
            pass
        else:
            pointer_repo.upsert(run_id, "ledger", str(pointer["uri"]), int(pointer["byte_size"]))
    if config.output.order_ledger_path is not None:
        try:
            pointer = artifact_pointer(config.output.order_ledger_path)
        except FileNotFoundError:
            pass
        else:
            pointer_repo.upsert(
                run_id,
                "order_ledger",
                str(pointer["uri"]),
                int(pointer["byte_size"]),
            )


def _run_backtest_job_db(factory: sessionmaker, run_id: UUID) -> None:
    try:
        with session_scope(factory) as session:
            run_repo = RunRepo(session)
            run = run_repo.get(run_id)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            run_repo.set_status(run_id, "running")
            session.commit()
            config = load_config_from_mapping(run.config)
            pointer_repo = ForecastPointerRepo(session)
            initial_ledger = read_initial_ledger(pointer_repo, run_id)
            result = run_config(
                config,
                run_id=run_id,
                conformal_state_store=SqlConformalStateStore(ConformalStateRepo(session)),
                initial_ledger=initial_ledger,
            )
            if isinstance(result, pd.DataFrame):
                run.config = {**run.config, "_rows": len(result)}
            else:
                run.config = {**run.config, "_rows": len(result.ledger.to_df())}
            _record_artifact_pointers(pointer_repo, run_id, config)
            run_repo.set_status(run_id, "succeeded")
    except Exception as exc:  # pragma: no cover - status path exercised by tests
        with session_scope(factory) as session:
            RunRepo(session).set_status(
                run_id,
                "failed",
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
