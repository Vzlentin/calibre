from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from calibre.api.run_store import MemoryRunStore, RunSnapshot, RunStore, SqlRunStore
from calibre.api.schemas import RunResponse
from calibre.storage.postgres import database_url, make_engine, make_session_factory

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


def _response(snapshot: RunSnapshot) -> RunResponse:
    return RunResponse(
        id=snapshot.id,
        status=snapshot.status,
        artifact_urls=dict(snapshot.artifact_urls),
        error=snapshot.error,
    )


def create_run(config: dict, *, idempotency_key: str | None = None) -> RunResponse:
    return _response(_run_store().create(config=config, idempotency_key=idempotency_key))


def get_run(run_id: str) -> RunResponse | None:
    run = _run_store().get(run_id)
    return _response(run) if run is not None else None


def queue_run(run_id: str) -> RunResponse:
    return _response(_run_store().queue(run_id))


def run_backtest_job(run_id: str) -> None:
    _run_store().run_backtest_job(run_id)
