from __future__ import annotations

from typing import Protocol
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy.orm import sessionmaker

from calibre.api.errors import format_error
from calibre.api.schemas import RunResponse
from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.core.run_status import RunStatus
from calibre.storage.objstore import (
    artifact_pointer,
    canonical_ledger_uri,
    read_initial_ledger,
    signed_url,
)
from calibre.storage.postgres import ConformalStateRepo, ForecastPointerRepo, RunRepo, session_scope
from calibre.storage.state import SqlConformalStateStore


class RunStore(Protocol):
    def create(self, config: dict, *, idempotency_key: str | None = None) -> RunResponse: ...

    def get(self, run_id: str) -> RunResponse | None: ...

    def queue(self, run_id: str) -> RunResponse: ...

    def run_backtest_job(self, run_id: str) -> None: ...


def _pointer_kinds(config) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if config.output.ledger_path is not None:
        pairs.append(("ledger", config.output.ledger_path))
    if config.output.order_ledger_path is not None:
        pairs.append(("order_ledger", config.output.order_ledger_path))
    return pairs


def _result_rows(result) -> int:
    return len(result) if isinstance(result, pd.DataFrame) else len(result.ledger.to_df())


class MemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunResponse] = {}
        self._configs: dict[str, dict] = {}
        self._idempotency: dict[str, str] = {}

    def create(self, config: dict, *, idempotency_key: str | None = None) -> RunResponse:
        if idempotency_key is not None and idempotency_key in self._idempotency:
            return self._runs[self._idempotency[idempotency_key]]
        run_id = str(uuid4())
        response = RunResponse(
            id=run_id,
            status=RunStatus.QUEUED,
        )
        self._runs[run_id] = response
        self._configs[run_id] = config
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = run_id
        return response

    def get(self, run_id: str) -> RunResponse | None:
        return self._runs.get(run_id)

    def queue(self, run_id: str) -> RunResponse:
        response = self._runs[run_id].model_copy(update={"status": RunStatus.QUEUED, "error": None})
        self._runs[run_id] = response
        return response

    def mark_running(self, run_id: str) -> None:
        self._runs[run_id] = self._runs[run_id].model_copy(update={"status": RunStatus.RUNNING})

    def mark_succeeded(self, run_id: str, *, rows: int) -> None:
        self._runs[run_id] = self._runs[run_id].model_copy(
            update={"status": RunStatus.SUCCEEDED, "row_count": int(rows), "error": None}
        )

    def mark_failed(self, run_id: str, *, error: str) -> None:
        self._runs[run_id] = self._runs[run_id].model_copy(
            update={"status": RunStatus.FAILED, "error": error}
        )

    def _record_artifact_urls(self, run_id: str, config) -> None:
        urls = dict(self._runs[run_id].artifact_urls)
        for kind, path in _pointer_kinds(config):
            uri = canonical_ledger_uri(path) if kind == "ledger" else path
            urls[kind] = signed_url(uri)
        self._runs[run_id] = self._runs[run_id].model_copy(update={"artifact_urls": urls})

    def run_backtest_job(self, run_id: str) -> None:
        self.mark_running(run_id)
        try:
            config = load_config_from_mapping(self._configs[run_id])
            result = run_config(config)
            self._record_artifact_urls(run_id, config)
            self.mark_succeeded(run_id, rows=_result_rows(result))
        except Exception as exc:
            self.mark_failed(run_id, error=format_error(exc))


class SqlRunStore:
    def __init__(self, factory: sessionmaker) -> None:
        self.factory = factory

    def create(self, config: dict, *, idempotency_key: str | None = None) -> RunResponse:
        with session_scope(self.factory) as session:
            run = RunRepo(session).create(config=config, idempotency_key=idempotency_key)
            return self._response(run, ForecastPointerRepo(session))

    def get(self, run_id: str) -> RunResponse | None:
        try:
            parsed = UUID(run_id)
        except ValueError:
            return None
        with session_scope(self.factory) as session:
            run = RunRepo(session).get(parsed)
            if run is None:
                return None
            return self._response(run, ForecastPointerRepo(session))

    def queue(self, run_id: str) -> RunResponse:
        parsed = UUID(run_id)
        with session_scope(self.factory) as session:
            repo = RunRepo(session)
            repo.set_status(parsed, RunStatus.QUEUED)
            run = repo.get(parsed)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            return self._response(run, ForecastPointerRepo(session))

    def mark_running(self, run_id: str) -> None:
        with session_scope(self.factory) as session:
            RunRepo(session).set_status(UUID(run_id), RunStatus.RUNNING)

    def mark_succeeded(self, run_id: str, *, rows: int) -> None:
        parsed = UUID(run_id)
        with session_scope(self.factory) as session:
            repo = RunRepo(session)
            run = repo.get(parsed)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            run.row_count = int(rows)
            repo.set_status(parsed, RunStatus.SUCCEEDED)

    def mark_failed(self, run_id: str, *, error: str) -> None:
        with session_scope(self.factory) as session:
            RunRepo(session).set_status(UUID(run_id), RunStatus.FAILED, error=error)

    def run_backtest_job(self, run_id: str) -> None:
        parsed = UUID(run_id)
        try:
            with session_scope(self.factory) as session:
                run_repo = RunRepo(session)
                run = run_repo.get(parsed)
                if run is None:
                    raise KeyError(f"Unknown run_id: {run_id}")
                run_repo.set_status(parsed, RunStatus.RUNNING)
                session.commit()
                config = load_config_from_mapping(run.config)
                pointer_repo = ForecastPointerRepo(session)
                for kind, path in _pointer_kinds(config):
                    if pointer_repo.get(parsed, kind) is None:
                        pointer_repo.upsert(parsed, kind, path, 0)
                session.commit()
                initial_ledger = read_initial_ledger(pointer_repo, parsed)
                result = run_config(
                    config,
                    run_id=parsed,
                    conformal_state_store=SqlConformalStateStore(ConformalStateRepo(session)),
                    initial_ledger=initial_ledger,
                )
                run.row_count = _result_rows(result)
                for kind, path in _pointer_kinds(config):
                    uri = canonical_ledger_uri(path) if kind == "ledger" else path
                    try:
                        pointer = artifact_pointer(uri)
                    except FileNotFoundError:
                        continue
                    pointer_repo.upsert(
                        parsed,
                        kind,
                        str(pointer["uri"]),
                        int(pointer["byte_size"]),
                    )
                run_repo.set_status(parsed, RunStatus.SUCCEEDED)
        except Exception as exc:
            self.mark_failed(run_id, error=format_error(exc))

    @staticmethod
    def _response(run, pointer_repo: ForecastPointerRepo) -> RunResponse:
        artifact_urls = {
            pointer.kind: signed_url(pointer.uri) for pointer in pointer_repo.list_for_run(run.id)
        }
        return RunResponse(
            id=str(run.id),
            status=RunStatus(run.status),
            artifact_urls=artifact_urls,
            row_count=run.row_count,
            error=run.error,
        )
