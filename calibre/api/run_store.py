from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy.orm import sessionmaker

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


@dataclass
class RunSnapshot:
    id: str
    config: dict
    status: RunStatus = RunStatus.QUEUED
    artifact_urls: dict[str, str] = field(default_factory=dict)
    error: str | None = None


class RunStore(Protocol):
    def create(self, config: dict, *, idempotency_key: str | None = None) -> RunSnapshot: ...

    def get(self, run_id: str) -> RunSnapshot | None: ...

    def queue(self, run_id: str) -> RunSnapshot: ...

    def mark_running(self, run_id: str) -> None: ...

    def mark_succeeded(self, run_id: str, *, rows: int) -> None: ...

    def mark_failed(self, run_id: str, *, error: str) -> None: ...

    def register_configured_artifact_pointers(self, run_id: str, config) -> None: ...

    def record_artifact_pointers(self, run_id: str, config) -> None: ...

    def initial_ledger(self, run_id: str) -> pd.DataFrame | None: ...

    def run_backtest_job(self, run_id: str) -> None: ...


def _pointer_kinds(config) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if config.output.ledger_path is not None:
        pairs.append(("ledger", config.output.ledger_path))
    if config.output.order_ledger_path is not None:
        pairs.append(("order_ledger", config.output.order_ledger_path))
    return pairs


def _format_error(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


class MemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunSnapshot] = {}
        self._idempotency: dict[str, str] = {}

    def create(self, config: dict, *, idempotency_key: str | None = None) -> RunSnapshot:
        if idempotency_key is not None and idempotency_key in self._idempotency:
            return self._runs[self._idempotency[idempotency_key]]
        run_id = str(uuid4())
        record = RunSnapshot(id=run_id, config=config)
        self._runs[run_id] = record
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = run_id
        return record

    def get(self, run_id: str) -> RunSnapshot | None:
        return self._runs.get(run_id)

    def queue(self, run_id: str) -> RunSnapshot:
        record = self._runs[run_id]
        record.status = RunStatus.QUEUED
        record.error = None
        return record

    def mark_running(self, run_id: str) -> None:
        self._runs[run_id].status = RunStatus.RUNNING

    def mark_succeeded(self, run_id: str, *, rows: int) -> None:
        record = self._runs[run_id]
        record.artifact_urls["rows"] = str(rows)
        record.status = RunStatus.SUCCEEDED
        record.error = None

    def mark_failed(self, run_id: str, *, error: str) -> None:
        record = self._runs[run_id]
        record.status = RunStatus.FAILED
        record.error = error

    def register_configured_artifact_pointers(self, run_id: str, config) -> None:
        del run_id, config

    def record_artifact_pointers(self, run_id: str, config) -> None:
        record = self._runs[run_id]
        for kind, path in _pointer_kinds(config):
            uri = canonical_ledger_uri(path) if kind == "ledger" else path
            record.artifact_urls[kind] = signed_url(uri)

    def initial_ledger(self, run_id: str) -> pd.DataFrame | None:
        del run_id
        return None

    def run_backtest_job(self, run_id: str) -> None:
        self.mark_running(run_id)
        try:
            record = self._runs[run_id]
            config = load_config_from_mapping(record.config)
            result = run_config(config)
            rows = len(result) if isinstance(result, pd.DataFrame) else len(result.ledger.to_df())
            self.record_artifact_pointers(run_id, config)
            self.mark_succeeded(run_id, rows=rows)
        except Exception as exc:  # pragma: no cover - exercised through API status tests
            self.mark_failed(run_id, error=_format_error(exc))


class SqlRunStore(RunStore):
    def __init__(self, factory: sessionmaker) -> None:
        self.factory = factory

    def create(self, config: dict, *, idempotency_key: str | None = None) -> RunSnapshot:
        with session_scope(self.factory) as session:
            run = RunRepo(session).create(config=config, idempotency_key=idempotency_key)
            return self._snapshot(run, ForecastPointerRepo(session))

    def get(self, run_id: str) -> RunSnapshot | None:
        parsed = self._parse_run_id(run_id)
        if parsed is None:
            return None
        with session_scope(self.factory) as session:
            run = RunRepo(session).get(parsed)
            if run is None:
                return None
            return self._snapshot(run, ForecastPointerRepo(session))

    def queue(self, run_id: str) -> RunSnapshot:
        parsed = UUID(run_id)
        with session_scope(self.factory) as session:
            repo = RunRepo(session)
            repo.set_status(parsed, RunStatus.QUEUED)
            run = repo.get(parsed)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            return self._snapshot(run, ForecastPointerRepo(session))

    def mark_running(self, run_id: str) -> None:
        with session_scope(self.factory) as session:
            RunRepo(session).set_status(UUID(run_id), RunStatus.RUNNING)

    def mark_succeeded(self, run_id: str, *, rows: int) -> None:
        with session_scope(self.factory) as session:
            repo = RunRepo(session)
            run = repo.get(UUID(run_id))
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            run.config = {**run.config, "_rows": int(rows)}
            repo.set_status(UUID(run_id), RunStatus.SUCCEEDED)

    def mark_failed(self, run_id: str, *, error: str) -> None:
        with session_scope(self.factory) as session:
            RunRepo(session).set_status(UUID(run_id), RunStatus.FAILED, error=error)

    def register_configured_artifact_pointers(self, run_id: str, config) -> None:
        parsed = UUID(run_id)
        with session_scope(self.factory) as session:
            pointer_repo = ForecastPointerRepo(session)
            for kind, path in _pointer_kinds(config):
                if pointer_repo.get(parsed, kind) is None:
                    pointer_repo.upsert(parsed, kind, path, 0)

    def record_artifact_pointers(self, run_id: str, config) -> None:
        parsed = UUID(run_id)
        with session_scope(self.factory) as session:
            pointer_repo = ForecastPointerRepo(session)
            for kind, path in _pointer_kinds(config):
                uri = canonical_ledger_uri(path) if kind == "ledger" else path
                try:
                    pointer = artifact_pointer(uri)
                except FileNotFoundError:
                    continue
                pointer_repo.upsert(parsed, kind, str(pointer["uri"]), int(pointer["byte_size"]))

    def initial_ledger(self, run_id: str) -> pd.DataFrame | None:
        with session_scope(self.factory) as session:
            return read_initial_ledger(ForecastPointerRepo(session), UUID(run_id))

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
                rows = (
                    len(result) if isinstance(result, pd.DataFrame) else len(result.ledger.to_df())
                )
                run.config = {**run.config, "_rows": int(rows)}
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
        except Exception as exc:  # pragma: no cover - status path exercised by tests
            self.mark_failed(run_id, error=_format_error(exc))

    @staticmethod
    def _parse_run_id(run_id: str) -> UUID | None:
        try:
            return UUID(run_id)
        except ValueError:
            return None

    @staticmethod
    def _snapshot(run, pointer_repo: ForecastPointerRepo) -> RunSnapshot:
        artifact_urls = {
            pointer.kind: signed_url(pointer.uri) for pointer in pointer_repo.list_for_run(run.id)
        }
        if "_rows" in run.config:
            artifact_urls["rows"] = str(run.config["_rows"])
        return RunSnapshot(
            id=str(run.id),
            config=run.config,
            status=RunStatus(run.status),
            artifact_urls=artifact_urls,
            error=run.error,
        )
