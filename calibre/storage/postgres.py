from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from calibre.core.run_status import RunStatus
from calibre.storage.models import ConformalState, ForecastPointer, Run


def database_url() -> str | None:
    return os.environ.get("CALIBRE_DATABASE_URL")


def make_engine(url: str, **kwargs) -> Engine:
    return create_engine(url, future=True, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class RunRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, *, config: dict, idempotency_key: str | None = None) -> Run:
        if idempotency_key is not None:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
        run = Run(config=config, idempotency_key=idempotency_key, status=RunStatus.QUEUED.value)
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: UUID) -> Run | None:
        return self.session.get(Run, run_id)

    def get_by_idempotency_key(self, key: str) -> Run | None:
        return self.session.scalar(select(Run).where(Run.idempotency_key == key))

    def set_status(
        self,
        run_id: UUID,
        status: RunStatus | str,
        *,
        error: str | None = None,
    ) -> None:
        parsed_status = RunStatus(status)
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        run.status = parsed_status.value
        run.error = error
        if parsed_status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
            run.finished_at = datetime.now(UTC)
        else:
            run.finished_at = None


class ConformalStateRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, run_id: UUID, partition: str) -> dict | None:
        row = self.session.get(ConformalState, (run_id, partition))
        return dict(row.state) if row is not None else None

    def upsert(self, run_id: UUID, partition: str, state: dict) -> None:
        row = self.session.get(ConformalState, (run_id, partition))
        if row is None:
            self.session.add(ConformalState(run_id=run_id, partition=partition, state=dict(state)))
        else:
            row.state = dict(state)


class ForecastPointerRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, run_id: UUID, kind: str) -> ForecastPointer | None:
        return self.session.get(ForecastPointer, (run_id, kind))

    def list_for_run(self, run_id: UUID) -> list[ForecastPointer]:
        return list(
            self.session.scalars(select(ForecastPointer).where(ForecastPointer.run_id == run_id))
        )

    def upsert(self, run_id: UUID, kind: str, uri: str, byte_size: int) -> None:
        row = self.session.get(ForecastPointer, (run_id, kind))
        if row is None:
            self.session.add(
                ForecastPointer(run_id=run_id, kind=kind, uri=uri, byte_size=byte_size)
            )
        else:
            row.uri = uri
            row.byte_size = int(byte_size)
