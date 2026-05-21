from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from sqlalchemy import delete

from calibre.storage.postgres import ConformalStateRepo
from calibre.storage.session import legacy_session_id

RUNTIME_PARTITION = "__runtime__"


class ConformalStateStore(Protocol):
    def get(self, run_id: UUID, partition: str = RUNTIME_PARTITION) -> dict | None: ...

    def list_for_run(self, run_id: UUID) -> dict[str, dict]: ...

    def upsert(self, run_id: UUID, partition: str, state: dict) -> None: ...


class SqlConformalStateStore:
    def __init__(
        self,
        repo: ConformalStateRepo,
        *,
        session_id: str | None = None,
        commit_on_upsert: bool = True,
    ) -> None:
        self.repo = repo
        self.session_id = session_id
        self.commit_on_upsert = bool(commit_on_upsert)

    def get(self, run_id: UUID, partition: str = RUNTIME_PARTITION) -> dict | None:
        return self.repo.get(self._session_id(run_id), partition)

    def list_for_run(self, run_id: UUID) -> dict[str, dict]:
        return self.repo.list_for_session(self._session_id(run_id))

    def upsert(self, run_id: UUID, partition: str, state: dict) -> None:
        self.repo.upsert(self._session_id(run_id), partition, state, run_id=run_id)
        if self.commit_on_upsert:
            self.repo.session.commit()

    def _session_id(self, run_id: UUID) -> str:
        return self.session_id or legacy_session_id(run_id)


def compact_old_state(session_id: str, older_than_days: int) -> int:
    from calibre.storage.models import ConformalState
    from calibre.storage.postgres import database_url, make_engine, make_session_factory

    if int(older_than_days) < 0:
        raise ValueError("older_than_days must be non-negative")
    url = database_url()
    if url is None:
        raise RuntimeError("Set CALIBRE_DATABASE_URL before compacting conformal state")

    cutoff = datetime.now(UTC) - timedelta(days=int(older_than_days))
    engine = make_engine(url)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        result = session.execute(
            delete(ConformalState).where(
                ConformalState.session_id == session_id,
                ConformalState.updated_at < cutoff,
            )
        )
        session.commit()
        return int(result.rowcount or 0)
