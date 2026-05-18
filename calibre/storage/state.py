from __future__ import annotations

from typing import Protocol
from uuid import UUID

from calibre.storage.postgres import ConformalStateRepo

RUNTIME_PARTITION = "__runtime__"


class ConformalStateStore(Protocol):
    def get(self, run_id: UUID, partition: str = RUNTIME_PARTITION) -> dict | None: ...

    def upsert(self, run_id: UUID, partition: str, state: dict) -> None: ...


class SqlConformalStateStore:
    def __init__(self, repo: ConformalStateRepo, *, commit_on_upsert: bool = True) -> None:
        self.repo = repo
        self.commit_on_upsert = bool(commit_on_upsert)

    def get(self, run_id: UUID, partition: str = RUNTIME_PARTITION) -> dict | None:
        return self.repo.get(run_id, partition)

    def upsert(self, run_id: UUID, partition: str, state: dict) -> None:
        self.repo.upsert(run_id, partition, state)
        if self.commit_on_upsert:
            self.repo.session.commit()
