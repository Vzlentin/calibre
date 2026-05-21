from __future__ import annotations

from uuid import uuid4

from calibre.storage.models import Base
from calibre.storage.postgres import (
    ConformalStateRepo,
    RunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.storage.session import derive_session_id
from calibre.storage.state import SqlConformalStateStore


def test_same_session_id_resumes_across_runs(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'state.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    session_id = derive_session_id("tenant", ["sku"], {"model": "m"}, {"method": "aci"})

    with session_scope(factory) as session:
        first_run = RunRepo(session).create(config={"run": 1})
        SqlConformalStateStore(
            ConformalStateRepo(session),
            session_id=session_id,
        ).upsert(first_run.id, "sku:model:h1", {"score": 1.0})

    with session_scope(factory) as session:
        second_run = RunRepo(session).create(config={"run": 2})
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=session_id)

        assert store.get(second_run.id, "sku:model:h1") == {"score": 1.0}


def test_different_session_id_starts_fresh(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'state.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    with session_scope(factory) as session:
        run_id = RunRepo(session).create(config={}).id
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=uuid4().hex)
        store.upsert(run_id, "sku:model:h1", {"score": 1.0})

    with session_scope(factory) as session:
        run_id = RunRepo(session).create(config={}).id
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=uuid4().hex)

        assert store.get(run_id, "sku:model:h1") is None
