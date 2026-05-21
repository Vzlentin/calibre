from __future__ import annotations

from calibre.storage.models import Base
from calibre.storage.postgres import (
    ConformalStateRepo,
    RunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.storage.state import SqlConformalStateStore


def test_partitions_round_trip_independently(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'state.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    session_id = "session"

    with session_scope(factory) as session:
        run_id = RunRepo(session).create(config={}).id
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=session_id)
        store.upsert(run_id, "sku:model:h1", {"score": 1.0})
        store.upsert(run_id, "sku:model:h2", {"score": 2.0})

    with session_scope(factory) as session:
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=session_id)

        assert store.get(run_id, "sku:model:h1") == {"score": 1.0}
        assert store.get(run_id, "sku:model:h2") == {"score": 2.0}


def test_no_single_blob_collision(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'state.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    with session_scope(factory) as session:
        run_id = RunRepo(session).create(config={}).id
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id="session")
        store.upsert(run_id, "model_a:h1", {"score": 1.0})
        store.upsert(run_id, "model_b:h1", {"score": 2.0})

    with session_scope(factory) as session:
        states = SqlConformalStateStore(
            ConformalStateRepo(session),
            session_id="session",
        ).list_for_run(run_id)

    assert states == {
        "model_a:h1": {"score": 1.0},
        "model_b:h1": {"score": 2.0},
    }
