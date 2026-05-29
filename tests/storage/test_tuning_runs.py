from __future__ import annotations

from calibre.storage.models import Base
from calibre.storage.postgres import (
    TuningRunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)


def test_tuning_run_upsert_round_trip(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'tuning.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            "session_a",
            "sku_1",
            config_signature="sig_a",
            candidate={"model_config": {"season_length": 13}},
            score=4.2,
        )
        TuningRunRepo(session).upsert(
            "session_a",
            "sku_2",
            config_signature="sig_a",
            candidate={"model_config": {"season_length": 26}},
            score=3.1,
        )

    with session_scope(factory) as session:
        rows = TuningRunRepo(session).list_for_session("session_a")
        by_uid = {row.unique_id: row for row in rows}
        assert set(by_uid) == {"sku_1", "sku_2"}
        assert by_uid["sku_1"].candidate == {"model_config": {"season_length": 13}}
        assert by_uid["sku_1"].score == 4.2
        assert by_uid["sku_2"].score == 3.1


def test_tuning_run_upsert_overrides_existing(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'tuning.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            "session_b",
            "sku_1",
            config_signature="sig_b",
            candidate={"model_config": {"season_length": 13}},
            score=4.2,
        )
    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            "session_b",
            "sku_1",
            config_signature="sig_b",
            candidate={"model_config": {"season_length": 52}},
            score=2.0,
        )

    with session_scope(factory) as session:
        row = TuningRunRepo(session).get("session_b", "sku_1")
        assert row is not None
        assert row.candidate == {"model_config": {"season_length": 52}}
        assert row.score == 2.0


def test_tuning_run_finished_at_updates(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'tuning.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    def _naive(dt):
        return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            "session_c", "sku_1", config_signature="sig_c", candidate={"k": "v"}, score=None
        )
        row = TuningRunRepo(session).get("session_c", "sku_1")
        assert row is not None
        first_finished = _naive(row.finished_at)

    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            "session_c", "sku_1", config_signature="sig_c", candidate={"k": "v2"}, score=None
        )
        row = TuningRunRepo(session).get("session_c", "sku_1")
        assert row is not None
        second_finished = _naive(row.finished_at)

    assert second_finished >= first_finished
