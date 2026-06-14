"""Tests for the Alembic storage migrations."""

from __future__ import annotations

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from calibre.storage.models import Base

# Diff operations that represent a real divergence between the schema produced
# by ``alembic upgrade head`` and the ORM the application actually queries: a
# table or column the ORM needs that no migration creates (or one a migration
# creates that the ORM no longer declares). This is exactly the failure class
# that shipped undetected in PR #38 — the ORM expected ``*_ref`` columns and a
# ``fit_frame_artifacts`` table that no migration ever created, yet CI was green
# because tests built the schema with ``Base.metadata.create_all()`` instead of
# the migrations. We deliberately ignore type / nullable / default / index
# modifications, which are noisy and dialect-dependent under SQLite and are not
# what broke #38.
_STRUCTURAL_OPS = {"add_table", "remove_table", "add_column", "remove_column"}


def _structural_diffs(diffs: list) -> list:
    out = []
    for diff in diffs:
        # compare_metadata yields either a single ``(op, ...)`` tuple or a list
        # of such tuples (grouped per-table column diffs).
        entries = diff if isinstance(diff, list) else [diff]
        for entry in entries:
            if entry and entry[0] in _STRUCTURAL_OPS:
                out.append(entry)
    return out


def _render(entry: tuple) -> str:
    op = entry[0]
    if op in {"add_table", "remove_table"}:
        return f"{op}: {entry[1].name}"
    if op in {"add_column", "remove_column"}:
        # ('add_column'|'remove_column', schema, table_name, Column)
        return f"{op}: {entry[2]}.{entry[3].name}"
    return repr(entry)


def test_revision_ids_fit_alembic_version_column() -> None:
    """Revision ids must fit Alembic's ``alembic_version.version_num``.

    That column is ``VARCHAR(32)`` by default. SQLite does not enforce VARCHAR
    length, so an over-length id (e.g. the 33-char ``0003_pending_observation_
    metadata``) passed locally and in every prior CI run but truncated and
    errored on the first real Postgres ``upgrade head``. This guard fails fast
    with a clear message instead of a cryptic ``StringDataRightTruncation``.
    """
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    too_long = {
        rev.revision: len(rev.revision) for rev in script.walk_revisions() if len(rev.revision) > 32
    }
    assert not too_long, f"revision ids exceed alembic_version VARCHAR(32): {too_long}"


def test_migrations_match_orm(fresh_db_url, monkeypatch) -> None:
    """``alembic upgrade head`` must produce a schema that matches the ORM.

    This is the gate that would have caught PR #38's blocker. It runs the real
    migration path (not ``create_all()``), then reflects the live schema and
    diffs it against ``Base.metadata``. Any missing/extra table or column fails.
    """
    monkeypatch.setenv("CALIBRE_DATABASE_URL", fresh_db_url)
    command.upgrade(Config("alembic.ini"), "head")

    engine = create_engine(fresh_db_url, future=True)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            diffs = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    structural = _structural_diffs(diffs)
    assert not structural, "migration↔ORM schema drift detected:\n" + "\n".join(
        _render(e) for e in structural
    )


def test_lifecycle_state_round_trip_on_migrated_db(fresh_db_url, monkeypatch) -> None:
    """The ORM repositories must read/write against the migration-built schema.

    Parity (above) proves the columns line up; this proves the ORM can actually
    execute a write/read round-trip on the schema that ``alembic upgrade head``
    produces — the path a real deployment takes, never exercised by
    ``create_all()`` in the rest of the suite.
    """
    monkeypatch.setenv("CALIBRE_DATABASE_URL", fresh_db_url)
    command.upgrade(Config("alembic.ini"), "head")

    from calibre.storage.postgres import (
        ConformalStateRepo,
        RunRepo,
        make_engine,
        make_session_factory,
        session_scope,
    )

    factory = make_session_factory(make_engine(fresh_db_url))
    with session_scope(factory) as session:
        run = RunRepo(session).create(config={"k": "v"})
        run_id = run.id
        ConformalStateRepo(session).upsert("sess-1", "uid=A", {"alpha": 0.1}, run_id=run_id)

    with session_scope(factory) as session:
        state = ConformalStateRepo(session).get("sess-1", "uid=A")
    assert state == {"alpha": 0.1}


def test_storage_alembic_upgrade_creates_run_tables(fresh_db_url, monkeypatch) -> None:
    monkeypatch.setenv("CALIBRE_DATABASE_URL", fresh_db_url)
    command.upgrade(Config("alembic.ini"), "head")

    engine = create_engine(fresh_db_url, future=True)
    inspector = inspect(engine)
    assert {
        "runs",
        "conformal_state",
        "forecast_pointers",
        "pending_observations",
        "tuning_runs",
        "orders",
        "sales",
        "alembic_version",
    }.issubset(inspector.get_table_names())
    conformal_pk = inspector.get_pk_constraint("conformal_state")["constrained_columns"]
    assert conformal_pk == ["session_id", "partition"]
    pending_pk = inspector.get_pk_constraint("pending_observations")["constrained_columns"]
    assert pending_pk == ["session_id", "uid", "model_name", "origin", "h"]
    tuning_pk = inspector.get_pk_constraint("tuning_runs")["constrained_columns"]
    assert tuning_pk == ["session_id", "unique_id"]
    orders_pk = inspector.get_pk_constraint("orders")["constrained_columns"]
    assert orders_pk == ["session_id", "unique_id", "forecast_origin", "model_name"]
    sales_pk = inspector.get_pk_constraint("sales")["constrained_columns"]
    assert sales_pk == ["unique_id", "ds"]
    with engine.connect() as connection:
        revision = connection.execute(text("select version_num from alembic_version")).scalar_one()
    head = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
    assert revision == head
    engine.dispose()
