from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_storage_alembic_upgrade_creates_run_tables(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "storage.db"
    db_url = f"sqlite+pysqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("CALIBRE_DATABASE_URL", db_url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    assert {
        "runs",
        "conformal_state",
        "forecast_pointers",
        "pending_observations",
        "tuning_runs",
        "fit_records",
        "tune_records",
        "alembic_version",
    }.issubset(inspector.get_table_names())
    conformal_pk = inspector.get_pk_constraint("conformal_state")["constrained_columns"]
    assert conformal_pk == ["session_id", "partition"]
    pending_pk = inspector.get_pk_constraint("pending_observations")["constrained_columns"]
    assert pending_pk == ["session_id", "uid", "model_name", "origin", "h"]
    tuning_pk = inspector.get_pk_constraint("tuning_runs")["constrained_columns"]
    assert tuning_pk == ["session_id", "unique_id"]
    fit_pk = inspector.get_pk_constraint("fit_records")["constrained_columns"]
    assert fit_pk == ["fit_id"]
    tune_pk = inspector.get_pk_constraint("tune_records")["constrained_columns"]
    assert tune_pk == ["study_id"]
    with engine.connect() as connection:
        revision = connection.execute(text("select version_num from alembic_version")).scalar_one()
    assert revision == "0005_lifecycle_store"
