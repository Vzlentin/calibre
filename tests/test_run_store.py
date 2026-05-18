from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calibre.api.run_store import MemoryRunStore, RunStore, SqlRunStore
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.run_status import RunStatus
from calibre.storage.models import Base
from calibre.storage.postgres import RunRepo, make_engine, make_session_factory, session_scope


def _sqlite_store(tmp_path: Path) -> SqlRunStore:
    db = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'runs.db').as_posix()}")
    Base.metadata.create_all(db)
    return SqlRunStore(make_session_factory(db))


@pytest.fixture(params=["memory", "sql"])
def store(request, tmp_path) -> RunStore:
    if request.param == "memory":
        return MemoryRunStore()
    return _sqlite_store(tmp_path)


def test_run_status_serializes_as_external_json_value() -> None:
    assert str(RunStatus.QUEUED) == "queued"
    assert RunStatus("succeeded") is RunStatus.SUCCEEDED


def test_run_repo_rejects_invalid_status(tmp_path) -> None:
    db = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'invalid.db').as_posix()}")
    Base.metadata.create_all(db)
    factory = make_session_factory(db)

    with session_scope(factory) as session:
        run = RunRepo(session).create(config={"name": "status-test"})
        run_id = run.id
        with pytest.raises(ValueError, match="'not-a-status'"):
            RunRepo(session).set_status(run_id, "not-a-status")


def test_run_store_contract_status_and_idempotency(store: RunStore) -> None:
    first = store.create({"name": "contract"}, idempotency_key="same")
    second = store.create({"name": "contract"}, idempotency_key="same")

    assert first.id == second.id
    assert first.status is RunStatus.QUEUED

    store.mark_running(first.id)
    assert store.get(first.id).status is RunStatus.RUNNING  # type: ignore[union-attr]

    store.mark_failed(first.id, error="boom")
    failed = store.create({"name": "contract"}, idempotency_key="same")
    assert failed.id == first.id
    assert failed.status is RunStatus.FAILED

    queued = store.queue(first.id)
    assert queued.status is RunStatus.QUEUED
    assert queued.error is None

    store.mark_succeeded(first.id, rows=7)
    succeeded = store.get(first.id)
    assert succeeded is not None
    assert succeeded.status is RunStatus.SUCCEEDED
    assert succeeded.artifact_urls["rows"] == "7"


def test_sql_run_store_records_artifacts_and_reads_initial_ledger(tmp_path) -> None:
    store = _sqlite_store(tmp_path)
    ledger_path = tmp_path / "ledger.parquet"
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-07")],
            Y: [1.0],
            Y_HAT: [1.5],
            H: [1],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")],
            MODEL_NAME: ["stub"],
        }
    )
    frame.to_parquet(ledger_path)
    config = type(
        "Config",
        (),
        {
            "output": type(
                "Output",
                (),
                {"ledger_path": ledger_path.as_posix(), "order_ledger_path": None},
            )()
        },
    )()

    run = store.create({"name": "artifact"})
    store.register_configured_artifact_pointers(run.id, config)
    store.record_artifact_pointers(run.id, config)

    snapshot = store.get(run.id)
    assert snapshot is not None
    assert Path(snapshot.artifact_urls["ledger"]) == ledger_path
    pd.testing.assert_frame_equal(store.initial_ledger(run.id), frame)
