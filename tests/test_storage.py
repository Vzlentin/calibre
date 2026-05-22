from __future__ import annotations

import fsspec
import pandas as pd

from calibre.storage.models import Base
from calibre.storage.objstore import read_run_artifacts, signed_url, write_ledger_shard


def test_storage_metadata_contains_plan_tables() -> None:
    assert {
        "runs",
        "conformal_state",
        "forecast_pointers",
        "pending_observations",
    }.issubset(Base.metadata.tables)


def test_objstore_round_trips_ledger_shard(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    frame = pd.DataFrame({"unique_id": ["A"], "value": [1.0]})

    pointer = write_ledger_shard(frame, str(path))
    loaded = read_run_artifacts({"ledger": str(path)})

    assert pointer["byte_size"] > 0
    pd.testing.assert_frame_equal(loaded["ledger"], frame)
    assert signed_url(str(path)) == str(path)


def test_objstore_writes_ledger_shard_to_fsspec_uri() -> None:
    uri = "memory://calibre-storage-test/artifacts/ledger.parquet"
    fs = fsspec.filesystem("memory")
    if fs.exists("calibre-storage-test"):
        fs.rm("calibre-storage-test", recursive=True)
    frame = pd.DataFrame({"unique_id": ["A"], "value": [1.0]})

    pointer = write_ledger_shard(frame, uri)
    loaded = read_run_artifacts({"ledger": uri})

    assert pointer["byte_size"] > 0
    pd.testing.assert_frame_equal(loaded["ledger"], frame)
