#!/usr/bin/env python3
"""Databricks smoke test for Calibre.

Run inside a Databricks notebook cell or as a Databricks job.

Prerequisites:
- %pip install /dbfs/mnt/calibre/calibre-0.1.0-py3-none-any.whl
- VN2 fixture data copied to /dbfs/mnt/calibre/vn2-fixture/

Expected result:
    Rows: 4
    Ledger columns: [...]
    Spark engine: OK
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Detect Databricks environment
IN_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ


def _log(msg: str) -> None:
    sys.stdout.write(f"{msg}\n")


def copy_fixture_to_dbfs(local_fixture: str, dbfs_target: str) -> None:
    """Copy local VN2 fixture directory to DBFS if not already present."""
    target = Path(dbfs_target)
    if target.exists() and any(target.iterdir()):
        _log(f"Fixture already present at {dbfs_target}")
        return

    target.mkdir(parents=True, exist_ok=True)
    local = Path(local_fixture)
    for src in local.iterdir():
        dst = target / src.name
        if src.is_file():
            dst.write_bytes(src.read_bytes())
    _log(f"Copied fixture from {local_fixture} to {dbfs_target}")


def run_smoke(config_path: str) -> dict:
    """Run Calibre smoke config and return diagnostics."""
    from calibre.cli.commands import run_config
    from calibre.cli.config import load_config
    from calibre.execution.backend import BackendResult

    config = load_config(config_path)
    result = run_config(config)

    df = result.ledger.to_df() if isinstance(result, BackendResult) else result

    return {
        "rows": len(df),
        "columns": sorted(df.columns.tolist()),
        "ledger_path": str(config.output.ledger_path) if config.output else None,
        "engine": config.execution.engine if config.execution else None,
    }


def main() -> int:
    # Paths
    repo_root = Path(__file__).resolve().parent.parent
    local_fixture = str(repo_root / "benchmarks" / "vn2" / "fixture")
    dbfs_base = "/dbfs/mnt/calibre"
    dbfs_fixture = f"{dbfs_base}/vn2-fixture"
    dbfs_results = f"{dbfs_base}/results"
    config_path = str(repo_root / "benchmarks" / "vn2" / "config" / "smoke_databricks.yaml")

    # Ensure fixture is on DBFS
    copy_fixture_to_dbfs(local_fixture, dbfs_fixture)

    # Ensure results directory exists
    Path(dbfs_results).mkdir(parents=True, exist_ok=True)

    _log(f"Running smoke test with config: {config_path}")
    diagnostics = run_smoke(config_path)

    _log("--- Smoke test results ---")
    _log(json.dumps(diagnostics, indent=2, default=str))

    # Assertions
    assert diagnostics["rows"] == 4, f"Expected 4 rows, got {diagnostics['rows']}"
    assert diagnostics["engine"] == "spark", (
        f"Expected spark engine, got {diagnostics['engine']}"
    )
    assert diagnostics["ledger_path"] is not None, "Ledger path not set"
    ledger_path = diagnostics["ledger_path"]
    assert Path(ledger_path).exists(), f"Ledger not written: {ledger_path}"

    _log("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
