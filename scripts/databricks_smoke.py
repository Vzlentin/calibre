#!/usr/bin/env python3
"""Databricks smoke test for Calibre.

Run inside a Databricks notebook cell or as a Databricks job.

Prerequisites:
- %pip install /dbfs/mnt/calibre/calibre-0.1.0-py3-none-any.whl[benchmarks,ray]
- VN2 fixture data copied to /dbfs/mnt/calibre/vn2-fixture/

Expected result:
    Rows: 4
    Ledger columns: [...]
    Backend: local
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


def run_smoke(data_path: str, ledger_path: str) -> dict:
    """Run Calibre smoke config and return diagnostics."""
    from calibre.cli.commands import run_config
    from calibre.cli.config import load_config_from_mapping
    from calibre.execution.backend import BackendResult

    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": data_path, "period": 0},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 2,
                    "config": {"backend": "statsforecast", "season_length": 2},
                }
            ],
            "origins": {"start": "2024-01-29", "end": "2024-01-29", "freq": "W-MON"},
            "output": {"ledger_path": ledger_path, "streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    result = run_config(config)

    df = result.ledger.to_df() if isinstance(result, BackendResult) else result

    return {
        "rows": len(df),
        "columns": sorted(df.columns.tolist()),
        "ledger_path": str(config.output.ledger_path) if config.output else None,
        "backend": config.execution.backend if config.execution else None,
    }


def main() -> int:
    """Run the Databricks smoke check and return a process exit code."""
    # Paths
    repo_root = Path(__file__).resolve().parent.parent
    local_fixture = str(repo_root / "benchmarks" / "vn2" / "fixture")
    dbfs_base = "/dbfs/mnt/calibre"
    dbfs_fixture = f"{dbfs_base}/vn2-fixture"
    dbfs_results = f"{dbfs_base}/results"
    ledger_path = f"{dbfs_results}/smoke-ledger.parquet"

    # Ensure fixture is on DBFS
    copy_fixture_to_dbfs(local_fixture, dbfs_fixture)

    # Ensure results directory exists
    Path(dbfs_results).mkdir(parents=True, exist_ok=True)

    _log(f"Running smoke test with fixture: {dbfs_fixture}")
    diagnostics = run_smoke(dbfs_fixture, ledger_path)

    _log("--- Smoke test results ---")
    _log(json.dumps(diagnostics, indent=2, default=str))

    # Assertions
    assert diagnostics["rows"] == 4, f"Expected 4 rows, got {diagnostics['rows']}"
    assert diagnostics["backend"] == "local", (
        f"Expected local backend, got {diagnostics['backend']}"
    )
    assert diagnostics["ledger_path"] is not None, "Ledger path not set"
    ledger_path = diagnostics["ledger_path"]
    assert Path(ledger_path).exists(), f"Ledger not written: {ledger_path}"

    _log("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
