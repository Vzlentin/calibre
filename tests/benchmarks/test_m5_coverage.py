from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.m5.coverage import (
    COVERAGE_BY_NODE_NAME,
    COVERAGE_REPORT_NAME,
    LEVEL_COLUMN,
    infer_m5_level,
    write_coverage_artifacts,
)
from calibre.cli.main import app
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.reconciliation.summing import TOTAL_LABEL


def _synthetic_resolved_ledger() -> pd.DataFrame:
    lower_col, upper_col = interval_column_names(0.9)
    return pd.DataFrame(
        {
            UNIQUE_ID: [
                "ITEM_1_STORE_1",
                "ITEM_1_STORE_1",
                "dept_id=FOODS_1",
                "dept_id=FOODS_1",
                TOTAL_LABEL,
                TOTAL_LABEL,
            ],
            DS: pd.date_range("2024-01-02", periods=6, freq="D"),
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")] * 6,
            H: [1, 2, 1, 2, 1, 2],
            MODEL_NAME: ["SeasonalNaive"] * 6,
            Y: [10.0, 12.0, 22.0, 25.0, 32.0, 37.0],
            Y_HAT: [9.0, 13.0, 21.0, 24.0, 30.0, 36.0],
            lower_col: [8.0, 13.0, 21.0, 20.0, 28.0, pd.NA],
            upper_col: [11.0, 15.0, 24.0, 26.0, 35.0, 40.0],
        }
    )


def test_infer_m5_level_from_node_label() -> None:
    assert infer_m5_level(TOTAL_LABEL) == "total"
    assert infer_m5_level("dept_id=FOODS_1") == "dept_id"
    assert infer_m5_level("ITEM_1_STORE_1") == "bottom"


def test_m5_coverage_writer_emits_node_artifact_and_report(tmp_path: Path) -> None:
    artifacts = write_coverage_artifacts(
        _synthetic_resolved_ledger(),
        output_dir=tmp_path,
        coverage=0.9,
    )

    assert artifacts.coverage_by_node_path == tmp_path / COVERAGE_BY_NODE_NAME
    assert artifacts.report_path == tmp_path / COVERAGE_REPORT_NAME
    assert artifacts.coverage_by_node_path.exists()
    assert artifacts.report_path.exists()

    node = pd.read_parquet(artifacts.coverage_by_node_path)
    assert {
        UNIQUE_ID,
        LEVEL_COLUMN,
        H,
        MODEL_NAME,
        "coverage",
        "mean_interval_width",
        "scored_rows",
        "unscored_rows",
        "is_outlier",
    }.issubset(node.columns)
    assert set(node[LEVEL_COLUMN]) == {"bottom", "dept_id", "total"}
    assert int(node["scored_rows"].sum()) == 5
    assert int(node["unscored_rows"].sum()) == 1
    assert artifacts.population["coverage"].iloc[0] == pytest.approx(0.8)
    assert artifacts.per_horizon.set_index(H).loc[1, "coverage"] == pytest.approx(1.0)
    assert artifacts.per_horizon.set_index(H).loc[2, "coverage"] == pytest.approx(0.5)
    per_level = artifacts.per_level.set_index(LEVEL_COLUMN)
    assert per_level.loc["bottom", "average_node_coverage"] == pytest.approx(0.5)
    assert per_level.loc["bottom", "outlier_node_groups"] == 1

    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "Target coverage" in report
    assert "Population marginal coverage: 80.00%" in report
    assert "Acceptance gate: FAIL" in report
    assert "Per-Level Diagnostics" in report
    assert "Per-Horizon Diagnostics" in report
    assert "Per-Node Outliers" in report
    assert "ITEM_1_STORE_1" in report
    assert "not coherent interval boxes or conditional per-node guarantees" in report


def test_m5_coverage_writer_rejects_missing_interval_columns(tmp_path: Path) -> None:
    lower_col, upper_col = interval_column_names(0.9)
    ledger = _synthetic_resolved_ledger().drop(columns=[upper_col])

    with pytest.raises(ValueError, match="missing required column"):
        write_coverage_artifacts(ledger, output_dir=tmp_path, coverage=0.9)

    assert lower_col in ledger.columns


def test_m5_coverage_writer_rejects_unresolved_only_ledgers(tmp_path: Path) -> None:
    ledger = _synthetic_resolved_ledger()
    ledger[Y] = pd.NA

    with pytest.raises(ValueError, match="no resolved actual rows"):
        write_coverage_artifacts(ledger, output_dir=tmp_path, coverage=0.9)


def test_score_m5_coverage_cli_writes_artifacts(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "forecast-ledger.resolved.parquet"
    _synthetic_resolved_ledger().to_parquet(ledger_path, index=False)

    assert app(["score-m5-coverage", "--ledger", str(ledger_path), "--coverage", "0.9"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["coverage_by_node"]) == tmp_path / COVERAGE_BY_NODE_NAME
    assert Path(payload["report"]) == tmp_path / COVERAGE_REPORT_NAME
    assert (tmp_path / COVERAGE_BY_NODE_NAME).exists()
    assert (tmp_path / COVERAGE_REPORT_NAME).exists()


def test_score_m5_coverage_cli_missing_ledger_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.resolved.parquet"

    with pytest.raises(FileNotFoundError, match="resolved ledger not found"):
        app(["score-m5-coverage", "--ledger", str(missing)])
