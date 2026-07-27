"""Exercise deterministic and atomic M5 diagnostic artifact publication."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.engine import LedgerBatch, LedgerSelection
from newcalibre.protocols.m5 import load_m5_config, score_m5
from newcalibre.protocols.m5.scorer import M5ScoringError
from tier1.test_m5_scorer import _GATE_C, _Reader, _rows

_EXPECTED_FILES = {
    "coverage-summary.json",
    "coverage-by-node.parquet",
    "report.md",
}
_EXPECTED_LEVELS = [
    "bottom",
    "item",
    "department",
    "category",
    "store",
    "state",
    "total",
]


def test_artifacts_have_exact_names_canonical_summary_and_fixed_node_schema(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostics"
    diagnostics = score_m5(
        load_m5_config(_GATE_C),
        _Reader(_rows()),
        output_dir=output,
    )

    assert {path.name for path in output.iterdir()} == _EXPECTED_FILES
    summary_bytes = diagnostics.summary_path.read_bytes()
    assert summary_bytes.endswith(b"\n")
    assert b"\r" not in summary_bytes
    summary = json.loads(summary_bytes)
    assert summary_bytes == canonical_json_bytes(summary, path="test summary") + b"\n"
    assert set(summary) == {
        "schema",
        "status",
        "metric",
        "context",
        "mask",
        "population",
        "per_model",
        "levels",
    }
    assert summary["schema"] == 1
    assert summary["status"] == "VALID"
    assert summary["metric"] == "sales-coverage"
    assert [level["level"] for level in summary["levels"]] == _EXPECTED_LEVELS
    assert all(level["label"] == "sales-coverage" for level in summary["levels"])

    table = pq.read_table(diagnostics.by_node_path)
    assert table.schema.names == [
        "schema",
        "status",
        "metric",
        "dataset",
        "phase",
        "session_id",
        "reconciler",
        "conformal_method",
        "conformal_partition",
        "origin_start",
        "origin_end",
        "origin_count",
        "horizon",
        "level",
        "node",
        "model",
        "target",
        "coverage",
        "deviation",
        "total",
        "resolved",
        "eligible",
        "scored",
        "covered",
        "mask_identity",
        "mask_equal",
    ]
    assert table.num_rows == 7
    rows = table.to_pylist()
    assert [(row["level"], row["node"].encode(), row["model"].encode()) for row in rows] == sorted(
        (row["level"], row["node"].encode(), row["model"].encode()) for row in rows
    )
    assert all(row["metric"] == "sales-coverage" for row in rows)
    assert all(row["mask_equal"] for row in rows)


def test_all_projections_repeat_complete_context_and_sales_limitation(tmp_path: Path) -> None:
    diagnostics = score_m5(
        load_m5_config(_GATE_C),
        _Reader(_rows()),
        output_dir=tmp_path / "diagnostics",
    )
    summary = json.loads(diagnostics.summary_path.read_bytes())
    context = summary["context"]

    assert context == {
        "conformal_method": "split-per-step",
        "conformal_partition": "series-horizon",
        "dataset": "m5",
        "expected_row_count": 7 * 64 * 28,
        "horizon": 28,
        "model_name": "seasonal-naive",
        "node_count": 7,
        "origin_count": 64,
        "origin_end": "2026-03-05",
        "origin_start": "2026-01-01",
        "phase": "evaluation",
        "reconciler": "wls_struct",
        "session_id": diagnostics.context.session_id,
        "target": 0.9,
    }
    node_row = pq.read_table(diagnostics.by_node_path).to_pylist()[0]
    for name in (
        "dataset",
        "phase",
        "session_id",
        "reconciler",
        "conformal_method",
        "conformal_partition",
        "origin_start",
        "origin_end",
        "origin_count",
        "horizon",
        "target",
    ):
        assert node_row[name] == context[name]

    report = diagnostics.report_path.read_text(encoding="utf-8")
    assert report.endswith("\n")
    assert "sales-coverage" in report
    assert "recorded sales" in report
    assert "not a demand or service guarantee" in report
    assert "coverage-summary.json" in report
    assert "coverage-by-node.parquet" in report
    for value in (
        "evaluation",
        "wls_struct",
        "split-per-step",
        "series-horizon",
        "2026-01-01",
        "2026-03-05",
        "0.9",
    ):
        assert value in report
    forbidden = ("confidence interval", "bootstrap", "outlier", "statistical verdict")
    assert all(term not in report.lower() for term in forbidden)


def test_batch_boundaries_do_not_change_logical_artifacts(tmp_path: Path) -> None:
    config = load_m5_config(_GATE_C)
    rows = _rows()
    first = score_m5(config, _Reader(rows, batch_size=1), output_dir=tmp_path / "one")
    second = score_m5(config, _Reader(rows, batch_size=503), output_dir=tmp_path / "many")

    assert first.summary_path.read_bytes() == second.summary_path.read_bytes()
    assert first.report_path.read_bytes() == second.report_path.read_bytes()
    assert pq.read_table(first.by_node_path).equals(pq.read_table(second.by_node_path))


def test_readable_mask_inconsistency_publishes_all_invalid_artifacts(tmp_path: Path) -> None:
    diagnostics = score_m5(
        load_m5_config(_GATE_C),
        _Reader(_rows(mutation="missing-eligible")),
        output_dir=tmp_path / "invalid",
    )

    assert diagnostics.status == "INVALID"
    assert {path.name for path in diagnostics.summary_path.parent.iterdir()} == _EXPECTED_FILES
    assert json.loads(diagnostics.summary_path.read_bytes())["status"] == "INVALID"
    assert set(pq.read_table(diagnostics.by_node_path)["status"].to_pylist()) == {"INVALID"}
    assert "**Structural status:** INVALID" in diagnostics.report_path.read_text(encoding="utf-8")


def test_scan_iteration_and_output_failures_publish_no_artifact_set(tmp_path: Path) -> None:
    config = load_m5_config(_GATE_C)

    class BrokenReader(_Reader):
        def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
            iterator = super().scan(selection)

            def broken() -> Iterator[LedgerBatch]:
                yield next(iterator)
                raise OSError("synthetic iteration failure")

            return broken()

    iteration_output = tmp_path / "iteration"
    with pytest.raises(M5ScoringError, match="iteration"):
        score_m5(config, BrokenReader(_rows()), output_dir=iteration_output)
    assert not iteration_output.exists()

    class ScanFailure(_Reader):
        def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
            del selection
            raise OSError("synthetic scan failure")

    scan_output = tmp_path / "scan"
    with pytest.raises(M5ScoringError, match="scan"):
        score_m5(config, ScanFailure(_rows()), output_dir=scan_output)
    assert not scan_output.exists()

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("occupied", encoding="utf-8")
    output = parent_file / "diagnostics"
    with pytest.raises(M5ScoringError, match="output|publication|unavailable"):
        score_m5(config, _Reader(_rows()), output_dir=output)
    assert not output.exists()
