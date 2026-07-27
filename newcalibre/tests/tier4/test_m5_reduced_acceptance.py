"""Run a digest-ranked real-M5 population through the generic engine path."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import newcalibre.protocols.m5.runner as runner
from newcalibre.protocols.m5 import load_m5_config, run_m5

pytestmark = pytest.mark.tier4

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = PROJECT_ROOT / "tests" / "fixtures" / "m5" / "reduced-real.yaml"
DATA = PROJECT_ROOT / "data" / "m5"
INVENTORY = PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"
LEVELS = frozenset({"bottom", "item", "department", "category", "store", "state", "total"})
ARTIFACTS = frozenset({"coverage-summary.json", "coverage-by-node.parquet", "report.md"})


def _isolated_project(root: Path) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "data" / "m5").symlink_to(DATA, target_is_directory=True)
    inventory = root / "benchmarks" / "m5" / "m5-inputs.json"
    inventory.parent.mkdir(parents=True)
    shutil.copy2(INVENTORY, inventory)
    return root


def _logical_result(result) -> tuple[object, ...]:
    diagnostics = result.diagnostics
    return (
        result.session,
        result.input_inventory_sha256,
        result.forecast_origin_count,
        result.commit_count,
        result.node_count,
        result.expected_row_count,
        result.resolved_row_count,
        result.eligible_row_count,
        result.scored_row_count,
        result.pending_row_count,
        diagnostics.status,
        diagnostics.context,
        diagnostics.population,
        dict(diagnostics.models),
        dict(diagnostics.levels),
    )


def test_reduced_real_m5_run_is_structurally_valid_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove mechanics only; timing and sales coverage cannot select Gate C inputs."""
    # Timing and observed sales coverage are neither Gate C evidence nor selection inputs.
    config = load_m5_config(CONFIG)
    assert config.population.kind == "digest_rank"
    assert config.population.bottom_count == 8

    first_root = _isolated_project(tmp_path / "first")
    monkeypatch.setattr(runner, "_PROJECT_ROOT", first_root)
    first = run_m5(CONFIG)
    second_root = _isolated_project(tmp_path / "second")
    monkeypatch.setattr(runner, "_PROJECT_ROOT", second_root)
    second = run_m5(CONFIG)

    assert first.diagnostics.status == "VALID"
    assert set(first.diagnostics.levels) == LEVELS
    assert all(summary.node_count > 0 for summary in first.diagnostics.levels.values())
    assert all(summary.scored_node_count > 0 for summary in first.diagnostics.levels.values())
    assert first.forecast_origin_count == config.origin_count == 64
    assert first.commit_count == config.origin_count + 1
    assert first.expected_row_count == first.node_count * config.origin_count * config.horizon
    assert first.resolved_row_count == first.node_count * 1414
    assert first.pending_row_count == first.node_count * 378
    assert first.resolved_row_count + first.pending_row_count == first.expected_row_count
    assert first.eligible_row_count == first.scored_row_count > 0
    assert first.diagnostics.population.mask_equal
    assert {path.name for path in first.diagnostics.paths} == ARTIFACTS
    assert {path.name for path in first.diagnostics.summary_path.parent.iterdir()} == ARTIFACTS
    assert _logical_result(first) == _logical_result(second)
    assert {path.name: path.read_bytes() for path in first.diagnostics.paths} == {
        path.name: path.read_bytes() for path in second.diagnostics.paths
    }
