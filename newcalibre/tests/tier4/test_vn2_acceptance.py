"""Run the full VN2 protocol and emit the self-validating R1-R4 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from newcalibre.protocols.vn2 import (
    capture_vn2_evidence_environment,
    emit_vn2_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
    validate_vn2_result_bundle,
    verify_vn2_inputs,
)

pytestmark = [
    pytest.mark.tier4,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Gate-A VN2 evidence is ratified only on Ubuntu 24.04 x86_64",
    ),
]

PROJECT_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "benchmarks" / "vn2" / "protocol.yaml"
INVENTORY_PATH = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"
LOCK_PATH = PROJECT_ROOT / "uv.lock"
DATA_PATH = PROJECT_ROOT / "data" / "vn2"
BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "vn2"
TRACKING_PATH = REPOSITORY_ROOT / "stage3" / "evidence" / "tracking" / "series.jsonl"


def test_full_vn2_run_emits_and_revalidates_exact_r1_r4_bundle() -> None:
    candidate_sha = _required_environment("VN2_CANDIDATE_SHA")
    workflow_sha = _required_environment("VN2_WORKFLOW_SHA")
    run_id = _required_environment("VN2_RUN_ID")
    run_url = _required_environment("VN2_RUN_URL")
    mode = _required_environment("VN2_MODE")
    assert mode in {"mint", "verify"}
    assert not BUNDLE_PATH.exists(), "Tier 4 requires a clean ignored artifact directory"

    tracking_before = _optional_digest(TRACKING_PATH)
    verify_vn2_inputs(DATA_PATH, INVENTORY_PATH)
    config = load_vn2_config(CONFIG_PATH)
    dataset = load_vn2_dataset(DATA_PATH, INVENTORY_PATH, config)
    result = run_vn2(dataset)

    assert len(result.series_identities) == 599
    assert len(result.orders) == 599 * 6
    assert len(result.settlements) == 599 * 8

    emitted = emit_vn2_result_bundle(
        BUNDLE_PATH,
        result=result,
        config=config,
        candidate_sha=candidate_sha,
        workflow_sha=workflow_sha,
        run_id=run_id,
        run_url=run_url,
        config_path=CONFIG_PATH,
        input_inventory_path=INVENTORY_PATH,
        lock_path=LOCK_PATH,
        environment=capture_vn2_evidence_environment(),
    )
    validated = validate_vn2_result_bundle(
        BUNDLE_PATH,
        expected_candidate_sha=candidate_sha,
        expected_workflow_sha=workflow_sha,
        expected_run_id=run_id,
        expected_config_path=CONFIG_PATH,
        expected_input_inventory_path=INVENTORY_PATH,
        expected_lock_path=LOCK_PATH,
    )

    assert validated == emitted
    assert _line_count(BUNDLE_PATH / "r1-orders.jsonl") == 599 * 6
    assert _line_count(BUNDLE_PATH / "r2-cost-ledger.jsonl") == 599 * 8
    trajectory = json.loads((BUNDLE_PATH / "r4-cost-trajectory.json").read_text(encoding="utf-8"))
    assert [row["round"] for row in trajectory["decision_rounds"]] == list(range(1, 7))
    assert len(trajectory["drain_remainder"]["periods"]) == 2
    assert all("r5" not in path.name.casefold() for path in BUNDLE_PATH.rglob("*"))
    assert _optional_digest(TRACKING_PATH) == tracking_before


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    assert value, f"{name} must be set by the evidence workflow"
    return value


def _optional_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _line_count(path: Path) -> int:
    return len(path.read_bytes().splitlines())
