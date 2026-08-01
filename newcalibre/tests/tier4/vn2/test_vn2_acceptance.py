"""Run the generic VN2 path and emit deterministic compact R1-R4 evidence."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from newcalibre.oracle import load_capture
from newcalibre.protocols.vn2 import (
    build_tracking_record,
    compare_tracking_records,
    emit_result_bundle,
    load_result_bundle,
    load_tracking_history,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
    verify_vn2_inputs,
)

pytestmark = [
    pytest.mark.tier4,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="VN2 evidence is ratified only on Ubuntu 24.04 x86_64",
    ),
]

PROJECT_ROOT = Path(__file__).parents[3]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CONFIG = PROJECT_ROOT / "benchmarks" / "vn2" / "protocol.yaml"
INVENTORY = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"
LOCK = PROJECT_ROOT / "uv.lock"
DATA = PROJECT_ROOT / "data" / "vn2"
BUNDLE = PROJECT_ROOT / "artifacts" / "vn2"
CAPTURE = REPOSITORY_ROOT / "stage3" / "evidence" / "captures" / "vn2"
ORACLE_CONFIG = REPOSITORY_ROOT / "benchmarks" / "vn2" / "config" / "vn2-winning-loop.yaml"
HISTORY = REPOSITORY_ROOT / "stage3" / "evidence" / "tracking" / "series.jsonl"


def test_generic_vn2_run_emits_deterministic_compact_r1_r4(tmp_path: Path) -> None:
    """Run 599 series and bind its ledger-derived result and tracking record."""
    candidate = os.environ.get("VN2_CANDIDATE_SHA")
    assert candidate and len(candidate) == 40, "VN2_CANDIDATE_SHA must be a full commit SHA"
    assert not BUNDLE.exists(), "Tier 4 requires a clean ignored artifact directory"

    capture = load_capture(CAPTURE, config_path=ORACLE_CONFIG, input_inventory_path=INVENTORY)
    verify_vn2_inputs(DATA, INVENTORY)
    config = load_vn2_config(CONFIG)
    result = run_vn2(load_vn2_dataset(DATA, INVENTORY, config))

    assert len(result.series_identities) == 599
    assert len(result.orders) == 3_594
    assert len(result.settlements) == 4_792

    emitted = emit_result_bundle(
        BUNDLE,
        result=result,
        config=config,
        candidate_sha=candidate,
        config_path=CONFIG,
        input_inventory_path=INVENTORY,
        lock_path=LOCK,
        capture_digest=capture.capture_digest,
    )
    second = emit_result_bundle(
        tmp_path / "second",
        result=result,
        config=config,
        candidate_sha=candidate,
        config_path=CONFIG,
        input_inventory_path=INVENTORY,
        lock_path=LOCK,
        capture_digest=capture.capture_digest,
    )
    for name in (*emitted.manifest.files, "manifest.json"):
        assert (emitted.root / name).read_bytes() == (second.root / name).read_bytes()
    assert (
        load_result_bundle(
            BUNDLE,
            expected_candidate_sha=candidate,
            config_path=CONFIG,
            input_inventory_path=INVENTORY,
            lock_path=LOCK,
            expected_capture_digest=capture.capture_digest,
        )
        == emitted
    )

    proposal = build_tracking_record(emitted)
    history = load_tracking_history(HISTORY)
    comparison = compare_tracking_records(history[-1], proposal)
    assert comparison.holding_delta == 0.0
    assert comparison.shortage_delta == 0.0
    assert comparison.total_delta == 0.0
    assert proposal.total_cost == proposal.holding_cost + proposal.shortage_cost

    shutil.rmtree(second.root)
