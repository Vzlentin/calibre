"""Validate the exact append-only Stage 3 tracking landing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from newcalibre.benchmarking import load_m5_gate_c_result
from newcalibre.tracking import (
    M5TrackingRecord,
    build_m5_tracking_record,
    load_tracking_history,
    validate_tracking_append,
)

pytestmark = pytest.mark.tier4

PROJECT_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
RESULT = PROJECT_ROOT / "benchmarks" / "results" / "m5-gate-c"
HISTORY = REPOSITORY_ROOT / "stage3" / "evidence" / "tracking" / "series.jsonl"
CONFIG = PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
INVENTORY = PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"
LOCK = PROJECT_ROOT / "uv.lock"
_HISTORICAL_PREFIX_SHA256 = "364372364a2170a168ee5fa92bcb3403256ed7d138b1e850fdf5f96e99933db1"

if not RESULT.exists():
    pytest.skip(
        "the one retained-host Gate C result has not been committed", allow_module_level=True
    )


def test_one_m5_record_is_an_exact_append_matching_all_five_files() -> None:
    """Preserve every historical byte and bind the final record to the result."""
    result = load_m5_gate_c_result(
        RESULT,
        config_path=CONFIG,
        inventory_path=INVENTORY,
        lock_path=LOCK,
    )
    payload = HISTORY.read_bytes()
    lines = payload.splitlines(keepends=True)
    assert len(lines) == 4
    base = b"".join(lines[:3])
    assert hashlib.sha256(base).hexdigest() == _HISTORICAL_PREFIX_SHA256
    appended = validate_tracking_append(base, payload)

    assert appended == (build_m5_tracking_record(result),)
    assert isinstance(load_tracking_history(payload)[-1], M5TrackingRecord)
