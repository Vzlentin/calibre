"""Gate tier 3 on the compact frozen oracle capture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from newcalibre.oracle import CaptureBundle, OracleEvidenceError, load_capture
from newcalibre.protocols.vn2 import (
    EXPECTED_INPUT_COUNT,
    VN2Dataset,
    load_vn2_config,
    load_vn2_dataset,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
CAPTURE = REPOSITORY_ROOT / "stage3" / "evidence" / "captures" / "vn2"
ORACLE_CONFIG = REPOSITORY_ROOT / "benchmarks" / "vn2" / "config" / "vn2-winning-loop.yaml"
INPUT_INVENTORY = REPOSITORY_ROOT / "newcalibre" / "benchmarks" / "vn2" / "vn2-input-digests.json"
SUCCESSOR_CONFIG = REPOSITORY_ROOT / "newcalibre" / "benchmarks" / "vn2" / "protocol.yaml"
SUCCESSOR_DATA = REPOSITORY_ROOT / "newcalibre" / "data" / "vn2"


@pytest.fixture(scope="session")
def promoted_captures_root() -> Path:
    """Expose the canonical capture or visibly skip only when it is absent."""
    if not CAPTURE.exists():
        pytest.skip("tier 3 skipped: canonical oracle capture is absent")
    if CAPTURE.is_symlink() or not CAPTURE.is_dir():
        pytest.fail("tier 3 canonical capture must be a real directory")
    return CAPTURE


@pytest.fixture(scope="session")
def validated_promoted_capture(promoted_captures_root: Path) -> CaptureBundle:
    """Validate the canonical capture using only committed trusted bytes."""
    if sys.platform == "win32":
        pytest.skip("tier 3 conditional replay is ratified only on Ubuntu 24.04 x86_64")
    try:
        return load_capture(
            promoted_captures_root,
            config_path=ORACLE_CONFIG,
            input_inventory_path=INPUT_INVENTORY,
        )
    except OracleEvidenceError as error:
        pytest.fail(str(error))


@pytest.fixture(scope="session")
def exact_vn2_dataset(validated_promoted_capture: CaptureBundle) -> VN2Dataset:
    """Verify and load exactly the twelve inputs bound by the capture."""
    del validated_promoted_capture
    config = load_vn2_config(SUCCESSOR_CONFIG)
    if len(config.files.all_names) != EXPECTED_INPUT_COUNT or EXPECTED_INPUT_COUNT != 12:
        pytest.fail("tier 3 requires exactly twelve configured VN2 inputs")
    return load_vn2_dataset(SUCCESSOR_DATA, INPUT_INVENTORY, config)
