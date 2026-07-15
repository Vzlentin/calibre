"""Gate tier 3 on promoted capture bytes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from newcalibre.oracle import (
    CaptureBundle,
    validate_capture_bundle,
    validate_capture_receipt,
)
from newcalibre.protocols.vn2 import (
    EXPECTED_INPUT_COUNT,
    VN2Dataset,
    load_vn2_config,
    load_vn2_dataset,
    verify_vn2_inputs,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
CAPTURES_ROOT = Path(__file__).parents[3] / "stage3" / "evidence" / "captures"
CAPTURE_SHA = "ba45e9463e6b9d2921ca0d9e9692d2645a228058"
WORKFLOW_RUN_ID = "29293635537"
ARTIFACT_ID = "8295964148"
ARTIFACT_DIGEST = "03b41985ac36f4a865a3489879ebccf19010dffabe36c463e0e48517bdbc962b"
ORACLE_CONFIG = REPOSITORY_ROOT / "benchmarks" / "vn2" / "config" / "vn2-winning-loop.yaml"
INPUT_INVENTORY = REPOSITORY_ROOT / "stage3" / "evidence" / "vn2-input-digests.json"
SUCCESSOR_CONFIG = REPOSITORY_ROOT / "newcalibre" / "benchmarks" / "vn2" / "protocol.yaml"
SUCCESSOR_DATA = REPOSITORY_ROOT / "newcalibre" / "data" / "vn2"
_CAPTURE_NAME = re.compile(r"[0-9a-f]{40}")
_RECEIPT_NAME = re.compile(r"([0-9a-f]{40})-receipt\.json")


@pytest.fixture(scope="session")
def promoted_captures_root() -> Path:
    """Expose promoted bytes or visibly skip only when they are absent."""
    if not CAPTURES_ROOT.is_dir():
        pytest.skip("tier 3 skipped: promoted oracle captures are absent")
    return CAPTURES_ROOT


@pytest.fixture(scope="session")
def validated_promoted_capture(promoted_captures_root: Path) -> CaptureBundle:
    """Validate the sole canonical bundle and receipt using only committed bytes."""
    if sys.platform == "win32":
        pytest.skip("tier 3 conditional replay is ratified only on Ubuntu 24.04 x86_64")

    entries = tuple(promoted_captures_root.iterdir())
    bundles = tuple(
        path for path in entries if path.is_dir() and _CAPTURE_NAME.fullmatch(path.name)
    )
    receipts = tuple(
        path for path in entries if path.is_file() and _RECEIPT_NAME.fullmatch(path.name)
    )
    expected_names = {CAPTURE_SHA, f"{CAPTURE_SHA}-receipt.json"}
    if len(bundles) != 1 or len(receipts) != 1 or {path.name for path in entries} != expected_names:
        pytest.fail("tier 3 requires exactly one canonical 40-SHA capture bundle and receipt")
    bundle_root = bundles[0]
    receipt_path = receipts[0]
    if bundle_root.name != CAPTURE_SHA or receipt_path.name != f"{CAPTURE_SHA}-receipt.json":
        pytest.fail("tier 3 promoted capture identity is not the canonical U7b evidence")

    bundle = validate_capture_bundle(
        bundle_root,
        expected_candidate_sha=CAPTURE_SHA,
        expected_workflow_sha=CAPTURE_SHA,
        expected_run_id=WORKFLOW_RUN_ID,
        expected_config_path=ORACLE_CONFIG,
        expected_input_inventory_path=INPUT_INVENTORY,
    )
    validate_capture_receipt(
        receipt_path,
        bundle=bundle,
        expected_artifact_id=ARTIFACT_ID,
        expected_artifact_digest=ARTIFACT_DIGEST,
        expected_artifact_name=f"oracle-capture-{CAPTURE_SHA}",
        expected_producer_sha=CAPTURE_SHA,
        expected_workflow_sha=CAPTURE_SHA,
        expected_workflow_run_id=WORKFLOW_RUN_ID,
    )
    return bundle


@pytest.fixture(scope="session")
def exact_vn2_dataset(validated_promoted_capture: CaptureBundle) -> VN2Dataset:
    """Verify and load exactly the twelve inputs bound by the promoted capture."""
    del validated_promoted_capture
    inventory = verify_vn2_inputs(SUCCESSOR_DATA, INPUT_INVENTORY)
    if len(inventory.files) != EXPECTED_INPUT_COUNT or EXPECTED_INPUT_COUNT != 12:
        pytest.fail("tier 3 requires the exact twelve-file approved VN2 inventory")
    config = load_vn2_config(SUCCESSOR_CONFIG)
    return load_vn2_dataset(SUCCESSOR_DATA, INPUT_INVENTORY, config)
