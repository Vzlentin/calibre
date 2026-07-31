"""Exercise the successor-owned VN2 inventory and verification boundary.

Inventory/schema/refusal assertions are exact tolerance-class-1 facts. The
approved file digests are byte-identity class-4 assertions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts import vn2_data
from tests.vn2_fixtures import EXPECTED_FILES, write_dataset

import newcalibre.protocols.vn2.inventory as inventory_module
from newcalibre.protocols import vn2 as vn2_module
from newcalibre.protocols.vn2 import (
    EXPECTED_INPUT_COUNT,
    VN2InputError,
    load_vn2_inventory,
    verify_vn2_inputs,
)
from newcalibre.protocols.vn2.inventory import read_verified_vn2_input

pytestmark = pytest.mark.tier1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROVED_INVENTORY = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"
APPROVED_LF_SHA256 = "12143fa694dd8e2ecfb295106a861f9905b7f1351610d86bd65ee0faaf52fd3e"
VN2_DATA_SCRIPT = PROJECT_ROOT / "scripts" / "vn2_data.py"


def test_committed_inventory_is_the_exact_approved_inventory_blob() -> None:
    approved = APPROVED_INVENTORY.read_bytes()
    payload = approved.replace(b"\r\n", b"\n")

    assert hashlib.sha256(payload).hexdigest() == APPROVED_LF_SHA256
    inventory = load_vn2_inventory(APPROVED_INVENTORY)
    assert inventory.content_sha256 == hashlib.sha256(approved).hexdigest()
    assert EXPECTED_INPUT_COUNT == 12
    assert len(inventory.files) == EXPECTED_INPUT_COUNT
    assert tuple(entry.name for entry in inventory.files) == (
        "week_0_initial_state.csv",
        "week_0_master.csv",
        "week_0_in_stock.csv",
        "week_0_sales.csv",
        "week_1_sales.csv",
        "week_2_sales.csv",
        "week_3_sales.csv",
        "week_4_sales.csv",
        "week_5_sales.csv",
        "week_6_sales.csv",
        "week_7_sales.csv",
        "week_8_sales.csv",
    )


def test_verifier_accepts_only_the_exact_file_set_and_every_digest(tmp_path: Path) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)

    inventory = verify_vn2_inputs(data, inventory_path)

    assert {entry.name for entry in inventory.files} == set(EXPECTED_FILES)


def test_selected_read_rehashes_only_its_approved_entry(tmp_path: Path) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    inventory = verify_vn2_inputs(data, inventory_path)
    selected = data / "week_4_sales.csv"
    expected = selected.read_bytes()

    (data / "late-extra.txt").write_text("not consumed", encoding="utf-8")
    assert read_verified_vn2_input(data, selected.name, inventory) == expected

    mutated = bytearray(expected)
    mutated[-2] ^= 1
    selected.write_bytes(bytes(mutated))
    with pytest.raises(VN2InputError, match=r"week_4_sales\.csv.*sha256"):
        read_verified_vn2_input(data, selected.name, inventory)


@pytest.mark.parametrize("fault", ["missing", "extra", "size", "digest"])
def test_verifier_refuses_attributable_file_set_and_content_faults(
    tmp_path: Path,
    fault: str,
) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / "week_4_sales.csv"
    if fault == "missing":
        victim.unlink()
        pattern = r"file-set mismatch.*missing=.*week_4_sales\.csv"
    elif fault == "extra":
        (data / "poison.txt").write_text("unexpected", encoding="utf-8")
        pattern = r"file-set mismatch.*extra=.*poison\.txt"
    elif fault == "size":
        victim.write_bytes(victim.read_bytes() + b"x")
        pattern = r"week_4_sales\.csv.*size"
    else:
        payload = bytearray(victim.read_bytes())
        payload[-2] ^= 1
        victim.write_bytes(bytes(payload))
        pattern = r"week_4_sales\.csv.*sha256"

    with pytest.raises(VN2InputError, match=pattern):
        verify_vn2_inputs(data, inventory_path)


def test_verifier_refuses_symlinked_approved_destination(tmp_path: Path) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / EXPECTED_FILES[0]
    backing = tmp_path / "backing.csv"
    backing.write_bytes(victim.read_bytes())
    victim.unlink()
    try:
        victim.symlink_to(backing)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(VN2InputError, match=rf"{victim.name}.*regular file"):
        verify_vn2_inputs(data, inventory_path)


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (lambda payload: payload.update(schema=2), "schema"),
        (lambda payload: payload.update(dataset="other"), "dataset"),
        (lambda payload: payload.update(files=[]), "exactly 12"),
        (lambda payload: payload["files"][1].update(name=payload["files"][0]["name"]), "unique"),
        (lambda payload: payload["files"][0].update(name="../escape.csv"), "basename"),
        (lambda payload: payload["files"][0].update(bytes=-1), "positive"),
        (lambda payload: payload["files"][0].update(sha256="not-a-digest"), "sha256"),
        (lambda payload: payload.update(extra="field"), "exact keys"),
        (lambda payload: payload.update(source_manifest="removed.json"), "exact keys"),
        (lambda payload: payload.update(source_manifest_sha256="0" * 64), "exact keys"),
    ],
    ids=[
        "schema",
        "dataset",
        "empty-files",
        "duplicate-name",
        "unsafe-name",
        "negative-size",
        "bad-digest",
        "extra-field",
        "removed-source-manifest",
        "removed-source-manifest-digest",
    ],
)
def test_inventory_schema_refuses_malformed_or_unsafe_facts(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
    pattern: str,
) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    del data
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    mutation(payload)
    inventory_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VN2InputError, match=pattern):
        load_vn2_inventory(inventory_path)


def test_inventory_refuses_a_reduced_compatible_file_list(
    tmp_path: Path,
) -> None:
    _data, inventory_path, _config = write_dataset(tmp_path)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload["files"] = [payload["files"][0]]
    inventory_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VN2InputError, match="exactly 12"):
        load_vn2_inventory(inventory_path)


@pytest.mark.parametrize(
    ("marker", "key"),
    [
        ('"dataset": "vn2",', "dataset"),
        (f'"name": "{EXPECTED_FILES[0]}",', "name"),
    ],
    ids=["top-level", "nested"],
)
def test_inventory_refuses_duplicate_json_keys_at_every_depth(
    tmp_path: Path,
    marker: str,
    key: str,
) -> None:
    _data, inventory_path, _config = write_dataset(tmp_path)
    text = inventory_path.read_text(encoding="utf-8")
    inventory_path.write_text(
        text.replace(marker, f"{marker}\n{marker}", 1),
        encoding="utf-8",
    )

    with pytest.raises(VN2InputError, match=rf"duplicate JSON key '{key}'"):
        load_vn2_inventory(inventory_path)


def _run_verify(data: Path, inventory_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VN2_DATA_SCRIPT),
            "verify",
            "--target",
            str(data),
            "--inventory",
            str(inventory_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_verify_script_executes_successfully_against_exact_inputs(tmp_path: Path) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)

    result = _run_verify(data, inventory_path)

    assert result.returncode == 0
    assert result.stdout == "verified 12 VN2 inputs\n"
    assert result.stderr == ""


def test_verify_script_reports_corrupt_input_attributably(tmp_path: Path) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / "week_4_sales.csv"
    corrupted = bytearray(victim.read_bytes())
    corrupted[-2] ^= 1
    victim.write_bytes(bytes(corrupted))

    result = _run_verify(data, inventory_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "week_4_sales.csv: sha256" in result.stderr


def test_inventory_and_cli_are_verification_only() -> None:
    expected_operations = {"verify_vn2_inputs"}
    module_operations = {
        name
        for name, value in vars(inventory_module).items()
        if not name.startswith("_") and name.endswith("_vn2_inputs") and callable(value)
    }
    package_operations = {name for name in vn2_module.__all__ if name.endswith("_vn2_inputs")}
    parser = vn2_data.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")

    assert module_operations == expected_operations
    assert package_operations == expected_operations
    assert not hasattr(inventory_module, "ByteFetcher")
    assert not hasattr(vn2_module, "download_vn2_inputs")
    assert not any(
        name.startswith("mint") and callable(value)
        for name, value in vars(inventory_module).items()
    )
    assert set(command_action.choices or ()) == {"verify"}
