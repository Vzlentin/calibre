"""Test the compact frozen-oracle capture contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from newcalibre.oracle import (
    ORACLE_COMMIT,
    ORACLE_LOCK_SHA256,
    ORACLE_TAG,
    OracleEvidenceError,
    load_capture,
)


def _write_capture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "capture"
    orders = root / "orders"
    orders.mkdir(parents=True)
    config = tmp_path / "oracle-config.yaml"
    inventory = tmp_path / "vn2-input-digests.json"
    config.write_bytes(b"synthetic oracle config\n")
    inventory.write_bytes(b'{"synthetic":"inventory"}\n')
    entries = []
    for round_number in range(1, 7):
        path = orders / f"round-{round_number}.json"
        payload = {
            "orders": {f"{index}_1": float(index) for index in range(599)},
            "origin": f"2024-0{round_number + 3}-15 00:00:00",
            "round_num": round_number,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries.append(
            {
                "path": f"orders/round-{round_number}.json",
                "round": round_number,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "actuals_semantics": "censored_sales_surrogate",
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "input_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "oracle_commit": ORACLE_COMMIT,
        "oracle_lock_sha256": ORACLE_LOCK_SHA256,
        "oracle_tag": ORACLE_TAG,
        "orders": entries,
        "schema": 1,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, config, inventory


def _rewrite_manifest(root: Path, **changes: object) -> None:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_load_capture_validates_compact_canonical_bundle(tmp_path: Path) -> None:
    """Load the seven-file capture and expose its frozen identities."""
    root, config, inventory = _write_capture(tmp_path)
    bundle = load_capture(root, config_path=config, input_inventory_path=inventory)

    assert bundle.manifest.oracle_commit == ORACLE_COMMIT
    assert bundle.manifest.actuals_semantics == "censored_sales_surrogate"
    assert [entry.round for entry in bundle.manifest.orders] == list(range(1, 7))
    assert bundle.manifest_sha256


def test_load_capture_rejects_changed_order_byte(tmp_path: Path) -> None:
    """Reject an order payload whose bytes no longer match its manifest."""
    root, config, inventory = _write_capture(tmp_path)
    path = root / "orders" / "round-1.json"
    path.write_bytes(path.read_bytes().replace(b'"0_1": 0.0', b'"0_1": 1.0', 1))

    with pytest.raises(OracleEvidenceError, match="digest"):
        load_capture(root, config_path=config, input_inventory_path=inventory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("oracle_commit", "a" * 40, "oracle_commit"),
        ("config_sha256", "a" * 64, "config"),
        ("input_inventory_sha256", "b" * 64, "input inventory"),
    ],
)
def test_load_capture_rejects_wrong_identity(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    """Bind oracle, configuration, and input-inventory identities."""
    root, config, inventory = _write_capture(tmp_path)
    _rewrite_manifest(root, **{field: value})

    with pytest.raises(OracleEvidenceError, match=message):
        load_capture(root, config_path=config, input_inventory_path=inventory)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_load_capture_rejects_file_set_drift(tmp_path: Path, mutation: str) -> None:
    """Require exactly one manifest and six round files."""
    root, config, inventory = _write_capture(tmp_path)
    if mutation == "missing":
        (root / "orders" / "round-6.json").unlink()
    else:
        (root / "orders" / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(OracleEvidenceError, match="file set"):
        load_capture(root, config_path=config, input_inventory_path=inventory)


def test_load_capture_rejects_symlinked_payload(tmp_path: Path) -> None:
    """Refuse symlinks before reading trusted capture bytes."""
    root, config, inventory = _write_capture(tmp_path)
    source = root / "orders" / "round-1.json"
    target = tmp_path / "round-1.json"
    source.replace(target)
    source.symlink_to(target)

    with pytest.raises(OracleEvidenceError, match="symbolic link"):
        load_capture(root, config_path=config, input_inventory_path=inventory)
