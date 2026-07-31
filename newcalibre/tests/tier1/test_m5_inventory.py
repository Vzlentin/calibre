"""Prove the M5 input verification-only boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import newcalibre.protocols.m5 as m5
import newcalibre.protocols.m5.inventory as inventory_module
from newcalibre.protocols.m5 import verify_m5_inputs
from newcalibre.protocols.m5.inventory import (
    M5InputError,
    load_m5_inventory,
    read_verified_m5_input,
)

_PROJECT_ROOT = Path(__file__).parents[2]
_INVENTORY = _PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"
_SCRIPT = _PROJECT_ROOT / "scripts" / "m5_data.py"


def _entry(name: str, payload: bytes) -> dict[str, object]:
    return {
        "name": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _inventory_path(
    tmp_path: Path,
    files: list[dict[str, object]] | None = None,
    **updates: object,
) -> Path:
    payload: dict[str, object] = {
        "schema": 1,
        "dataset": "m5",
        "files": files
        if files is not None
        else [_entry("calendar.csv", b"calendar"), _entry("sales_train_evaluation.csv", b"sales")],
    }
    payload.update(updates)
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_directory(tmp_path: Path) -> Path:
    target = tmp_path / "inputs"
    target.mkdir()
    (target / "calendar.csv").write_bytes(b"calendar")
    (target / "sales_train_evaluation.csv").write_bytes(b"sales")
    return target


def test_committed_inventory_pins_only_consumed_canonical_inputs() -> None:
    raw = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert raw == {
        "schema": 1,
        "dataset": "m5",
        "files": [
            {
                "name": "calendar.csv",
                "bytes": 112477,
                "sha256": "568d0fe5f41790142379698732908e4e57432c1c6396f3f59fb880a9c2b54231",
            },
            {
                "name": "sales_train_evaluation.csv",
                "bytes": 121168898,
                "sha256": "c21a519596680feb86f27a9e62f6c8b583f8be60c2c195f080ae8ca2990af2b7",
            },
        ],
    }
    inventory = load_m5_inventory(_INVENTORY)
    assert tuple(inventory.by_name) == ("calendar.csv", "sales_train_evaluation.csv")


def test_verification_rejects_unrelated_directory_entries(tmp_path: Path) -> None:
    target = _input_directory(tmp_path)
    (target / "sell_prices.csv").write_bytes(b"not consumed")
    (target / "notes.txt").write_text("ignored", encoding="utf-8")

    with pytest.raises(
        M5InputError,
        match=r"file-set mismatch.*extra=.*notes\.txt.*sell_prices\.csv",
    ):
        verify_m5_inputs(target, _inventory_path(tmp_path))


@pytest.mark.parametrize("name", ["calendar.csv", "sales_train_evaluation.csv"])
def test_verification_rejects_missing_consumed_file(tmp_path: Path, name: str) -> None:
    target = _input_directory(tmp_path)
    (target / name).unlink()
    with pytest.raises(M5InputError, match="missing"):
        verify_m5_inputs(target, _inventory_path(tmp_path))


@pytest.mark.parametrize("replacement", [b"wrong-size", b"sales"])
def test_verification_rejects_wrong_size_or_digest(
    tmp_path: Path,
    replacement: bytes,
) -> None:
    target = _input_directory(tmp_path)
    (target / "calendar.csv").write_bytes(replacement)
    with pytest.raises(M5InputError, match="size|sha256"):
        verify_m5_inputs(target, _inventory_path(tmp_path))


def test_verification_rejects_symlinked_consumed_file(tmp_path: Path) -> None:
    target = _input_directory(tmp_path)
    (target / "calendar.csv").unlink()
    (target / "calendar.csv").symlink_to(target / "sales_train_evaluation.csv")
    with pytest.raises(M5InputError, match="regular file"):
        verify_m5_inputs(target, _inventory_path(tmp_path))


def test_verification_translates_unreadable_input_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _input_directory(tmp_path)
    inventory = _inventory_path(tmp_path)
    permission_error = PermissionError("refused")

    def refuse_iteration(_path: Path):
        raise permission_error

    monkeypatch.setattr(Path, "iterdir", refuse_iteration)

    with pytest.raises(
        M5InputError,
        match=r"M5 input directory cannot be read: .*inputs",
    ) as exc_info:
        verify_m5_inputs(target, inventory)

    assert exc_info.value.__cause__ is permission_error


@pytest.mark.parametrize(
    "files",
    [
        [_entry("calendar.csv", b"calendar")],
        [
            _entry("calendar.csv", b"calendar"),
            _entry("calendar.csv", b"calendar"),
        ],
        [
            _entry("../calendar.csv", b"calendar"),
            _entry("sales_train_evaluation.csv", b"sales"),
        ],
        [
            {"name": "calendar.csv", "bytes": 0, "sha256": "0" * 64},
            _entry("sales_train_evaluation.csv", b"sales"),
        ],
        [
            {"name": "calendar.csv", "bytes": 8, "sha256": "A" * 64},
            _entry("sales_train_evaluation.csv", b"sales"),
        ],
    ],
)
def test_inventory_rejects_malformed_file_entries(
    tmp_path: Path,
    files: list[dict[str, object]],
) -> None:
    with pytest.raises(M5InputError):
        load_m5_inventory(_inventory_path(tmp_path, files))


@pytest.mark.parametrize(
    "updates",
    [
        {"schema": 2},
        {"dataset": "other"},
        {"extra": True},
    ],
)
def test_inventory_rejects_unknown_schema_or_keys(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    with pytest.raises(M5InputError):
        load_m5_inventory(_inventory_path(tmp_path, **updates))


def test_inventory_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":1,"schema":1,"dataset":"m5","files":[]}',
        encoding="utf-8",
    )
    with pytest.raises(M5InputError, match="duplicate JSON key 'schema'"):
        load_m5_inventory(path)


def test_selected_reads_rehash_immediately_before_consumption(tmp_path: Path) -> None:
    target = _input_directory(tmp_path)
    inventory = verify_m5_inputs(target, _inventory_path(tmp_path))
    (target / "calendar.csv").write_bytes(b"calendaR")

    with pytest.raises(M5InputError, match="sha256"):
        read_verified_m5_input(target, "calendar.csv", inventory)


def test_package_and_script_are_verification_only() -> None:
    assert m5.__all__ == [
        "M5Diagnostics",
        "M5RunResult",
        "load_m5_config",
        "run_m5",
        "score_m5",
        "verify_m5_inputs",
    ]
    assert not hasattr(inventory_module, "ByteFetcher")
    assert not hasattr(inventory_module, "download_m5_inputs")
    assert not hasattr(m5, "download_m5_inputs")
    assert tuple(inspect.signature(verify_m5_inputs).parameters) == ("target", "inventory_path")

    namespace = runpy.run_path(str(_SCRIPT))
    parser = namespace["build_parser"]()
    commands = cast(object, parser._subparsers._group_actions[0]).choices  # type: ignore[attr-defined]
    assert set(commands) == {"verify"}


def _run_verify(target: Path, inventory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "verify",
            "--target",
            str(target),
            "--inventory",
            str(inventory),
        ],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_verify_script_executes_successfully_against_exact_inputs(tmp_path: Path) -> None:
    target = _input_directory(tmp_path)

    result = _run_verify(target, _inventory_path(tmp_path))

    assert result.returncode == 0
    assert result.stdout == "verified 2 consumed M5 inputs\n"
    assert result.stderr == ""


def test_verify_script_reports_extra_input_attributably(tmp_path: Path) -> None:
    target = _input_directory(tmp_path)
    (target / "poison.csv").write_bytes(b"unexpected")

    result = _run_verify(target, _inventory_path(tmp_path))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "file-set mismatch" in result.stderr
    assert "poison.csv" in result.stderr
