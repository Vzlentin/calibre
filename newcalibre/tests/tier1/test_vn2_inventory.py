"""Exercise the successor-owned VN2 inventory and acquisition boundary.

Inventory/schema/refusal assertions are exact tolerance-class-1 facts. File
digests and the approved-copy receipt are byte-identity class-4 assertions.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    download_vn2_inputs,
    load_vn2_inventory,
    verify_vn2_inputs,
)
from newcalibre.protocols.vn2.inventory import read_verified_vn2_input

pytestmark = pytest.mark.tier1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROVED_INVENTORY = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"
APPROVED_LF_SHA256 = "54f8556a811eac81c9597c0fb0d2ef16dca5a6d84936188b7322fcdb8f15ed97"
VN2_DATA_SCRIPT = PROJECT_ROOT / "scripts" / "vn2_data.py"


def test_committed_inventory_is_the_exact_approved_inventory_blob() -> None:
    approved = APPROVED_INVENTORY.read_bytes()
    payload = approved.replace(b"\r\n", b"\n")

    assert hashlib.sha256(payload).hexdigest() == APPROVED_LF_SHA256
    inventory = load_vn2_inventory(APPROVED_INVENTORY)
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
        (lambda payload: payload.update(files=[]), "non-empty"),
        (lambda payload: payload["files"][1].update(name=payload["files"][0]["name"]), "unique"),
        (lambda payload: payload["files"][0].update(name="../escape.csv"), "basename"),
        (lambda payload: payload["files"][0].update(bytes=-1), "positive"),
        (lambda payload: payload["files"][0].update(sha256="not-a-digest"), "sha256"),
        (lambda payload: payload.update(extra="field"), "exact keys"),
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


def test_generic_inventory_accepts_an_arbitrary_nonempty_compatible_file_list(
    tmp_path: Path,
) -> None:
    _data, inventory_path, _config = write_dataset(tmp_path)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload["files"] = [payload["files"][0]]
    inventory_path.write_text(json.dumps(payload), encoding="utf-8")

    inventory = load_vn2_inventory(inventory_path)

    assert tuple(entry.name for entry in inventory.files) == (EXPECTED_FILES[0],)


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


def test_downloader_validates_source_names_and_consumed_bytes(tmp_path: Path) -> None:
    source_data, inventory_path, _config = write_dataset(tmp_path / "source")
    target = tmp_path / "target"
    sources = {name: f"memory://{name}" for name in EXPECTED_FILES}
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return (source_data / url.removeprefix("memory://")).read_bytes()

    download_vn2_inputs(
        target,
        sources,
        inventory_path,
        fetcher=fetch,
    )

    assert fetched == [sources[name] for name in EXPECTED_FILES]
    verify_vn2_inputs(target, inventory_path)

    with pytest.raises(VN2InputError, match=r"source names.*missing=.*week_8_sales\.csv"):
        download_vn2_inputs(
            tmp_path / "wrong-names",
            {name: url for name, url in sources.items() if name != "week_8_sales.csv"},
            inventory_path,
            fetcher=fetch,
        )


def test_default_downloader_bounds_urllib_read_to_approved_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_data, inventory_path, _config = write_dataset(tmp_path / "source")
    source_payload = (source_data / EXPECTED_FILES[0]).read_bytes()
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_payload["files"] = [inventory_payload["files"][0]]
    inventory_path.write_text(json.dumps(inventory_payload), encoding="utf-8")
    read_limits: list[int] = []
    opened: list[tuple[str, int]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            read_limits.append(limit)
            return source_payload

    def open_url(url: str, *, timeout: int) -> Response:
        opened.append((url, timeout))
        return Response()

    monkeypatch.setattr(inventory_module.urllib.request, "urlopen", open_url)
    target = tmp_path / "target"

    download_vn2_inputs(
        target,
        {EXPECTED_FILES[0]: "https://approved.example/input.csv"},
        inventory_path,
    )

    assert opened == [("https://approved.example/input.csv", 120)]
    assert read_limits == [len(source_payload) + 1]
    assert (target / EXPECTED_FILES[0]).read_bytes() == source_payload


def test_default_downloader_refuses_oversized_response_after_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_data, inventory_path, _config = write_dataset(tmp_path / "source")
    source_payload = (source_data / EXPECTED_FILES[0]).read_bytes()
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_payload["files"] = [inventory_payload["files"][0]]
    inventory_path.write_text(json.dumps(inventory_payload), encoding="utf-8")
    read_limits: list[int] = []

    class OversizedResponse:
        def __enter__(self) -> OversizedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            read_limits.append(limit)
            return source_payload + b"x"

    monkeypatch.setattr(
        inventory_module.urllib.request,
        "urlopen",
        lambda _url, *, timeout: OversizedResponse(),
    )
    target = tmp_path / "target"

    with pytest.raises(VN2InputError, match=rf"{EXPECTED_FILES[0]}.*size"):
        download_vn2_inputs(
            target,
            {EXPECTED_FILES[0]: "https://approved.example/input.csv"},
            inventory_path,
        )

    assert read_limits == [len(source_payload) + 1]
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    ("fault", "pattern"),
    [
        ("exception", r"download failed for week_0_master\.csv.*offline"),
        ("non-bytes", r"download for week_0_master\.csv did not return bytes"),
        ("wrong-digest", r"week_0_master\.csv.*sha256"),
    ],
)
def test_downloader_refuses_failed_or_invalid_fetch_without_touching_destination(
    tmp_path: Path,
    fault: str,
    pattern: str,
) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / EXPECTED_FILES[0]
    original = victim.read_bytes()
    corrupted = bytearray(original)
    corrupted[-2] ^= 1

    def fetch(_url: str) -> bytes:
        if fault == "exception":
            raise RuntimeError("offline")
        if fault == "non-bytes":
            payload: Any = "not bytes"
            return payload
        return bytes(corrupted)

    with pytest.raises(VN2InputError, match=pattern):
        download_vn2_inputs(
            data,
            {name: f"memory://{name}" for name in EXPECTED_FILES},
            inventory_path,
            fetcher=fetch,
        )

    assert victim.read_bytes() == original
    assert not list(data.glob(".*.part"))


def test_downloader_atomic_install_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / EXPECTED_FILES[0]
    original = victim.read_bytes()
    temporaries: list[Path] = []

    def fail_install(source: Path, target: Path) -> None:
        temporary = Path(source)
        temporaries.append(temporary)
        assert temporary.parent == data
        assert temporary.name.startswith(f".{victim.name}.")
        assert temporary.name.endswith(".part")
        assert temporary.is_file()
        assert Path(target) == victim
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_install)

    for _attempt in range(2):
        with pytest.raises(
            VN2InputError,
            match=r"cannot install downloaded input week_0_master\.csv",
        ):
            download_vn2_inputs(
                data,
                {name: f"memory://{name}" for name in EXPECTED_FILES},
                inventory_path,
                fetcher=lambda _url: original,
            )

    assert victim.read_bytes() == original
    assert len({temporary.name for temporary in temporaries}) == 2
    assert not any(temporary.exists() for temporary in temporaries)


def test_downloader_reports_install_and_cleanup_failures_without_masking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / EXPECTED_FILES[0]
    original = victim.read_bytes()
    temporaries: list[Path] = []

    def fail_install(source: Path, _target: Path) -> None:
        temporaries.append(Path(source))
        raise OSError("disk full")

    original_unlink = Path.unlink

    def fail_cleanup(self: Path, *, missing_ok: bool = False) -> None:
        if temporaries and self == temporaries[-1]:
            raise OSError("permission denied")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(os, "replace", fail_install)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(VN2InputError) as raised:
        download_vn2_inputs(
            data,
            {name: f"memory://{name}" for name in EXPECTED_FILES},
            inventory_path,
            fetcher=lambda _url: original,
        )

    message = str(raised.value)
    assert "cannot install downloaded input week_0_master.csv: disk full" in message
    assert "cleanup failed" in message
    assert "temporary file: permission denied" in message
    assert victim.read_bytes() == original
    assert len(temporaries) == 1
    os.unlink(temporaries[0])


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        ({}, "exactly one 'files' list"),
        ({"files": {}}, "exactly one 'files' list"),
        ({"files": [{"name": "a.csv", "url": "memory://a", "extra": True}]}, "exact"),
        ({"files": [{"name": 1, "url": "memory://a"}]}, "must be a string"),
        (
            {
                "files": [
                    {"name": "a.csv", "url": "memory://a"},
                    {"name": "a.csv", "url": "memory://other"},
                ]
            },
            "duplicate name",
        ),
    ],
    ids=["missing-files", "files-not-list", "entry-shape", "non-string", "duplicate"],
)
def test_source_mapping_refuses_malformed_or_duplicate_entries(
    tmp_path: Path,
    payload: object,
    pattern: str,
) -> None:
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VN2InputError, match=pattern):
        vn2_data._sources(source_path)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('{"files": [], "files": []}', "files"),
        (
            '{"files": [{"name": "a.csv", "name": "a.csv", "url": "memory://a"}]}',
            "name",
        ),
    ],
    ids=["top-level", "nested"],
)
def test_source_mapping_refuses_duplicate_json_keys_at_every_depth(
    tmp_path: Path,
    text: str,
    key: str,
) -> None:
    source_path = tmp_path / "sources.json"
    source_path.write_text(text, encoding="utf-8")

    with pytest.raises(VN2InputError, match=rf"duplicate JSON key '{key}'"):
        vn2_data._sources(source_path)


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


def test_if_missing_never_allows_cache_poison_to_bypass_verification(tmp_path: Path) -> None:
    data, inventory_path, _config = write_dataset(tmp_path)
    victim = data / "week_2_sales.csv"
    victim.write_bytes(b"cache poison")
    fetch_calls = 0

    def should_not_fetch(_url: str) -> bytes:
        nonlocal fetch_calls
        fetch_calls += 1
        raise AssertionError("if_missing must retain present cache entries until verification")

    with pytest.raises(VN2InputError, match=r"week_2_sales\.csv.*size"):
        download_vn2_inputs(
            data,
            {name: f"memory://{name}" for name in EXPECTED_FILES},
            inventory_path,
            if_missing=True,
            fetcher=should_not_fetch,
        )

    assert fetch_calls == 0


def test_inventory_has_no_successor_digest_mint_operation() -> None:
    expected_operations = {"download_vn2_inputs", "verify_vn2_inputs"}
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
    assert not any(
        name.startswith("mint") and callable(value)
        for name, value in vars(inventory_module).items()
    )
    assert set(command_action.choices or ()) == {"download", "verify"}
