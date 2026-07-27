"""Prove M5 input verification and script-only acquisition boundaries."""

from __future__ import annotations

import hashlib
import inspect
import json
import runpy
from pathlib import Path
from typing import cast

import pytest

import newcalibre.protocols.m5 as m5
from newcalibre.protocols.m5 import verify_m5_inputs
from newcalibre.protocols.m5.inventory import (
    M5InputError,
    download_m5_inputs,
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


def test_verification_accepts_unrelated_directory_entries(tmp_path: Path) -> None:
    target = _input_directory(tmp_path)
    (target / "sell_prices.csv").write_bytes(b"not consumed")
    (target / "notes.txt").write_text("ignored", encoding="utf-8")

    inventory = verify_m5_inputs(target, _inventory_path(tmp_path))

    assert set(inventory.by_name) == {"calendar.csv", "sales_train_evaluation.csv"}


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


def test_download_refuses_unapproved_names_and_overlong_bytes(tmp_path: Path) -> None:
    inventory_path = _inventory_path(tmp_path)
    sources = {
        "calendar.csv": "https://example.test/calendar",
        "sales_train_evaluation.csv": "https://example.test/sales",
    }
    with pytest.raises(M5InputError, match="source names mismatch"):
        download_m5_inputs(
            tmp_path / "download-extra",
            {**sources, "sell_prices.csv": "https://example.test/prices"},
            inventory_path,
            fetcher=lambda _url: b"",
        )
    with pytest.raises(M5InputError, match="size"):
        download_m5_inputs(
            tmp_path / "download-long",
            sources,
            inventory_path,
            fetcher=lambda _url: b"x" * 20,
        )
    assert not (tmp_path / "download-long" / "calendar.csv").exists()


def test_download_bounds_the_real_network_read_before_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_path = _inventory_path(tmp_path)
    read_sizes: list[int] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return b"x" * size

    def fake_urlopen(_url: str, *, timeout: int) -> Response:
        assert timeout == 120
        return Response()

    monkeypatch.setattr(
        "newcalibre.protocols.m5.inventory.urllib.request.urlopen",
        fake_urlopen,
    )
    target = tmp_path / "download-bounded"
    with pytest.raises(M5InputError, match="size"):
        download_m5_inputs(
            target,
            {
                "calendar.csv": "https://example.test/calendar",
                "sales_train_evaluation.csv": "https://example.test/sales",
            },
            inventory_path,
        )

    assert read_sizes == [len(b"calendar") + 1]
    assert not (target / "calendar.csv").exists()


def test_download_installs_only_verified_bytes_then_publicly_reverifies(tmp_path: Path) -> None:
    inventory_path = _inventory_path(tmp_path)
    payloads = {
        "https://example.test/calendar": b"calendar",
        "https://example.test/sales": b"sales",
    }
    target = tmp_path / "download"
    inventory = download_m5_inputs(
        target,
        {
            "calendar.csv": "https://example.test/calendar",
            "sales_train_evaluation.csv": "https://example.test/sales",
        },
        inventory_path,
        fetcher=lambda url: payloads[url],
    )
    assert inventory.content_sha256 == verify_m5_inputs(target, inventory_path).content_sha256


def test_package_and_script_keep_acquisition_private() -> None:
    assert m5.__all__ == ["load_m5_config", "verify_m5_inputs"]
    assert not hasattr(m5, "download_m5_inputs")
    assert tuple(inspect.signature(verify_m5_inputs).parameters) == ("target", "inventory_path")

    namespace = runpy.run_path(str(_SCRIPT))
    parser = namespace["build_parser"]()
    commands = cast(object, parser._subparsers._group_actions[0]).choices  # type: ignore[attr-defined]
    assert set(commands) == {"download", "verify"}
