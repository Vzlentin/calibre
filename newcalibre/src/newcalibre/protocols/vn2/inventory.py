"""Acquire and verify the exact approved VN2 input inventory."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

EXPECTED_INPUT_COUNT = 12
_INVENTORY_KEYS = frozenset(
    {
        "dataset",
        "files",
        "minted_run_id",
        "minted_sha",
        "schema",
        "source_manifest",
        "source_manifest_sha256",
    }
)
_FILE_KEYS = frozenset({"bytes", "name", "sha256"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

type ByteFetcher = Callable[[str], bytes]


class VN2InputError(ValueError):
    """Report malformed, unavailable, or integrity-invalid VN2 inputs."""


@dataclass(frozen=True, slots=True)
class VN2InputFile:
    """Describe one approved input file without providing a minting surface."""

    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VN2InputInventory:
    """Carry the immutable approved twelve-file VN2 inventory."""

    schema: int
    dataset: str
    source_manifest: str
    source_manifest_sha256: str
    minted_run_id: str
    minted_sha: str
    files: tuple[VN2InputFile, ...]

    @property
    def by_name(self) -> Mapping[str, VN2InputFile]:
        """Return an immutable name lookup derived from approved facts."""
        return {entry.name: entry for entry in self.files}


def load_vn2_inventory(path: Path) -> VN2InputInventory:
    """Load the approved inventory or reject its complete schema."""
    if not isinstance(path, Path):
        raise VN2InputError("inventory path must be a pathlib.Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VN2InputError(f"inventory {path} is not readable canonical JSON") from error
    if not isinstance(raw, dict) or set(raw) != _INVENTORY_KEYS:
        raise VN2InputError("inventory must contain the exact keys defined by schema 1")
    payload = cast(dict[str, object], raw)
    if payload["schema"] != 1:
        raise VN2InputError("inventory schema must equal 1")
    if payload["dataset"] != "vn2":
        raise VN2InputError("inventory dataset must equal 'vn2'")

    source_manifest = _nonempty_string(payload["source_manifest"], name="source_manifest")
    source_digest = _digest(payload["source_manifest_sha256"], name="source manifest sha256")
    run_id = _nonempty_string(payload["minted_run_id"], name="minted_run_id")
    if re.fullmatch(r"[1-9][0-9]*", run_id) is None:
        raise VN2InputError("inventory minted_run_id must be a positive decimal string")
    minted_sha = _nonempty_string(payload["minted_sha"], name="minted_sha")
    if _COMMIT_SHA.fullmatch(minted_sha) is None:
        raise VN2InputError("inventory minted_sha must be a lowercase full commit SHA")

    raw_files = payload["files"]
    if not isinstance(raw_files, list) or len(raw_files) != EXPECTED_INPUT_COUNT:
        count = len(raw_files) if isinstance(raw_files, list) else "non-list"
        raise VN2InputError(
            f"inventory must carry exactly {EXPECTED_INPUT_COUNT} files, found {count}"
        )
    files: list[VN2InputFile] = []
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_KEYS:
            raise VN2InputError(f"inventory file {index} must contain exact keys")
        file_payload = cast(dict[str, object], raw_file)
        name = _safe_csv_name(file_payload["name"])
        size = file_payload["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise VN2InputError(f"inventory file {name!r} bytes must be a positive integer")
        digest = _digest(file_payload["sha256"], name=f"{name} sha256")
        files.append(VN2InputFile(name=name, bytes=size, sha256=digest))
    names = [entry.name for entry in files]
    if len(set(names)) != len(names):
        raise VN2InputError("inventory file names must be unique")
    return VN2InputInventory(
        schema=1,
        dataset="vn2",
        source_manifest=source_manifest,
        source_manifest_sha256=source_digest,
        minted_run_id=run_id,
        minted_sha=minted_sha,
        files=tuple(files),
    )


def verify_vn2_inputs(target: Path, inventory_path: Path) -> VN2InputInventory:
    """Verify the exact directory entry set, byte sizes, and all sha256 values."""
    inventory = load_vn2_inventory(inventory_path)
    if not isinstance(target, Path) or not target.is_dir():
        raise VN2InputError(f"VN2 input directory does not exist: {target}")
    expected = set(inventory.by_name)
    try:
        present = {entry.name for entry in target.iterdir()}
    except OSError as error:
        raise VN2InputError(f"VN2 input directory cannot be read: {target}") from error
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise VN2InputError(f"file-set mismatch: missing={missing} extra={extra}")
    for entry in inventory.files:
        _verified_file_bytes(target / entry.name, entry)
    return inventory


def read_verified_vn2_input(
    target: Path,
    name: str,
    inventory_path: Path,
) -> bytes:
    """Verify the complete directory, then reverify bytes immediately before parsing."""
    inventory = verify_vn2_inputs(target, inventory_path)
    entry = inventory.by_name.get(name)
    if entry is None:
        raise VN2InputError(f"input {name!r} is absent from the approved inventory")
    return _verified_file_bytes(target / name, entry)


def download_vn2_inputs(
    target: Path,
    sources: Mapping[str, str],
    inventory_path: Path,
    *,
    if_missing: bool = False,
    fetcher: ByteFetcher | None = None,
) -> VN2InputInventory:
    """Fetch approved names and unconditionally verify the consumed directory.

    ``sources`` is caller-owned acquisition data. The successor never reads a
    frozen benchmark manifest, and this module deliberately exposes no digest
    mint operation.
    """
    inventory = load_vn2_inventory(inventory_path)
    if not isinstance(sources, Mapping):
        raise VN2InputError("download sources must be a name-to-URL mapping")
    source_snapshot = dict(sources)
    expected = set(inventory.by_name)
    supplied = set(source_snapshot)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise VN2InputError(f"source names mismatch: missing={missing} extra={extra}")
    if any(not isinstance(url, str) or not url for url in source_snapshot.values()):
        raise VN2InputError("every VN2 source URL must be a non-empty string")
    if not isinstance(if_missing, bool):
        raise VN2InputError("if_missing must be a boolean")
    if not isinstance(target, Path):
        raise VN2InputError("target must be a pathlib.Path")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VN2InputError(f"cannot create VN2 input directory: {target}") from error

    fetch = fetcher or _fetch_url
    if not callable(fetch):
        raise VN2InputError("fetcher must be callable")
    for entry in inventory.files:
        destination = target / entry.name
        if if_missing and destination.exists():
            continue
        try:
            payload = fetch(source_snapshot[entry.name])
        except Exception as error:
            raise VN2InputError(f"download failed for {entry.name}: {error}") from error
        if not isinstance(payload, bytes):
            raise VN2InputError(f"download for {entry.name} did not return bytes")
        _verify_bytes(payload, entry)
        temporary = target / f".{entry.name}.part"
        try:
            temporary.write_bytes(payload)
            temporary.replace(destination)
        except OSError as error:
            raise VN2InputError(f"cannot install downloaded input {entry.name}") from error
        finally:
            temporary.unlink(missing_ok=True)
    return verify_vn2_inputs(target, inventory_path)


def _fetch_url(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except OSError as error:
        raise VN2InputError(f"source URL is unavailable: {url}") from error


def _verified_file_bytes(path: Path, entry: VN2InputFile) -> bytes:
    try:
        if not path.is_file():
            raise VN2InputError(f"{entry.name}: expected a regular file")
        payload = path.read_bytes()
    except OSError as error:
        raise VN2InputError(f"{entry.name}: could not read input bytes") from error
    _verify_bytes(payload, entry)
    return payload


def _verify_bytes(payload: bytes, entry: VN2InputFile) -> None:
    if len(payload) != entry.bytes:
        raise VN2InputError(f"{entry.name}: size {len(payload)} != inventory {entry.bytes}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry.sha256:
        raise VN2InputError(f"{entry.name}: sha256 {digest} != inventory {entry.sha256}")


def _safe_csv_name(value: object) -> str:
    name = _nonempty_string(value, name="file name")
    if Path(name).name != name or name in {".", ".."} or not name.endswith(".csv"):
        raise VN2InputError(f"inventory file name {name!r} must be a safe CSV basename")
    return name


def _digest(value: object, *, name: str) -> str:
    digest = _nonempty_string(value, name=name)
    if _SHA256.fullmatch(digest) is None:
        raise VN2InputError(f"{name} must be a lowercase sha256 hex digest")
    return digest


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise VN2InputError(f"inventory {name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise VN2InputError(f"inventory {name} must be valid UTF-8") from error
    return value
