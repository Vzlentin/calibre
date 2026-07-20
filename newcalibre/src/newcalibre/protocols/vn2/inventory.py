"""Acquire and verify the exact approved VN2 input inventory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
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


class _DuplicateJSONKey(ValueError):
    """Retain the duplicate key refused by the strict JSON decoder."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


@dataclass(frozen=True, slots=True)
class VN2InputFile:
    """Describe one approved input file without providing a minting surface."""

    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VN2InputInventory:
    """Carry one immutable approved VN2 input inventory."""

    schema: int
    dataset: str
    content_sha256: str
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
    raw, raw_bytes = _load_unique_json_payload(path, subject=f"inventory {path}")
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
    if not isinstance(raw_files, list) or not raw_files:
        raise VN2InputError("inventory files must be a non-empty list")
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
        content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
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
    inventory: VN2InputInventory,
) -> bytes:
    """Return bytes reverified immediately before one parser consumes them."""
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

    if fetcher is not None and not callable(fetcher):
        raise VN2InputError("fetcher must be callable")
    for entry in inventory.files:
        destination = target / entry.name
        if if_missing and destination.exists():
            continue
        try:
            if fetcher is None:
                payload = _fetch_url(source_snapshot[entry.name], max_bytes=entry.bytes)
            else:
                payload = fetcher(source_snapshot[entry.name])
        except Exception as error:
            raise VN2InputError(f"download failed for {entry.name}: {error}") from error
        if not isinstance(payload, bytes):
            raise VN2InputError(f"download for {entry.name} did not return bytes")
        _verify_bytes(payload, entry)
        _install_download(destination, payload, name=entry.name)
    return verify_vn2_inputs(target, inventory_path)


def load_unique_json(path: Path, *, subject: str) -> object:
    """Load UTF-8 JSON while refusing duplicate object keys at every depth."""
    value, _raw_bytes = _load_unique_json_payload(path, subject=subject)
    return value


def _load_unique_json_payload(path: Path, *, subject: str) -> tuple[object, bytes]:
    """Return strict decoded JSON together with the exact consumed bytes."""
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except _DuplicateJSONKey as error:
        raise VN2InputError(f"{subject} contains duplicate JSON key {error.key!r}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VN2InputError(f"{subject} is not readable canonical JSON") from error
    return value, raw_bytes


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def _fetch_url(url: str, *, max_bytes: int) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read(max_bytes + 1)
    except OSError as error:
        raise VN2InputError(f"source URL is unavailable: {url}") from error


def _install_download(destination: Path, payload: bytes, *, name: str) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    install_error: OSError | None = None
    cleanup_errors: list[str] = []
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as error:
        install_error = error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                cleanup_errors.append(f"file descriptor: {error}")
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                cleanup_errors.append(f"temporary file: {error}")
    if install_error is not None:
        message = f"cannot install downloaded input {name}: {install_error}"
        if cleanup_errors:
            message += f"; cleanup failed ({'; '.join(cleanup_errors)})"
        raise VN2InputError(message) from install_error
    if cleanup_errors:
        raise VN2InputError(f"cannot clean up downloaded input {name}: {'; '.join(cleanup_errors)}")


def _verified_file_bytes(path: Path, entry: VN2InputFile) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
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
