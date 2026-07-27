"""Acquire and verify only the compact consumed M5 input inventory."""

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
from types import MappingProxyType
from typing import cast

_EXPECTED_NAMES = frozenset({"calendar.csv", "sales_train_evaluation.csv"})
_INVENTORY_KEYS = frozenset({"schema", "dataset", "files"})
_FILE_KEYS = frozenset({"bytes", "name", "sha256"})
_SHA256 = re.compile(r"[0-9a-f]{64}")

type ByteFetcher = Callable[[str], bytes]


class M5InputError(ValueError):
    """Report malformed, unavailable, or integrity-invalid M5 inputs."""


class _DuplicateJSONKey(ValueError):
    """Retain the duplicate key refused by the strict JSON decoder."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


@dataclass(frozen=True, slots=True)
class M5InputFile:
    """Describe one approved consumed input file."""

    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class M5InputInventory:
    """Carry one immutable approved M5 input inventory."""

    schema: int
    dataset: str
    content_sha256: str
    files: tuple[M5InputFile, ...]

    @property
    def by_name(self) -> Mapping[str, M5InputFile]:
        """Return an immutable name lookup derived from approved facts."""
        return MappingProxyType({entry.name: entry for entry in self.files})


def load_m5_inventory(path: Path) -> M5InputInventory:
    """Load the compact approved inventory or reject its complete schema."""
    if not isinstance(path, Path):
        raise M5InputError("inventory path must be a pathlib.Path")
    raw, raw_bytes = _load_unique_json_payload(path, subject=f"inventory {path}")
    if not isinstance(raw, dict) or set(raw) != _INVENTORY_KEYS:
        raise M5InputError("inventory must contain the exact keys defined by schema 1")
    payload = cast(dict[str, object], raw)
    if payload["schema"] != 1:
        raise M5InputError("inventory schema must equal 1")
    if payload["dataset"] != "m5":
        raise M5InputError("inventory dataset must equal 'm5'")

    raw_files = payload["files"]
    if not isinstance(raw_files, list):
        raise M5InputError("inventory files must be a list")
    files: list[M5InputFile] = []
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_KEYS:
            raise M5InputError(f"inventory file {index} must contain exact keys")
        file_payload = cast(dict[str, object], raw_file)
        name = _safe_csv_name(file_payload["name"])
        size = file_payload["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise M5InputError(f"inventory file {name!r} bytes must be a positive integer")
        digest = _digest(file_payload["sha256"], name=f"{name} sha256")
        files.append(M5InputFile(name, size, digest))
    names = [entry.name for entry in files]
    if len(set(names)) != len(names):
        raise M5InputError("inventory file names must be unique")
    if set(names) != _EXPECTED_NAMES or len(names) != len(_EXPECTED_NAMES):
        missing = sorted(_EXPECTED_NAMES - set(names))
        extra = sorted(set(names) - _EXPECTED_NAMES)
        raise M5InputError(
            f"inventory must name the exact consumed files: missing={missing} extra={extra}"
        )
    return M5InputInventory(
        schema=1,
        dataset="m5",
        content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        files=tuple(files),
    )


def verify_m5_inputs(target: Path, inventory_path: Path) -> M5InputInventory:
    """Verify each consumed regular file's exact byte size and SHA-256 digest."""
    inventory = load_m5_inventory(inventory_path)
    if not isinstance(target, Path) or not target.is_dir():
        raise M5InputError(f"M5 input directory does not exist: {target}")
    missing = sorted(name for name in inventory.by_name if not (target / name).exists())
    if missing:
        raise M5InputError(f"consumed M5 inputs are missing: {missing}")
    for entry in inventory.files:
        _verified_file_bytes(target / entry.name, entry)
    return inventory


def read_verified_m5_input(
    target: Path,
    name: str,
    inventory: M5InputInventory,
) -> bytes:
    """Return bytes reverified immediately before one CSV parser consumes them."""
    if not isinstance(target, Path):
        raise M5InputError("target must be a pathlib.Path")
    if not isinstance(inventory, M5InputInventory):
        raise M5InputError("inventory must be an M5InputInventory")
    entry = inventory.by_name.get(name)
    if entry is None:
        raise M5InputError(f"input {name!r} is absent from the approved inventory")
    return _verified_file_bytes(target / name, entry)


def download_m5_inputs(
    target: Path,
    sources: Mapping[str, str],
    inventory_path: Path,
    *,
    fetcher: ByteFetcher | None = None,
) -> M5InputInventory:
    """Fetch approved names, install verified bytes, and verify consumed files."""
    inventory = load_m5_inventory(inventory_path)
    if not isinstance(sources, Mapping):
        raise M5InputError("download sources must be a name-to-URL mapping")
    source_snapshot = dict(sources)
    expected = set(inventory.by_name)
    supplied = set(source_snapshot)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise M5InputError(f"source names mismatch: missing={missing} extra={extra}")
    if any(
        not isinstance(url, str) or not url or url != url.strip()
        for url in source_snapshot.values()
    ):
        raise M5InputError("every M5 source URL must be a non-empty trimmed string")
    if not isinstance(target, Path):
        raise M5InputError("target must be a pathlib.Path")
    if fetcher is not None and not callable(fetcher):
        raise M5InputError("fetcher must be callable")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise M5InputError(f"cannot create M5 input directory: {target}") from error

    for entry in inventory.files:
        url = source_snapshot[entry.name]
        try:
            payload = _fetch_url(url, max_bytes=entry.bytes) if fetcher is None else fetcher(url)
        except Exception as error:
            raise M5InputError(f"download failed for {entry.name}: {error}") from error
        if not isinstance(payload, bytes):
            raise M5InputError(f"download for {entry.name} did not return bytes")
        _verify_bytes(payload, entry)
        _install_download(target / entry.name, payload, name=entry.name)
    return verify_m5_inputs(target, inventory_path)


def load_unique_json(path: Path, *, subject: str) -> object:
    """Load UTF-8 JSON while refusing duplicate object keys at every depth."""
    value, _raw_bytes = _load_unique_json_payload(path, subject=subject)
    return value


def _load_unique_json_payload(path: Path, *, subject: str) -> tuple[object, bytes]:
    if not isinstance(path, Path):
        raise M5InputError(f"{subject} path must be a pathlib.Path")
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except _DuplicateJSONKey as error:
        raise M5InputError(f"{subject} contains duplicate JSON key {error.key!r}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M5InputError(f"{subject} is not readable JSON") from error
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
        raise M5InputError(f"source URL is unavailable: {url}") from error


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
        raise M5InputError(message) from install_error
    if cleanup_errors:
        raise M5InputError(f"cannot clean up downloaded input {name}: {'; '.join(cleanup_errors)}")


def _verified_file_bytes(path: Path, entry: M5InputFile) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise M5InputError(f"{entry.name}: expected a regular file")
        payload = path.read_bytes()
    except OSError as error:
        raise M5InputError(f"{entry.name}: could not read input bytes") from error
    _verify_bytes(payload, entry)
    return payload


def _verify_bytes(payload: bytes, entry: M5InputFile) -> None:
    if len(payload) != entry.bytes:
        raise M5InputError(f"{entry.name}: size {len(payload)} != inventory {entry.bytes}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry.sha256:
        raise M5InputError(f"{entry.name}: sha256 {digest} != inventory {entry.sha256}")


def _safe_csv_name(value: object) -> str:
    name = _nonempty_string(value, name="file name")
    if Path(name).name != name or name in {".", ".."} or not name.endswith(".csv"):
        raise M5InputError(f"inventory file name {name!r} must be a safe CSV basename")
    return name


def _digest(value: object, *, name: str) -> str:
    digest = _nonempty_string(value, name=name)
    if _SHA256.fullmatch(digest) is None:
        raise M5InputError(f"{name} must be a lowercase SHA-256 hex digest")
    return digest


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise M5InputError(f"inventory {name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise M5InputError(f"inventory {name} must be valid UTF-8") from error
    return value
