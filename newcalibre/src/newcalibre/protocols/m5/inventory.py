"""Verify only the compact consumed M5 input inventory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

_EXPECTED_NAMES = frozenset({"calendar.csv", "sales_train_evaluation.csv"})
_INVENTORY_KEYS = frozenset({"schema", "dataset", "files"})
_FILE_KEYS = frozenset({"bytes", "name", "sha256"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


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
    """Verify the exact directory entry set, byte sizes, and SHA-256 digests."""
    inventory = load_m5_inventory(inventory_path)
    if not isinstance(target, Path) or not target.is_dir():
        raise M5InputError(f"M5 input directory does not exist: {target}")
    expected = set(inventory.by_name)
    try:
        present = {entry.name for entry in target.iterdir()}
    except OSError as error:
        raise M5InputError(f"M5 input directory cannot be read: {target}") from error
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise M5InputError(f"file-set mismatch: missing={missing} extra={extra}")
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
