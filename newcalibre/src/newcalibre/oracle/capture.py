"""Load and validate the compact frozen VN2 oracle capture."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ORACLE_TAG = "oracle-freeze-2026-07-06"
ORACLE_COMMIT = "686a1b284a4f4879123b4095d306f07b88d2ddc3"
ORACLE_LOCK_SHA256 = "5cc585d347195861d81760e16a675bd2a05b51777cf90c13d9af0ab05bb743f3"
ACTUALS_SEMANTICS = "censored_sales_surrogate"
EXPECTED_ROUNDS = 6
EXPECTED_SERIES = 599

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = frozenset(
    {
        "actuals_semantics",
        "config_sha256",
        "input_inventory_sha256",
        "oracle_commit",
        "oracle_lock_sha256",
        "oracle_tag",
        "orders",
        "schema",
    }
)
_ORDER_ENTRY_KEYS = frozenset({"path", "round", "sha256"})
_ORDER_PAYLOAD_KEYS = frozenset({"orders", "origin", "round_num"})


class OracleEvidenceError(ValueError):
    """Report malformed or identity-mismatched frozen oracle evidence."""


class _DuplicateKey(ValueError):
    """Retain a duplicate JSON key for a useful validation error."""


@dataclass(frozen=True, slots=True)
class CaptureFile:
    """Bind one oracle round to its canonical file digest."""

    round: int
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    """Expose the validated identities in the compact capture manifest."""

    oracle_tag: str
    oracle_commit: str
    oracle_lock_sha256: str
    config_sha256: str
    input_inventory_sha256: str
    actuals_semantics: str
    orders: tuple[CaptureFile, ...]


@dataclass(frozen=True, slots=True)
class CaptureBundle:
    """Return a validated capture root and its stable order-stream identity."""

    root: Path
    manifest: CaptureManifest
    manifest_sha256: str

    @property
    def capture_digest(self) -> str:
        """Return the stable digest of the six canonical order payloads."""
        listing = "".join(
            f"{entry.sha256}  {entry.path}\n" for entry in self.manifest.orders
        ).encode("utf-8")
        return _sha256(listing)


def load_capture(
    root: Path,
    *,
    config_path: Path,
    input_inventory_path: Path,
) -> CaptureBundle:
    """Load the canonical seven-file VN2 capture and validate every binding."""
    capture_root = Path(root)
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise OracleEvidenceError("capture root must be a real directory, not a symbolic link")
    expected_files = {
        "manifest.json",
        *(f"orders/round-{round_number}.json" for round_number in range(1, 7)),
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in capture_root.rglob("*"):
        relative = path.relative_to(capture_root).as_posix()
        if path.is_symlink():
            raise OracleEvidenceError(f"capture contains a symbolic link: {relative}")
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise OracleEvidenceError(f"capture contains a non-regular path: {relative}")
    if actual_directories != {"orders"} or actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise OracleEvidenceError(f"capture file set mismatch: missing={missing} extra={extra}")

    manifest_path = capture_root / "manifest.json"
    manifest_bytes = _read_regular(manifest_path, name="capture manifest")
    raw = _json_object(manifest_bytes, name="capture manifest")
    if set(raw) != _MANIFEST_KEYS or raw["schema"] != 1:
        raise OracleEvidenceError("capture manifest must use the exact schema 1 keys")
    expected_identities = {
        "oracle_tag": ORACLE_TAG,
        "oracle_commit": ORACLE_COMMIT,
        "oracle_lock_sha256": ORACLE_LOCK_SHA256,
        "actuals_semantics": ACTUALS_SEMANTICS,
    }
    for name, expected in expected_identities.items():
        if raw[name] != expected:
            raise OracleEvidenceError(f"capture {name} does not match the canonical identity")

    config_digest = _digest(raw["config_sha256"], name="config sha256")
    input_digest = _digest(raw["input_inventory_sha256"], name="input inventory sha256")
    if _sha256(_read_regular(Path(config_path), name="trusted config")) != config_digest:
        raise OracleEvidenceError("capture config digest does not match trusted config bytes")
    if (
        _sha256(_read_regular(Path(input_inventory_path), name="trusted input inventory"))
        != input_digest
    ):
        raise OracleEvidenceError(
            "capture input inventory digest does not match trusted input inventory bytes"
        )

    raw_orders = raw["orders"]
    if not isinstance(raw_orders, list) or len(raw_orders) != EXPECTED_ROUNDS:
        raise OracleEvidenceError("capture manifest must bind exactly six order files")
    entries: list[CaptureFile] = []
    for index, value in enumerate(raw_orders, start=1):
        if not isinstance(value, dict) or set(value) != _ORDER_ENTRY_KEYS:
            raise OracleEvidenceError(f"capture order entry {index} has invalid keys")
        entry = cast(dict[str, object], value)
        if entry["round"] != index or entry["path"] != f"orders/round-{index}.json":
            raise OracleEvidenceError("capture order entries must follow rounds 1 through 6")
        digest = _digest(entry["sha256"], name=f"round {index} digest")
        payload = _read_regular(capture_root / str(entry["path"]), name=f"round {index}")
        if _sha256(payload) != digest:
            raise OracleEvidenceError(f"round {index} order digest mismatch")
        _validate_order_payload(payload, round_number=index)
        entries.append(CaptureFile(round=index, path=str(entry["path"]), sha256=digest))

    return CaptureBundle(
        root=capture_root.resolve(),
        manifest=CaptureManifest(
            oracle_tag=ORACLE_TAG,
            oracle_commit=ORACLE_COMMIT,
            oracle_lock_sha256=ORACLE_LOCK_SHA256,
            config_sha256=config_digest,
            input_inventory_sha256=input_digest,
            actuals_semantics=ACTUALS_SEMANTICS,
            orders=tuple(entries),
        ),
        manifest_sha256=_sha256(manifest_bytes),
    )


def _validate_order_payload(payload: bytes, *, round_number: int) -> None:
    value = _json_object(payload, name=f"round {round_number} order payload")
    if set(value) != _ORDER_PAYLOAD_KEYS or value["round_num"] != round_number:
        raise OracleEvidenceError(f"round {round_number} order payload has the wrong identity")
    if not isinstance(value["origin"], str) or not value["origin"]:
        raise OracleEvidenceError(f"round {round_number} origin must be a string")
    orders = value["orders"]
    if not isinstance(orders, dict) or len(orders) != EXPECTED_SERIES:
        raise OracleEvidenceError(f"round {round_number} must contain 599 unique orders")
    for series_key, quantity in orders.items():
        if not isinstance(series_key, str) or not series_key:
            raise OracleEvidenceError(f"round {round_number} contains an invalid series key")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            raise OracleEvidenceError(f"round {round_number} contains a non-numeric order")
        if not math.isfinite(float(quantity)) or float(quantity) < 0.0:
            raise OracleEvidenceError(f"round {round_number} contains a negative order")


def _json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (_DuplicateKey, UnicodeError, json.JSONDecodeError) as error:
        raise OracleEvidenceError(f"{name} is not unique-key UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise OracleEvidenceError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _read_regular(path: Path, *, name: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise OracleEvidenceError(f"{name} must be a regular file")
        return path.read_bytes()
    except OSError as error:
        raise OracleEvidenceError(f"{name} is unreadable") from error


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OracleEvidenceError(f"{name} must be a lowercase sha256 digest")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACTUALS_SEMANTICS",
    "ORACLE_COMMIT",
    "ORACLE_LOCK_SHA256",
    "ORACLE_TAG",
    "CaptureBundle",
    "CaptureFile",
    "CaptureManifest",
    "OracleEvidenceError",
    "load_capture",
]
