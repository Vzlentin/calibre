"""Build, compare, and append compact VN2 regression records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.protocols.vn2.artifacts import PLATFORM, VN2ResultBundle

TRACKING_SCHEMA = 2
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LEGACY_RECORD_SHA256 = "5f094f8e1c10c2528671281e5435061544e2378fc44ba5ff6c5e82935dec179c"
_COMPACT_CAPTURE_DIGEST = "16f86c7cbe2d39b51346b8cb2b02bf434c9f1ea5da0c73186629a68803f33904"
_RECORD_KEYS = frozenset(
    {
        "actuals_semantics",
        "candidate_sha",
        "capture_digest",
        "config_digest",
        "holding_cost",
        "input_inventory_digest",
        "lock_digest",
        "platform",
        "result_manifest_digest",
        "schema",
        "shortage_cost",
        "total_cost",
    }
)
_COMPARABILITY_FIELDS = (
    "config_digest",
    "input_inventory_digest",
    "capture_digest",
    "lock_digest",
    "platform",
    "actuals_semantics",
)


class TrackingError(ValueError):
    """Report malformed, incomparable, or non-append-only tracking data."""


@dataclass(frozen=True, slots=True)
class VN2TrackingRecord:
    """Carry one canonical compact VN2 regression observation."""

    candidate_sha: str
    config_digest: str
    input_inventory_digest: str
    capture_digest: str
    lock_digest: str
    platform: str
    actuals_semantics: str
    result_manifest_digest: str
    holding_cost: float
    shortage_cost: float
    total_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_sha", _commit(self.candidate_sha, name="candidate"))
        for name in (
            "config_digest",
            "input_inventory_digest",
            "capture_digest",
            "lock_digest",
            "result_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if not isinstance(self.platform, str) or not self.platform:
            raise TrackingError("platform must be a non-empty string")
        if not isinstance(self.actuals_semantics, str) or not self.actuals_semantics:
            raise TrackingError("actuals_semantics must be a non-empty string")
        holding = _cost(self.holding_cost, name="holding_cost")
        shortage = _cost(self.shortage_cost, name="shortage_cost")
        total = _cost(self.total_cost, name="total_cost")
        if total != holding + shortage:
            raise TrackingError("total_cost must equal holding plus shortage")
        object.__setattr__(self, "holding_cost", holding)
        object.__setattr__(self, "shortage_cost", shortage)
        object.__setattr__(self, "total_cost", total)

    @property
    def comparability_key(self) -> tuple[str, ...]:
        """Return the exact six-field identity required for cost comparison."""
        return tuple(cast(str, getattr(self, name)) for name in _COMPARABILITY_FIELDS)

    def to_bytes(self) -> bytes:
        """Serialize one record as canonical LF-terminated JSONL."""
        return _canonical_bytes(_record_value(self))


@dataclass(frozen=True, slots=True)
class TrackingComparison:
    """Expose signed candidate-minus-baseline cost changes."""

    holding_delta: float
    shortage_delta: float
    total_delta: float


def build_tracking_record(bundle: VN2ResultBundle) -> VN2TrackingRecord:
    """Reduce one validated result bundle to a compact tracking record."""
    if not isinstance(bundle, VN2ResultBundle):
        raise TrackingError("tracking projection requires a validated VN2ResultBundle")
    manifest = bundle.manifest
    return VN2TrackingRecord(
        candidate_sha=manifest.candidate_sha,
        config_digest=manifest.config_digest,
        input_inventory_digest=manifest.input_inventory_digest,
        capture_digest=manifest.capture_digest,
        lock_digest=manifest.lock_digest,
        platform=manifest.platform,
        actuals_semantics=manifest.actuals_semantics,
        result_manifest_digest=bundle.manifest_sha256,
        holding_cost=bundle.holding_cost,
        shortage_cost=bundle.shortage_cost,
        total_cost=bundle.total_cost,
    )


def load_tracking_history(
    value: bytes | bytearray | str | Path,
) -> tuple[VN2TrackingRecord, ...]:
    """Load canonical compact JSONL records in their exact append order."""
    payload = _tracking_bytes(value)
    if not payload:
        return ()
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise TrackingError("tracking history must use LF-terminated JSONL")
    records: list[VN2TrackingRecord] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TrackingError(f"tracking line {index} is not UTF-8 JSON") from error
        encoded_line = line + b"\n"
        if hashlib.sha256(encoded_line).hexdigest() == _LEGACY_RECORD_SHA256:
            if index != 1 or not isinstance(raw, dict):
                raise TrackingError("the historical Gate A record must remain first")
            record = _legacy_record(cast(dict[str, object], raw))
        else:
            if (
                not isinstance(raw, dict)
                or set(raw) != _RECORD_KEYS
                or raw["schema"] != TRACKING_SCHEMA
            ):
                raise TrackingError(f"tracking line {index} does not use compact schema 2")
            record = VN2TrackingRecord(
                candidate_sha=raw["candidate_sha"],
                config_digest=raw["config_digest"],
                input_inventory_digest=raw["input_inventory_digest"],
                capture_digest=raw["capture_digest"],
                lock_digest=raw["lock_digest"],
                platform=raw["platform"],
                actuals_semantics=raw["actuals_semantics"],
                result_manifest_digest=raw["result_manifest_digest"],
                holding_cost=raw["holding_cost"],
                shortage_cost=raw["shortage_cost"],
                total_cost=raw["total_cost"],
            )
            if record.to_bytes() != encoded_line:
                raise TrackingError(f"tracking line {index} is not canonical JSON")
        records.append(record)
    candidates = [record.candidate_sha for record in records]
    if len(set(candidates)) != len(candidates):
        raise TrackingError("tracking candidate SHAs must be unique")
    return tuple(records)


def compare_tracking_records(
    baseline: VN2TrackingRecord,
    candidate: VN2TrackingRecord,
) -> TrackingComparison:
    """Compare costs only when the exact six-field comparability key matches."""
    if not isinstance(baseline, VN2TrackingRecord) or not isinstance(candidate, VN2TrackingRecord):
        raise TrackingError("comparison requires two VN2TrackingRecord values")
    if baseline.comparability_key != candidate.comparability_key:
        raise TrackingError("tracking records have a comparability-key mismatch")
    return TrackingComparison(
        holding_delta=candidate.holding_cost - baseline.holding_cost,
        shortage_delta=candidate.shortage_cost - baseline.shortage_cost,
        total_delta=candidate.total_cost - baseline.total_cost,
    )


def validate_tracking_append(
    base: bytes | bytearray | str | Path,
    head: bytes | bytearray | str | Path,
) -> tuple[VN2TrackingRecord, ...]:
    """Require the proposed history to preserve every base byte as an exact prefix."""
    base_bytes = _tracking_bytes(base)
    head_bytes = _tracking_bytes(head)
    base_records = load_tracking_history(base_bytes)
    head_records = load_tracking_history(head_bytes)
    if not head_bytes.startswith(base_bytes) or head_records[: len(base_records)] != base_records:
        raise TrackingError("tracking history update is not an exact append")
    return head_records[len(base_records) :]


def _legacy_record(raw: dict[str, object]) -> VN2TrackingRecord:
    subject = _object(raw.get("subject"), name="legacy subject")
    evidence = _object(raw.get("evidence"), name="legacy evidence")
    environment = _object(raw.get("environment"), name="legacy environment")
    facts = _object(environment.get("facts"), name="legacy environment facts")
    os_facts = _object(facts.get("os"), name="legacy OS")
    objective = _object(raw.get("objective"), name="legacy objective")
    result_bundle = _object(raw.get("result_bundle"), name="legacy result bundle")
    config = _object(evidence.get("config"), name="legacy config")
    inventory = _object(evidence.get("input_inventory"), name="legacy input inventory")
    lockfile = _object(evidence.get("lockfile"), name="legacy lockfile")
    _object(evidence.get("promoted_capture"), name="legacy capture")
    return VN2TrackingRecord(
        candidate_sha=cast(str, subject.get("candidate_sha")),
        config_digest=cast(str, config.get("digest")),
        input_inventory_digest=cast(str, inventory.get("digest")),
        capture_digest=_COMPACT_CAPTURE_DIGEST,
        lock_digest=cast(str, lockfile.get("digest")),
        platform=(f"{os_facts.get('id')}-{os_facts.get('version_id')}/{facts.get('arch')}"),
        actuals_semantics=cast(str, evidence.get("actuals_semantics")),
        result_manifest_digest=cast(str, result_bundle.get("manifest_sha256")),
        holding_cost=cast(float, objective.get("holding_cost")),
        shortage_cost=cast(float, objective.get("shortage_cost")),
        total_cost=cast(float, objective.get("total_cost")),
    )


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrackingError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _record_value(record: VN2TrackingRecord) -> dict[str, object]:
    return {
        "actuals_semantics": record.actuals_semantics,
        "candidate_sha": record.candidate_sha,
        "capture_digest": record.capture_digest,
        "config_digest": record.config_digest,
        "holding_cost": record.holding_cost,
        "input_inventory_digest": record.input_inventory_digest,
        "lock_digest": record.lock_digest,
        "platform": record.platform,
        "result_manifest_digest": record.result_manifest_digest,
        "schema": TRACKING_SCHEMA,
        "shortage_cost": record.shortage_cost,
        "total_cost": record.total_cost,
    }


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value, path="VN2 tracking record") + b"\n"
    except CanonicalJsonError as error:
        raise TrackingError("tracking record is not canonical JSON") from error


def _tracking_bytes(value: bytes | bytearray | str | Path) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Path):
        try:
            if value.is_symlink() or not value.is_file():
                raise TrackingError("tracking history path must be a regular file")
            return value.read_bytes()
        except OSError as error:
            raise TrackingError("tracking history path is unreadable") from error
    raise TrackingError("tracking history must be bytes, text, or a pathlib.Path")


def _commit(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise TrackingError(f"{name} must be a lowercase full commit SHA")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrackingError(f"{name} must be a lowercase sha256 digest")
    return value


def _cost(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TrackingError(f"{name} must be finite and non-negative")
    return result


__all__ = [
    "PLATFORM",
    "TRACKING_SCHEMA",
    "TrackingComparison",
    "TrackingError",
    "VN2TrackingRecord",
    "build_tracking_record",
    "compare_tracking_records",
    "load_tracking_history",
    "validate_tracking_append",
]
