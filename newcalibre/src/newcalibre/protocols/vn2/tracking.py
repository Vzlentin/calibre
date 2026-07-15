"""Strict VN2 Gate-A tracking records and proposal-only persistence.

The tracking surface is deliberately successor-only: records are derived from
validated VN2 result/capture evidence and contain no frozen-engine baseline.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlparse

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.oracle import (
    CaptureBundle,
    CaptureReceipt,
    validate_committed_promoted_capture,
)
from newcalibre.protocols.vn2.artifacts import (
    CONFIG_PATH,
    INPUT_INVENTORY_PATH,
    LOCK_PATH,
    RESULT_KIND,
    VN2EvidenceEnvironment,
    VN2ResultBundle,
    validate_vn2_result_bundle,
)

TRACKING_SCHEMA = 1
TRACKING_KIND = "vn2-gate-a-tracking-record"
REPOSITORY = "Vzlentin/calibre"
_CAPTURE_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[1-9][0-9]*")
_TOP_KEYS = frozenset(
    {
        "schema",
        "record_kind",
        "identity",
        "subject",
        "workflow",
        "result_artifact",
        "result_bundle",
        "evidence",
        "environment",
        "objective",
    }
)
_SUBJECT_KEYS = frozenset({"repository", "candidate_sha"})
_WORKFLOW_KEYS = frozenset({"definition_ref", "definition_sha", "run_id", "run_url"})
_ARTIFACT_KEYS = frozenset({"id", "name", "digest"})
_BUNDLE_KEYS = frozenset(
    {"artifact_kind", "manifest_sha256", "inner_bundle_digest", "provenance_digest", "files"}
)
_FILE_NAMES = frozenset(
    {
        "environment.json",
        "r1-orders.jsonl",
        "r2-cost-ledger.jsonl",
        "r3-final-triple.json",
        "r4-cost-trajectory.json",
    }
)
_EVIDENCE_KEYS = frozenset(
    {"config", "input_inventory", "lockfile", "promoted_capture", "actuals_semantics", "session"}
)
_PATH_DIGEST_KEYS = frozenset({"path", "digest"})
_CAPTURE_KEYS = frozenset(
    {
        "artifact_id",
        "artifact_digest",
        "artifact_name",
        "capture_digest",
        "manifest_sha256",
        "inner_bundle_digest",
        "environment_digest",
        "producer_sha",
        "workflow_sha",
        "workflow_run_id",
        "run_url",
    }
)
_SESSION_KEYS = frozenset({"id", "series_count", "series_identity_digest"})
_ENVIRONMENT_KEYS = frozenset({"facts", "digest", "toolchain_digest"})
_OBJECTIVE_KEYS = frozenset({"holding_cost", "shortage_cost", "total_cost"})
_GA1_FIELDS = (
    "architecture",
    "os_id",
    "os_version",
    "lockfile_digest",
    "config_digest",
    "input_inventory_digest",
    "promoted_capture_digest",
    "actuals_semantics",
)


class TrackingError(ValueError):
    """Report malformed, untrusted, or conflicting tracking evidence."""


@dataclass(frozen=True, slots=True)
class VN2TrackingRecord:
    """One validated canonical v1 tracking record."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        value = _validate_payload(dict(self.payload))
        object.__setattr__(self, "payload", MappingProxyType(value))

    @property
    def identity(self) -> str:
        return cast(str, self.payload["identity"])

    @property
    def total_cost(self) -> float:
        return _finite_cost(
            cast(Mapping[str, object], self.payload["objective"])["total_cost"],
            name="objective.total_cost",
        )

    def to_bytes(self) -> bytes:
        return _canonical_record_bytes(self.payload)

    def to_json(self) -> str:
        return self.to_bytes().decode("utf-8")

    def __getitem__(self, key: str) -> object:
        return self.payload[key]


TrackingRecord = VN2TrackingRecord


@dataclass(frozen=True, slots=True)
class TrackingComparison:
    """Return the exact GA1 comparison result for two records."""

    comparable: bool
    mismatched_fields: tuple[str, ...]
    total_cost_delta: float | None
    cost_jump_detected: bool | None


@dataclass(frozen=True, slots=True)
class AppendDecision:
    """Describe whether a proposed record may be appended to history."""

    action: Literal["append", "noop", "conflict"]
    record: VN2TrackingRecord

    @property
    def should_append(self) -> bool:
        return self.action == "append"


def parse_tracking_record(value: bytes | bytearray | str | Path) -> VN2TrackingRecord:
    """Parse exactly one canonical JSONL record."""
    payload = _read_bytes(value, name="tracking record")
    if not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
        raise TrackingError("tracking record must end with exactly one LF")
    body = payload[:-1]
    if not body or b"\n" in body or b"\r" in body:
        raise TrackingError("tracking record must contain exactly one JSON object line")
    parsed = _parse_json(body, name="tracking record")
    if _canonical_record_bytes(parsed) != payload:
        raise TrackingError("tracking record bytes are not canonical")
    return VN2TrackingRecord(parsed)


def parse_tracking_history(value: bytes | bytearray | str | Path) -> tuple[VN2TrackingRecord, ...]:
    """Parse canonical history rows and reject repeated identities."""
    payload = _read_bytes(value, name="tracking history")
    if not payload:
        raise TrackingError("tracking history must contain at least one record")
    if payload.endswith(b"\r\n") or b"\r" in payload:
        raise TrackingError("tracking history must use LF-only line endings")
    rows = payload.split(b"\n")
    if rows[-1] != b"":
        raise TrackingError("tracking history must end with LF")
    rows = rows[:-1]
    if not rows or any(not row for row in rows):
        raise TrackingError("tracking history contains a blank line")
    records = tuple(parse_tracking_record(row + b"\n") for row in rows)
    identities = [record.identity for record in records]
    if len(set(identities)) != len(identities):
        raise TrackingError("tracking history contains a duplicate identity")
    return records


def write_tracking_record(record: VN2TrackingRecord, path: Path) -> bool:
    """Atomically write one record; return False for an exact existing replay."""
    destination = _validate_output_path(Path(path))
    payload = record.to_bytes()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise TrackingError("tracking proposal destination must be a regular non-symlink file")
        try:
            existing = destination.read_bytes()
        except OSError as error:
            raise TrackingError("tracking proposal destination is unreadable") from error
        if existing == payload:
            return False
        raise TrackingError("tracking proposal destination conflicts with the proposed record")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as error:
        raise TrackingError("tracking proposal publication failed") from error
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    return True


def write_proposal_record(record: VN2TrackingRecord, path: Path) -> bool:
    """Publish a proposal only beneath a real ``newcalibre/artifacts`` root."""
    destination = Path(path)
    parts = destination.parts
    try:
        marker = parts.index("newcalibre")
    except ValueError as error:
        raise TrackingError("proposal path must be beneath newcalibre/artifacts") from error
    if marker + 1 >= len(parts) or parts[marker + 1] != "artifacts":
        raise TrackingError("proposal path must be beneath newcalibre/artifacts")
    return write_tracking_record(record, destination)


def decide_append(
    record: VN2TrackingRecord, history: Sequence[VN2TrackingRecord]
) -> AppendDecision:
    """Return append/no-op/conflict semantics for one valid history."""
    for existing in history:
        if existing.identity != record.identity:
            continue
        if existing.to_bytes() == record.to_bytes():
            return AppendDecision("noop", record)
        return AppendDecision("conflict", record)
    return AppendDecision("append", record)


append_decision = decide_append


def compare_tracking_records(
    current: VN2TrackingRecord,
    prior: VN2TrackingRecord,
) -> TrackingComparison:
    """Compare costs only when every exact GA1 field matches."""
    current_key = _comparability_key(current.payload)
    prior_key = _comparability_key(prior.payload)
    mismatches = tuple(field for field in _GA1_FIELDS if current_key[field] != prior_key[field])
    if mismatches:
        return TrackingComparison(False, mismatches, None, None)
    delta = current.total_cost - prior.total_cost
    return TrackingComparison(True, (), delta, delta > 0.0)


compare_records = compare_tracking_records


def build_tracking_record(
    result_root: Path,
    capture_root: Path,
    *,
    candidate_sha: str,
    definition_ref: str,
    definition_sha: str,
    run_id: str,
    run_url: str,
    result_artifact_id: str | int,
    result_artifact_name: str,
    result_artifact_digest: str,
    config_path: Path = Path(CONFIG_PATH),
    input_inventory_path: Path = Path(INPUT_INVENTORY_PATH),
    lockfile_path: Path = Path(LOCK_PATH),
) -> VN2TrackingRecord:
    """Validate evidence and derive one complete canonical tracking record."""
    candidate_sha = _commit_sha(candidate_sha, name="candidate_sha")
    definition_sha = _commit_sha(definition_sha, name="workflow.definition_sha")
    run_id = _run_id(run_id, name="workflow.run_id")
    run_url = _run_url(run_url, run_id=run_id)
    result_artifact_id = _run_id(str(result_artifact_id), name="result_artifact.id")
    result_artifact_name = _text(result_artifact_name, name="result_artifact.name")
    result_artifact_digest = _normalized_digest(
        result_artifact_digest,
        name="result_artifact.digest",
    )
    result = validate_vn2_result_bundle(
        Path(result_root),
        expected_candidate_sha=candidate_sha,
        expected_workflow_sha=definition_sha,
        expected_run_id=run_id,
        expected_config_path=Path(config_path),
        expected_input_inventory_path=Path(input_inventory_path),
        expected_lock_path=Path(lockfile_path),
    )
    if not isinstance(result, VN2ResultBundle):
        raise TrackingError("VN2 result validator returned an invalid bundle")
    if result.manifest.artifact_name != result_artifact_name:
        raise TrackingError("result artifact name does not match the validated bundle")
    capture, receipt = validate_committed_promoted_capture(Path(capture_root))
    manifest = result.manifest
    if manifest.run_url != run_url:
        raise TrackingError("workflow.run_url does not match the validated result bundle")
    result_files = {entry.path: entry.sha256 for entry in manifest.files}
    environment = _environment_payload(manifest.environment)
    toolchain_digest = _toolchain_digest(environment)
    promoted = _capture_payload(capture, receipt)
    evidence = {
        "actuals_semantics": manifest.actuals_semantics,
        "config": {"digest": manifest.config_digest, "path": manifest.config_path},
        "input_inventory": {
            "digest": manifest.input_inventory_digest,
            "path": manifest.input_inventory_path,
        },
        "lockfile": {"digest": manifest.lock_digest, "path": manifest.lock_path},
        "promoted_capture": promoted,
        "session": {
            "id": manifest.session_id,
            "series_count": manifest.series_count,
            "series_identity_digest": manifest.series_identity_digest,
        },
    }
    payload: dict[str, object] = {
        "environment": {
            "digest": manifest.environment_digest,
            "facts": environment,
            "toolchain_digest": toolchain_digest,
        },
        "evidence": evidence,
        "identity": "",
        "objective": {
            "holding_cost": result.cost.holding.value,
            "shortage_cost": result.cost.shortage.value,
            "total_cost": result.cost.total.value,
        },
        "record_kind": TRACKING_KIND,
        "result_artifact": {
            "digest": result_artifact_digest,
            "id": result_artifact_id,
            "name": result_artifact_name,
        },
        "result_bundle": {
            "artifact_kind": RESULT_KIND,
            "files": result_files,
            "inner_bundle_digest": manifest.inner_bundle_digest,
            "manifest_sha256": result.manifest_sha256,
            "provenance_digest": manifest.provenance_digest,
        },
        "schema": TRACKING_SCHEMA,
        "subject": {"candidate_sha": candidate_sha, "repository": REPOSITORY},
        "workflow": {
            "definition_ref": _text(definition_ref, name="workflow.definition_ref"),
            "definition_sha": definition_sha,
            "run_id": run_id,
            "run_url": run_url,
        },
    }
    identity_preimage = {
        "artifact_kind": RESULT_KIND,
        "candidate_sha": candidate_sha,
        "config_digest": manifest.config_digest,
        "environment_digest": manifest.environment_digest,
        "input_inventory_digest": manifest.input_inventory_digest,
    }
    payload["identity"] = hashlib.sha256(
        canonical_json_bytes(identity_preimage, path="tracking identity")
    ).hexdigest()
    try:
        return VN2TrackingRecord(payload)
    except (TrackingError, ValueError, TypeError) as error:
        if isinstance(error, TrackingError):
            raise
        raise TrackingError("validated evidence did not form a tracking record") from error


build_proposed_record = build_tracking_record


def _validate_payload(value: dict[str, object]) -> dict[str, object]:
    _exact_keys(value, _TOP_KEYS, "tracking record")
    if value.get("schema") != TRACKING_SCHEMA:
        raise TrackingError("tracking schema must equal 1")
    if value.get("record_kind") != TRACKING_KIND:
        raise TrackingError(f"record_kind must equal {TRACKING_KIND!r}")
    subject = _object(value["subject"], "subject")
    _exact_keys(subject, _SUBJECT_KEYS, "subject")
    _commit_sha(subject["candidate_sha"], name="subject.candidate_sha")
    if subject["repository"] != REPOSITORY:
        raise TrackingError(f"subject.repository must equal {REPOSITORY!r}")
    workflow = _object(value["workflow"], "workflow")
    _exact_keys(workflow, _WORKFLOW_KEYS, "workflow")
    definition_ref = _text(workflow["definition_ref"], name="workflow.definition_ref")
    if "@" not in definition_ref or "/.github/workflows/" not in definition_ref:
        raise TrackingError("workflow.definition_ref must identify a GitHub workflow ref")
    _commit_sha(workflow["definition_sha"], name="workflow.definition_sha")
    run_id = _run_id(workflow["run_id"], name="workflow.run_id")
    _run_url(workflow["run_url"], run_id=run_id)
    artifact = _object(value["result_artifact"], "result_artifact")
    _exact_keys(artifact, _ARTIFACT_KEYS, "result_artifact")
    _run_id(artifact["id"], name="result_artifact.id")
    artifact_name = _text(artifact["name"], name="result_artifact.name")
    if artifact_name != f"vn2-acceptance-{subject['candidate_sha']}":
        raise TrackingError("result_artifact.name must bind the subject candidate SHA")
    _digest(artifact["digest"], name="result_artifact.digest")
    bundle = _object(value["result_bundle"], "result_bundle")
    _exact_keys(bundle, _BUNDLE_KEYS, "result_bundle")
    if bundle["artifact_kind"] != RESULT_KIND:
        raise TrackingError(f"result_bundle.artifact_kind must equal {RESULT_KIND!r}")
    _digest(bundle["manifest_sha256"], name="result_bundle.manifest_sha256")
    _digest(bundle["inner_bundle_digest"], name="result_bundle.inner_bundle_digest")
    _digest(bundle["provenance_digest"], name="result_bundle.provenance_digest")
    files = _object(bundle["files"], "result_bundle.files")
    if set(files) != _FILE_NAMES:
        raise TrackingError(
            "result_bundle.files must contain exactly the R1-R4 and environment files"
        )
    for path, digest in files.items():
        _payload_path(path, name=f"result_bundle.files[{path!r}]")
        _digest(digest, name=f"result_bundle.files[{path!r}]")
    evidence = _object(value["evidence"], "evidence")
    _exact_keys(evidence, _EVIDENCE_KEYS, "evidence")
    expected_paths = {
        "config": CONFIG_PATH,
        "input_inventory": INPUT_INVENTORY_PATH,
        "lockfile": LOCK_PATH,
    }
    for name in ("config", "input_inventory", "lockfile"):
        item = _object(evidence[name], f"evidence.{name}")
        _exact_keys(item, _PATH_DIGEST_KEYS, f"evidence.{name}")
        path = _payload_path(item["path"], name=f"evidence.{name}.path")
        if path != expected_paths[name]:
            raise TrackingError(f"evidence.{name}.path must equal {expected_paths[name]!r}")
        _digest(item["digest"], name=f"evidence.{name}.digest")
    semantics = _text(evidence["actuals_semantics"], name="evidence.actuals_semantics")
    if semantics != "censored_sales_surrogate":
        raise TrackingError("actuals_semantics must equal censored_sales_surrogate")
    session = _object(evidence["session"], "evidence.session")
    _exact_keys(session, _SESSION_KEYS, "evidence.session")
    _digest(session["id"], name="evidence.session.id")
    if (
        isinstance(session["series_count"], bool)
        or not isinstance(session["series_count"], int)
        or session["series_count"] < 1
    ):
        raise TrackingError("evidence.session.series_count must be a positive integer")
    _digest(session["series_identity_digest"], name="evidence.session.series_identity_digest")
    _validate_capture_payload(evidence["promoted_capture"])
    environment = _object(value["environment"], "environment")
    _exact_keys(environment, _ENVIRONMENT_KEYS, "environment")
    environment_digest = _digest(environment["digest"], name="environment.digest")
    toolchain_digest = _digest(environment["toolchain_digest"], name="environment.toolchain_digest")
    facts = cast(Mapping[str, object], environment["facts"])
    expected_environment_digest = hashlib.sha256(
        canonical_json_bytes(dict(facts), path="environment.facts")
    ).hexdigest()
    if environment_digest != expected_environment_digest:
        raise TrackingError("environment.digest does not match environment facts")
    if toolchain_digest != _toolchain_digest(facts):
        raise TrackingError("environment.toolchain_digest does not match environment facts")
    _validate_environment(facts)
    objective = _object(value["objective"], "objective")
    _exact_keys(objective, _OBJECTIVE_KEYS, "objective")
    holding = _finite_cost(objective["holding_cost"], name="objective.holding_cost")
    shortage = _finite_cost(objective["shortage_cost"], name="objective.shortage_cost")
    total = _finite_cost(objective["total_cost"], name="objective.total_cost")
    if total != holding + shortage:
        raise TrackingError("objective.total_cost must equal holding_cost + shortage_cost")
    identity = _digest(value["identity"], name="identity")
    expected_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "artifact_kind": RESULT_KIND,
                "candidate_sha": subject["candidate_sha"],
                "config_digest": cast(Mapping[str, object], evidence["config"])["digest"],
                "environment_digest": environment["digest"],
                "input_inventory_digest": cast(Mapping[str, object], evidence["input_inventory"])[
                    "digest"
                ],
            },
            path="tracking identity",
        )
    ).hexdigest()
    if identity != expected_identity:
        raise TrackingError("identity does not match the R6 identity preimage")
    return value


def _validate_capture_payload(value: object) -> None:
    capture = _object(value, "evidence.promoted_capture")
    _exact_keys(capture, _CAPTURE_KEYS, "evidence.promoted_capture")
    _run_id(capture["artifact_id"], name="evidence.promoted_capture.artifact_id")
    for name in (
        "artifact_digest",
        "capture_digest",
        "manifest_sha256",
        "inner_bundle_digest",
        "environment_digest",
    ):
        _digest(capture[name], name=f"evidence.promoted_capture.{name}")
    _text(capture["artifact_name"], name="evidence.promoted_capture.artifact_name")
    _commit_sha(capture["producer_sha"], name="evidence.promoted_capture.producer_sha")
    _commit_sha(capture["workflow_sha"], name="evidence.promoted_capture.workflow_sha")
    workflow_run_id = _run_id(
        capture["workflow_run_id"], name="evidence.promoted_capture.workflow_run_id"
    )
    _run_url(capture["run_url"], run_id=workflow_run_id)


def _validate_environment(value: object) -> None:
    facts = _object(value, "environment.facts")
    expected = {
        "arch",
        "cpu_model",
        "os",
        "python",
        "numpy",
        "numpy_config",
        "runner_image",
        "thread_policy",
    }
    _exact_keys(facts, frozenset(expected), "environment.facts")
    if facts["arch"] != "x86_64":
        raise TrackingError("environment.facts.arch must equal x86_64")
    os_value = _object(facts["os"], "environment.facts.os")
    _exact_keys(os_value, frozenset({"id", "version_id", "pretty_name"}), "environment.facts.os")
    _text(os_value["id"], name="environment.facts.os.id")
    _text(os_value["version_id"], name="environment.facts.os.version_id")
    _text(os_value["pretty_name"], name="environment.facts.os.pretty_name")
    for name in ("cpu_model", "python", "numpy", "numpy_config", "runner_image"):
        _text(facts[name], name=f"environment.facts.{name}")
    policy = _object(facts["thread_policy"], "environment.facts.thread_policy")
    if not policy or any(not isinstance(key, str) for key in policy):
        raise TrackingError("environment.facts.thread_policy must be a non-empty object")
    for key, item in policy.items():
        _text(key, name="environment.facts.thread_policy key")
        _text(item, name=f"environment.facts.thread_policy.{key}")


def _environment_payload(environment: VN2EvidenceEnvironment) -> dict[str, object]:
    return {
        "arch": environment.arch,
        "cpu_model": environment.cpu_model,
        "numpy": environment.numpy,
        "numpy_config": environment.numpy_config,
        "os": {
            "id": environment.os_id,
            "pretty_name": environment.os_pretty_name,
            "version_id": environment.os_version_id,
        },
        "python": environment.python,
        "runner_image": environment.runner_image,
        "thread_policy": dict(environment.thread_policy),
    }


def _capture_payload(bundle: CaptureBundle, receipt: CaptureReceipt) -> dict[str, object]:
    manifest = bundle.manifest
    if receipt.artifact_name != manifest.artifact_name:
        raise TrackingError("committed capture receipt name does not match its bundle")
    return {
        "artifact_digest": receipt.artifact_digest,
        "artifact_id": receipt.artifact_id,
        "artifact_name": receipt.artifact_name,
        "capture_digest": manifest.capture_digest,
        "environment_digest": manifest.environment_digest,
        "inner_bundle_digest": manifest.inner_bundle_digest,
        "manifest_sha256": bundle.manifest_sha256,
        "producer_sha": receipt.producer_sha,
        "run_url": receipt.run_url,
        "workflow_run_id": receipt.workflow_run_id,
        "workflow_sha": receipt.workflow_sha,
    }


def _toolchain_digest(environment: Mapping[str, object]) -> str:
    preimage = {
        "numpy": environment["numpy"],
        "numpy_config": environment["numpy_config"],
        "python": environment["python"],
        "schema": 1,
    }
    return hashlib.sha256(canonical_json_bytes(preimage, path="tracking toolchain")).hexdigest()


def _comparability_key(payload: Mapping[str, object]) -> dict[str, object]:
    evidence = cast(Mapping[str, object], payload["evidence"])
    environment = cast(Mapping[str, object], payload["environment"])
    facts = cast(Mapping[str, object], environment["facts"])
    os_value = cast(Mapping[str, object], facts["os"])
    capture = cast(Mapping[str, object], evidence["promoted_capture"])
    return {
        "architecture": facts["arch"],
        "os_id": os_value["id"],
        "os_version": os_value["version_id"],
        "lockfile_digest": cast(Mapping[str, object], evidence["lockfile"])["digest"],
        "config_digest": cast(Mapping[str, object], evidence["config"])["digest"],
        "input_inventory_digest": cast(Mapping[str, object], evidence["input_inventory"])["digest"],
        "promoted_capture_digest": capture["capture_digest"],
        "actuals_semantics": evidence["actuals_semantics"],
    }


def _canonical_record_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return canonical_json_bytes(dict(value), path="tracking record") + b"\n"
    except ValueError as error:
        raise TrackingError("tracking record contains non-canonical JSON values") from error


def _read_bytes(value: bytes | bytearray | str | Path, *, name: str) -> bytes:
    if isinstance(value, Path):
        try:
            return value.read_bytes()
        except OSError as error:
            raise TrackingError(f"{name} path is unreadable") from error
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TrackingError(f"{name} must be bytes, text, or a path")


def _parse_json(value: bytes, *, name: str) -> dict[str, object]:
    try:
        text = value.decode("utf-8")
        parsed = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TrackingError(f"{name} must be valid UTF-8 JSON") from error
    return _object(parsed, name)


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TrackingError(f"tracking JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrackingError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise TrackingError(
            f"{name} fields mismatch: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrackingError(f"{name} must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise TrackingError(f"{name} must be valid UTF-8") from error
    return value


def _commit_sha(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _COMMIT_SHA.fullmatch(text) is None:
        raise TrackingError(f"{name} must be a lowercase 40-hex SHA")
    return text


def _digest(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _CAPTURE_SHA256.fullmatch(text) is None:
        raise TrackingError(f"{name} must be a lowercase 64-hex SHA-256")
    return text


def _normalized_digest(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    return _digest(text, name=name)


def _run_id(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _RUN_ID.fullmatch(text) is None:
        raise TrackingError(f"{name} must be a positive decimal identifier")
    return text


def _run_url(value: object, *, run_id: str) -> str:
    text = _text(value, name="workflow.run_url")
    parsed = urlparse(text)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or parsed.path != f"/{REPOSITORY}/actions/runs/{run_id}"
    ):
        raise TrackingError("workflow.run_url must equal the repository Actions URL")
    return text


def _payload_path(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != text
        or "\\" in text
    ):
        raise TrackingError(f"{name} must be a canonical relative POSIX path")
    return text


def _finite_cost(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise TrackingError(f"{name} must be finite and non-negative")
    return number


def _validate_output_path(path: Path) -> Path:
    if path.name == "series.jsonl" or ("stage3" in path.parts and "tracking" in path.parts):
        raise TrackingError("tracking writers never accept the tracked history path")
    if not path.name or path.name in {".", ".."}:
        raise TrackingError("tracking proposal destination must be a file path")
    parent = path.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise TrackingError("tracking proposal output root must be a real directory")
    current = parent
    while current != current.parent:
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise TrackingError("tracking proposal output ancestors must be inspectable") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise TrackingError("tracking proposal output ancestors must be real directories")
        current = current.parent
    return path


__all__ = [
    "AppendDecision",
    "TRACKING_KIND",
    "TRACKING_SCHEMA",
    "TrackingComparison",
    "TrackingError",
    "TrackingRecord",
    "VN2TrackingRecord",
    "append_decision",
    "build_proposed_record",
    "build_tracking_record",
    "compare_records",
    "compare_tracking_records",
    "decide_append",
    "parse_tracking_history",
    "parse_tracking_record",
    "write_proposal_record",
    "write_tracking_record",
]
