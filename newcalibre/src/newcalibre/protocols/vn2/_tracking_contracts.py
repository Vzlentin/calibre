"""Private VN2 tracking contracts and strict value validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlparse

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.domain.actuals import ActualsSemantics
from newcalibre.protocols.vn2._artifact_contracts import (
    CONFIG_PATH,
    GITHUB_REPOSITORY,
    INPUT_INVENTORY_PATH,
    LOCK_PATH,
    RESULT_KIND,
    THREAD_VARIABLES,
)

TRACKING_SCHEMA = 1
TRACKING_KIND = "vn2-gate-a-tracking-record"
REPOSITORY = GITHUB_REPOSITORY
_CAPTURES_ROOT = "stage3/evidence/captures"
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
        "r1-orders.jsonl",
        "r2-cost-ledger.jsonl",
        "r3-final-triple.json",
        "r4-cost-trajectory.json",
    }
)
_MAX_TRACKING_BYTES = 4 * 1024 * 1024
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


def _owned_value(
    value: object,
    *,
    name: str,
    _active: set[int] | None = None,
    _depth: int = 0,
) -> object:
    if _depth > 64:
        raise TrackingError(f"{name} exceeds the maximum JSON nesting depth")
    active = set() if _active is None else _active
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise TrackingError(f"{name} contains a cyclic JSON value")
        active.add(identity)
        try:
            owned: dict[object, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TrackingError(f"{name} object keys must be strings")
                owned[key] = _owned_value(
                    item,
                    name=f"{name}.{key}",
                    _active=active,
                    _depth=_depth + 1,
                )
            return owned
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise TrackingError(f"{name} contains a cyclic JSON value")
        active.add(identity)
        try:
            return [
                _owned_value(
                    item,
                    name=f"{name}[]",
                    _active=active,
                    _depth=_depth + 1,
                )
                for item in value
            ]
        finally:
            active.remove(identity)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TrackingError(f"{name} contains an unsupported JSON value")


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _frozen_payload(value: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze_value(dict(value)))


def _thawed_payload(value: Mapping[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _thaw_value(value))


@dataclass(frozen=True, slots=True, init=False)
class VN2TrackingRecord:
    """Represent one validated canonical v1 tracking record."""

    payload: Mapping[str, object]
    _publication_eligible: bool = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TrackingError(
            "VN2TrackingRecord construction is private; use validated parsing or "
            "evidence projection"
        )

    @classmethod
    def _from_parsed(cls, payload: Mapping[str, object]) -> VN2TrackingRecord:
        """Construct one structurally valid record parsed from canonical bytes."""
        return cls._construct(payload, publication_eligible=False)

    @classmethod
    def _from_evidence(cls, payload: Mapping[str, object]) -> VN2TrackingRecord:
        """Construct one publication-eligible record from validated evidence."""
        return cls._construct(payload, publication_eligible=True)

    @classmethod
    def _construct(
        cls,
        payload: Mapping[str, object],
        *,
        publication_eligible: bool,
    ) -> VN2TrackingRecord:
        self = object.__new__(cls)
        owned = _owned_value(payload, name="tracking record")
        if not isinstance(owned, dict):
            raise TrackingError("tracking record must be a JSON object")
        value = _validate_payload(cast(dict[str, object], owned))
        _canonical_record_bytes(value)
        object.__setattr__(self, "payload", _frozen_payload(value))
        object.__setattr__(self, "_publication_eligible", publication_eligible)
        return self

    @property
    def identity(self) -> str:
        """Return the immutable record identity."""
        return cast(str, self.payload["identity"])

    @property
    def total_cost(self) -> float:
        """Return the validated total realized cost."""
        return _finite_cost(
            cast(Mapping[str, object], self.payload["objective"])["total_cost"],
            name="objective.total_cost",
        )

    def to_bytes(self) -> bytes:
        """Serialize the record as canonical JSONL bytes."""
        return _canonical_record_bytes(_thawed_payload(self.payload))

    def to_json(self) -> str:
        """Serialize the record as canonical JSONL text."""
        return self.to_bytes().decode("utf-8")

    def __getitem__(self, key: str) -> object:
        """Return one immutable payload field."""
        return self.payload[key]


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
        """Return whether publication should append this record."""
        return self.action == "append"


def _validate_payload(value: dict[str, object]) -> dict[str, object]:
    _exact_keys(value, _TOP_KEYS, "tracking record")
    schema = value.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != TRACKING_SCHEMA:
        raise TrackingError("tracking schema must be the integer 1")
    if value.get("record_kind") != TRACKING_KIND:
        raise TrackingError(f"record_kind must equal {TRACKING_KIND!r}")
    subject = _object(value["subject"], "subject")
    _exact_keys(subject, _SUBJECT_KEYS, "subject")
    _commit_sha(subject["candidate_sha"], name="subject.candidate_sha")
    if subject["repository"] != REPOSITORY:
        raise TrackingError(f"subject.repository must equal {REPOSITORY!r}")
    workflow = _object(value["workflow"], "workflow")
    _exact_keys(workflow, _WORKFLOW_KEYS, "workflow")
    _validate_definition_ref(workflow["definition_ref"])
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
        raise TrackingError("result_bundle.files must contain exactly the four R1-R4 paths")
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
    if semantics not in {item.value for item in ActualsSemantics}:
        raise TrackingError("actuals_semantics must be a supported ActualsSemantics value")
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
    facts = _object(environment["facts"], "environment.facts")
    _validate_environment(facts)
    environment_digest = _digest(environment["digest"], name="environment.digest")
    toolchain_digest = _digest(environment["toolchain_digest"], name="environment.toolchain_digest")
    try:
        expected_environment_digest = hashlib.sha256(
            canonical_json_bytes(dict(facts), path="environment.facts")
        ).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as error:
        raise TrackingError("environment.facts contains non-canonical JSON values") from error
    if environment_digest != expected_environment_digest:
        raise TrackingError("environment.digest does not match environment facts")
    if toolchain_digest != _toolchain_digest(facts):
        raise TrackingError("environment.toolchain_digest does not match environment facts")
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
    producer_sha = _commit_sha(
        capture["producer_sha"],
        name="evidence.promoted_capture.producer_sha",
    )
    artifact_name = _text(capture["artifact_name"], name="evidence.promoted_capture.artifact_name")
    if artifact_name != f"oracle-capture-{producer_sha}":
        raise TrackingError("promoted capture artifact_name must bind the producer SHA")
    _commit_sha(capture["workflow_sha"], name="evidence.promoted_capture.workflow_sha")
    workflow_run_id = _run_id(
        capture["workflow_run_id"], name="evidence.promoted_capture.workflow_run_id"
    )
    _run_url(capture["run_url"], run_id=workflow_run_id)


def _validate_environment(value: object) -> None:
    facts = _object(value, "environment.facts")
    expected = frozenset(
        {
            "arch",
            "cpu_model",
            "os",
            "python",
            "numpy",
            "numpy_config",
            "runner_image",
            "thread_policy",
        }
    )
    _exact_keys(facts, expected, "environment.facts")
    for name in ("arch", "cpu_model", "python", "numpy", "numpy_config", "runner_image"):
        _text(facts[name], name=f"environment.facts.{name}")
    os_value = _object(facts["os"], "environment.facts.os")
    _exact_keys(os_value, frozenset({"id", "version_id", "pretty_name"}), "environment.facts.os")
    for name in ("id", "version_id", "pretty_name"):
        _text(os_value[name], name=f"environment.facts.os.{name}")
    thread_policy = _object(facts["thread_policy"], "environment.facts.thread_policy")
    _exact_keys(
        thread_policy,
        frozenset(THREAD_VARIABLES),
        "environment.facts.thread_policy",
    )
    for name in THREAD_VARIABLES:
        _text(thread_policy[name], name=f"environment.facts.thread_policy.{name}")


def _validate_definition_ref(value: object) -> str:
    text = _text(value, name="workflow.definition_ref")
    if text.count("@") != 1:
        raise TrackingError("workflow.definition_ref must contain one @ separator")
    repository_path, ref = text.split("@")
    prefix = f"{REPOSITORY}/.github/workflows/"
    if not repository_path.startswith(prefix) or not ref:
        raise TrackingError("workflow.definition_ref must identify a workflow path and ref")
    workflow_path = repository_path[len(prefix) :]
    if (
        not workflow_path
        or workflow_path.startswith("/")
        or workflow_path.endswith("/")
        or "\\" in workflow_path
        or any(part in {"", ".", ".."} for part in workflow_path.split("/"))
        or PurePosixPath(workflow_path).as_posix() != workflow_path
        or PurePosixPath(ref).as_posix() != ref
        or ref.startswith("/")
        or ref.endswith("/")
        or "\\" in ref
        or any(part in {"", ".", ".."} for part in ref.split("/"))
    ):
        raise TrackingError(
            "workflow.definition_ref must use canonical repository workflow path and ref"
        )
    return text


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
    except (
        RecursionError,
        TypeError,
        ValueError,
        OverflowError,
        UnicodeError,
    ) as error:
        raise TrackingError("tracking record contains non-canonical JSON values") from error


def _read_bytes(value: bytes | bytearray | str | Path, *, name: str) -> bytes:
    if isinstance(value, Path):
        try:
            payload = value.read_bytes()
        except OSError as error:
            raise TrackingError(f"{name} path is unreadable") from error
        return _bounded_bytes(payload, name=name)
    if isinstance(value, bytearray):
        return _bounded_bytes(bytes(value), name=name)
    if isinstance(value, bytes):
        return _bounded_bytes(value, name=name)
    if isinstance(value, str):
        try:
            payload = value.encode("utf-8")
        except UnicodeError as error:
            raise TrackingError(f"{name} must be valid UTF-8") from error
        return _bounded_bytes(payload, name=name)
    raise TrackingError(f"{name} must be bytes, text, or a path")


def _bounded_bytes(value: bytes, *, name: str) -> bytes:
    if len(value) > _MAX_TRACKING_BYTES:
        raise TrackingError(f"{name} exceeds the maximum size")
    return value


def _parse_json(value: bytes, *, name: str) -> dict[str, object]:
    try:
        text = value.decode("utf-8")
        parsed = json.loads(text, object_pairs_hook=_unique_object)
    except (
        UnicodeError,
        ValueError,
        OverflowError,
        RecursionError,
        json.JSONDecodeError,
    ) as error:
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
