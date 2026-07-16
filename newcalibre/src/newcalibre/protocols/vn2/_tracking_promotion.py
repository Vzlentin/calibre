"""Validate exact VN2 tracking promotion receipts and append-only bytes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.protocols.vn2._tracking_contracts import (
    REPOSITORY,
    TrackingError,
    VN2TrackingRecord,
    _commit_sha,
    _digest,
    _exact_keys,
    _normalized_digest,
    _object,
    _parse_json,
    _read_bytes,
    _regular_file_sha256,
    _run_id,
    _run_url,
    _text,
    _validate_definition_ref,
)
from newcalibre.protocols.vn2._tracking_persistence import _write_artifact_bytes
from newcalibre.protocols.vn2._tracking_validation import (
    decide_append,
    parse_tracking_history,
    parse_tracking_record,
)

PROMOTION_RECEIPT_KIND = "vn2-tracking-promotion-receipt"
TRACKING_SERIES_PATH = "stage3/evidence/tracking/series.jsonl"
_RESULT_WORKFLOW_PATH = ".github/workflows/newcalibre.yml"
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "receipt_kind",
        "repository",
        "candidate_sha",
        "workflow",
        "result_artifact",
        "proposal_artifact",
        "record_sha256",
    }
)
_WORKFLOW_KEYS = frozenset({"definition_ref", "definition_sha", "run_id", "run_url"})
_ARTIFACT_KEYS = frozenset({"id", "name", "digest"})


@dataclass(frozen=True, slots=True)
class _ArtifactBinding:
    """Bind one live GitHub artifact to its immutable archive digest."""

    id: str
    name: str
    digest: str

    def to_value(self) -> dict[str, str]:
        """Return the canonical receipt value."""
        return {"digest": self.digest, "id": self.id, "name": self.name}


@dataclass(frozen=True, slots=True)
class PromotionReceipt:
    """Represent one strict canonical tracking-promotion receipt."""

    candidate_sha: str
    definition_ref: str
    definition_sha: str
    run_id: str
    run_url: str
    result_artifact: _ArtifactBinding
    proposal_artifact: _ArtifactBinding
    record_sha256: str

    def to_bytes(self) -> bytes:
        """Serialize the receipt as one canonical LF-terminated JSON object."""
        value = {
            "candidate_sha": self.candidate_sha,
            "proposal_artifact": self.proposal_artifact.to_value(),
            "receipt_kind": PROMOTION_RECEIPT_KIND,
            "record_sha256": self.record_sha256,
            "repository": REPOSITORY,
            "result_artifact": self.result_artifact.to_value(),
            "schema": 1,
            "workflow": {
                "definition_ref": self.definition_ref,
                "definition_sha": self.definition_sha,
                "run_id": self.run_id,
                "run_url": self.run_url,
            },
        }
        try:
            return canonical_json_bytes(value, path="tracking promotion receipt") + b"\n"
        except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as error:
            raise TrackingError("promotion receipt contains non-canonical JSON values") from error


def promotion_receipt_path(candidate_sha: str) -> str:
    """Return the sole canonical receipt path for one tracking candidate."""
    candidate = _commit_sha(candidate_sha, name="candidate_sha")
    return f"stage3/evidence/tracking/{candidate}-receipt.json"


def validate_promotion_paths(paths: Sequence[str], *, candidate_sha: str) -> None:
    """Require exactly one canonical series path and its matching receipt."""
    normalized: list[str] = []
    for path in paths:
        text = _text(path, name="changed tracking path")
        if (
            "\\" in text
            or text.startswith("/")
            or text.endswith("/")
            or PurePosixPath(text).as_posix() != text
            or any(part in {"", ".", ".."} for part in text.split("/"))
        ):
            raise TrackingError(
                "changed tracking paths must be canonical repository-relative paths"
            )
        normalized.append(text)
    expected = {TRACKING_SERIES_PATH, promotion_receipt_path(candidate_sha)}
    if len(normalized) != 2 or set(normalized) != expected:
        raise TrackingError(
            "tracking promotion must change exactly the series and matching receipt"
        )


def load_promotion_metadata(path: Path, *, name: str) -> dict[str, object]:
    """Load one bounded regular GitHub metadata response."""
    return _parse_json(_read_bytes(Path(path), name=name), name=name)


def parse_promotion_receipt(value: bytes | bytearray | str | Path) -> PromotionReceipt:
    """Parse exactly one canonical tracking-promotion receipt."""
    payload = _read_bytes(value, name="tracking promotion receipt")
    if not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
        raise TrackingError("promotion receipt must end with exactly one LF")
    body = payload[:-1]
    if not body or b"\n" in body or b"\r" in body:
        raise TrackingError("promotion receipt must contain exactly one JSON object line")
    parsed = _parse_json(body, name="tracking promotion receipt")
    receipt = _parse_receipt_value(parsed)
    if receipt.to_bytes() != payload:
        raise TrackingError("promotion receipt bytes are not canonical")
    return receipt


def build_promotion_receipt(
    record: VN2TrackingRecord,
    proposal: bytes | bytearray | str | Path,
    *,
    result_artifact_metadata: object,
    proposal_artifact_metadata: object,
    run_metadata: object,
    result_archive: Path,
    proposal_archive: Path,
) -> PromotionReceipt:
    """Build a receipt only after live metadata and both archive digests validate."""
    if not isinstance(record, VN2TrackingRecord) or not record._publication_eligible:
        raise TrackingError("promotion requires a record freshly rebuilt from validated evidence")
    proposal_bytes = _read_bytes(proposal, name="downloaded tracking proposal")
    parsed = parse_tracking_record(proposal_bytes)
    if parsed.to_bytes() != record.to_bytes():
        raise TrackingError("downloaded proposal bytes do not match the rebuilt tracking record")
    result, proposal_binding = _validate_live_artifacts(
        record,
        result_artifact_metadata=result_artifact_metadata,
        proposal_artifact_metadata=proposal_artifact_metadata,
        run_metadata=run_metadata,
    )
    if _regular_file_sha256(Path(result_archive), name="result artifact archive") != result.digest:
        raise TrackingError("downloaded result archive digest does not match GitHub metadata")
    if (
        _regular_file_sha256(Path(proposal_archive), name="proposal artifact archive")
        != proposal_binding.digest
    ):
        raise TrackingError("downloaded proposal archive digest does not match GitHub metadata")
    subject = cast(Mapping[str, object], record.payload["subject"])
    workflow = cast(Mapping[str, object], record.payload["workflow"])
    return PromotionReceipt(
        candidate_sha=cast(str, subject["candidate_sha"]),
        definition_ref=cast(str, workflow["definition_ref"]),
        definition_sha=cast(str, workflow["definition_sha"]),
        run_id=cast(str, workflow["run_id"]),
        run_url=cast(str, workflow["run_url"]),
        result_artifact=result,
        proposal_artifact=proposal_binding,
        record_sha256=hashlib.sha256(proposal_bytes).hexdigest(),
    )


def write_promotion_receipt(receipt: PromotionReceipt, path: Path) -> bool:
    """Publish a canonical receipt only beneath the successor artifacts root."""
    if not isinstance(receipt, PromotionReceipt):
        raise TrackingError("receipt writer requires a PromotionReceipt")
    return _write_artifact_bytes(receipt.to_bytes(), Path(path))


def validate_tracking_promotion(
    record: VN2TrackingRecord,
    proposal: bytes | bytearray | str | Path,
    receipt: bytes | bytearray | str | Path,
    promoted_history: bytes | bytearray | str | Path,
    *,
    prior_history: bytes | bytearray | str | Path | None,
    result_artifact_metadata: object,
    proposal_artifact_metadata: object,
    run_metadata: object,
    result_archive: Path,
    proposal_archive: Path,
    base_sha: str,
    default_branch_sha: str,
) -> PromotionReceipt:
    """Validate live provenance, receipt wire, and exact append-only history bytes."""
    subject = cast(Mapping[str, object], record.payload["subject"])
    candidate = cast(str, subject["candidate_sha"])
    if _commit_sha(base_sha, name="promotion base SHA") != candidate:
        raise TrackingError("promotion PR base does not equal the record candidate SHA")
    if _commit_sha(default_branch_sha, name="default branch SHA") != candidate:
        raise TrackingError("live default-branch tip does not equal the record candidate SHA")
    expected_receipt = build_promotion_receipt(
        record,
        proposal,
        result_artifact_metadata=result_artifact_metadata,
        proposal_artifact_metadata=proposal_artifact_metadata,
        run_metadata=run_metadata,
        result_archive=result_archive,
        proposal_archive=proposal_archive,
    )
    actual_receipt = parse_promotion_receipt(receipt)
    if actual_receipt.to_bytes() != expected_receipt.to_bytes():
        raise TrackingError("promotion receipt does not bind the validated live evidence")

    proposal_bytes = record.to_bytes()
    if prior_history is None:
        prior_bytes = b""
        prior_records: tuple[VN2TrackingRecord, ...] = ()
    else:
        prior_bytes = _read_bytes(prior_history, name="prior tracking history")
        prior_records = parse_tracking_history(prior_bytes)
    decision = decide_append(record, prior_records)
    if decision.action == "noop":
        raise TrackingError("tracking promotion exact replay is a refused no-op")
    if decision.action == "conflict":
        raise TrackingError("tracking promotion conflicts with an existing identity")

    promoted_bytes = _read_bytes(promoted_history, name="promoted tracking history")
    is_exact_append = (
        len(promoted_bytes) == len(prior_bytes) + len(proposal_bytes)
        and promoted_bytes.startswith(prior_bytes)
        and promoted_bytes.endswith(proposal_bytes)
    )
    if not is_exact_append:
        parse_tracking_history(promoted_bytes)
        raise TrackingError("promoted history must equal the exact old prefix plus proposal bytes")
    return actual_receipt


def _parse_receipt_value(value: dict[str, object]) -> PromotionReceipt:
    _exact_keys(value, _RECEIPT_KEYS, "tracking promotion receipt")
    schema = value.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise TrackingError("promotion receipt schema must be the integer 1")
    if value.get("receipt_kind") != PROMOTION_RECEIPT_KIND:
        raise TrackingError(f"receipt_kind must equal {PROMOTION_RECEIPT_KIND!r}")
    if value.get("repository") != REPOSITORY:
        raise TrackingError(f"promotion receipt repository must equal {REPOSITORY!r}")
    candidate = _commit_sha(value.get("candidate_sha"), name="receipt.candidate_sha")
    workflow = _object(value.get("workflow"), "receipt.workflow")
    _exact_keys(workflow, _WORKFLOW_KEYS, "receipt.workflow")
    definition_ref = _validate_definition_ref(workflow["definition_ref"])
    definition_sha = _commit_sha(workflow["definition_sha"], name="receipt.workflow.definition_sha")
    run_id = _run_id(workflow["run_id"], name="receipt.workflow.run_id")
    run_url = _run_url(workflow["run_url"], run_id=run_id)
    result = _parse_receipt_artifact(value.get("result_artifact"), name="receipt.result_artifact")
    proposal = _parse_receipt_artifact(
        value.get("proposal_artifact"), name="receipt.proposal_artifact"
    )
    if result.name != f"vn2-acceptance-{candidate}":
        raise TrackingError("receipt result artifact name must bind the candidate SHA")
    if proposal.name != f"vn2-tracking-proposal-{candidate}":
        raise TrackingError("receipt proposal artifact name must bind the candidate SHA")
    return PromotionReceipt(
        candidate_sha=candidate,
        definition_ref=definition_ref,
        definition_sha=definition_sha,
        run_id=run_id,
        run_url=run_url,
        result_artifact=result,
        proposal_artifact=proposal,
        record_sha256=_digest(value.get("record_sha256"), name="receipt.record_sha256"),
    )


def _parse_receipt_artifact(value: object, *, name: str) -> _ArtifactBinding:
    artifact = _object(value, name)
    _exact_keys(artifact, _ARTIFACT_KEYS, name)
    return _ArtifactBinding(
        id=_run_id(artifact["id"], name=f"{name}.id"),
        name=_text(artifact["name"], name=f"{name}.name"),
        digest=_normalized_digest(artifact["digest"], name=f"{name}.digest"),
    )


def _validate_live_artifacts(
    record: VN2TrackingRecord,
    *,
    result_artifact_metadata: object,
    proposal_artifact_metadata: object,
    run_metadata: object,
) -> tuple[_ArtifactBinding, _ArtifactBinding]:
    subject = cast(Mapping[str, object], record.payload["subject"])
    workflow = cast(Mapping[str, object], record.payload["workflow"])
    result_record = cast(Mapping[str, object], record.payload["result_artifact"])
    candidate = cast(str, subject["candidate_sha"])
    run_id = cast(str, workflow["run_id"])
    repository_id = _validate_live_run(record, run_metadata)
    result = _validate_artifact_metadata(
        result_artifact_metadata,
        name="result artifact metadata",
        expected_name=f"vn2-acceptance-{candidate}",
        expected_candidate=candidate,
        expected_run_id=run_id,
        expected_repository_id=repository_id,
    )
    proposal = _validate_artifact_metadata(
        proposal_artifact_metadata,
        name="proposal artifact metadata",
        expected_name=f"vn2-tracking-proposal-{candidate}",
        expected_candidate=candidate,
        expected_run_id=run_id,
        expected_repository_id=repository_id,
    )
    expected_result = _ArtifactBinding(
        id=cast(str, result_record["id"]),
        name=cast(str, result_record["name"]),
        digest=cast(str, result_record["digest"]),
    )
    if result != expected_result:
        raise TrackingError("live result artifact metadata does not match the rebuilt record")
    if proposal.id == result.id:
        raise TrackingError("result and proposal artifacts must be distinct")
    return result, proposal


def _validate_live_run(record: VN2TrackingRecord, value: object) -> int:
    run = _object(value, "workflow-run metadata")
    subject = cast(Mapping[str, object], record.payload["subject"])
    workflow = cast(Mapping[str, object], record.payload["workflow"])
    candidate = cast(str, subject["candidate_sha"])
    run_id = cast(str, workflow["run_id"])
    actual_run_id = _positive_int(run.get("id"), name="workflow-run metadata id")
    if str(actual_run_id) != run_id:
        raise TrackingError("workflow-run metadata ID does not match the rebuilt record")
    expectations = {
        "event": "workflow_dispatch",
        "path": _RESULT_WORKFLOW_PATH,
        "head_branch": "main",
        "head_sha": candidate,
        "status": "completed",
        "conclusion": "success",
    }
    for name, expected in expectations.items():
        if run.get(name) != expected:
            raise TrackingError(f"workflow-run metadata {name} must equal {expected!r}")
    if _positive_int(run.get("run_attempt"), name="workflow-run metadata run_attempt") != 1:
        raise TrackingError("workflow-run metadata run_attempt must equal 1")
    repository = _object(run.get("repository"), "workflow-run metadata repository")
    if repository.get("full_name") != REPOSITORY:
        raise TrackingError(f"workflow-run metadata repository must equal {REPOSITORY!r}")
    repository_id = _positive_int(repository.get("id"), name="workflow-run repository id")
    actual_url = _run_url(run.get("html_url"), run_id=run_id)
    if actual_url != workflow["run_url"]:
        raise TrackingError("workflow-run URL does not match the rebuilt record")
    expected_ref = f"{REPOSITORY}/{_RESULT_WORKFLOW_PATH}@refs/heads/main"
    if workflow["definition_ref"] != expected_ref:
        raise TrackingError("tracking workflow definition ref must bind newcalibre.yml on main")
    if workflow["definition_sha"] != candidate:
        raise TrackingError("tracking workflow definition SHA must equal the candidate SHA")
    return repository_id


def _validate_artifact_metadata(
    value: object,
    *,
    name: str,
    expected_name: str,
    expected_candidate: str,
    expected_run_id: str,
    expected_repository_id: int,
) -> _ArtifactBinding:
    metadata = _object(value, name)
    artifact_id = str(_positive_int(metadata.get("id"), name=f"{name} id"))
    artifact_name = _text(metadata.get("name"), name=f"{name} name")
    if artifact_name != expected_name:
        raise TrackingError(f"{name} name must equal {expected_name!r}")
    digest = _normalized_digest(metadata.get("digest"), name=f"{name} digest")
    if metadata.get("expired") is not False:
        raise TrackingError(f"{name} must exist and be unexpired")
    workflow_run = _object(metadata.get("workflow_run"), f"{name} workflow_run")
    if (
        str(_positive_int(workflow_run.get("id"), name=f"{name} workflow run id"))
        != expected_run_id
    ):
        raise TrackingError(f"{name} workflow run ID does not match")
    if workflow_run.get("head_sha") != expected_candidate:
        raise TrackingError(f"{name} workflow head SHA does not match")
    if workflow_run.get("head_branch") != "main":
        raise TrackingError(f"{name} workflow head branch must equal 'main'")
    for key in ("repository_id", "head_repository_id"):
        if (
            _positive_int(workflow_run.get(key), name=f"{name} workflow {key}")
            != expected_repository_id
        ):
            raise TrackingError(f"{name} workflow {key} does not match the repository")
    return _ArtifactBinding(id=artifact_id, name=artifact_name, digest=digest)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrackingError(f"{name} must be a positive integer")
    return value
