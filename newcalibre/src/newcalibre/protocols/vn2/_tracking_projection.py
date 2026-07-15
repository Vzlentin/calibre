"""Private evidence projection for VN2 tracking records."""

from __future__ import annotations

import hashlib
from pathlib import Path

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.oracle import (
    CaptureBundle,
    CaptureReceipt,
    OracleEvidenceError,
    validate_committed_promoted_capture,
)
from newcalibre.protocols.vn2._artifact_contracts import (
    CONFIG_PATH,
    INPUT_INVENTORY_PATH,
    LOCK_PATH,
    RESULT_KIND,
    VN2EvidenceEnvironment,
    VN2ResultBundle,
    VN2ResultError,
    _environment_value,
)
from newcalibre.protocols.vn2._tracking_contracts import (
    REPOSITORY,
    TRACKING_KIND,
    TRACKING_SCHEMA,
    TrackingError,
    VN2TrackingRecord,
    _commit_sha,
    _normalized_digest,
    _run_id,
    _run_url,
    _text,
    _toolchain_digest,
)
from newcalibre.protocols.vn2.artifacts import validate_vn2_result_bundle


def _build_tracking_record(
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


def _environment_payload(environment: VN2EvidenceEnvironment) -> dict[str, object]:
    return dict(_environment_value(environment))


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
    try:
        return _build_tracking_record(
            result_root,
            capture_root,
            candidate_sha=candidate_sha,
            definition_ref=definition_ref,
            definition_sha=definition_sha,
            run_id=run_id,
            run_url=run_url,
            result_artifact_id=result_artifact_id,
            result_artifact_name=result_artifact_name,
            result_artifact_digest=result_artifact_digest,
            config_path=config_path,
            input_inventory_path=input_inventory_path,
            lockfile_path=lockfile_path,
        )
    except TrackingError:
        raise
    except (
        OracleEvidenceError,
        VN2ResultError,
        OSError,
        UnicodeError,
        TypeError,
        ValueError,
    ) as error:
        raise TrackingError("tracking evidence validation failed") from error
