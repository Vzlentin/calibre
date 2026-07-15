"""Exercise strict oracle manifest, bundle, and promotion-receipt validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from newcalibre.oracle import (
    ORACLE_COMMIT,
    ORACLE_LOCK_SHA256,
    ORACLE_TAG,
    CaptureBundle,
    OracleEvidenceError,
    validate_capture_bundle,
    validate_capture_receipt,
    validate_committed_promoted_capture,
    validate_promoted_capture,
    validate_promoted_captures_root,
)

pytestmark = pytest.mark.tier1

CANDIDATE_SHA = "a" * 40
WORKFLOW_SHA = "b" * 40
RUN_ID = "123456"
ARTIFACT_ID = "789012"
ARTIFACT_DIGEST = "1" * 64
CONFIG_IDENTITY = "benchmarks/vn2/config/vn2-winning-loop.yaml"
INVENTORY_IDENTITY = "stage3/evidence/vn2-input-digests.json"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _environment() -> dict[str, object]:
    return {
        "arch": "x86_64",
        "cpu_model": "fixture cpu",
        "numpy": "2.3.1",
        "numpy_config": "OpenBLAS fixture",
        "os": {
            "id": "ubuntu",
            "pretty_name": "Ubuntu 24.04.3 LTS",
            "version_id": "24.04",
        },
        "python": "3.12.13",
        "runner_image": "ubuntu24/20260701.1",
        "thread_policy": {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }


def _valid_bundle(root: Path) -> Path:
    config_path, inventory_path = _trusted_inputs(root)
    environment = _environment()
    _write_json(root / "environment.json", environment)
    round_digests: dict[str, str] = {}
    for round_number in range(1, 7):
        orders = {f"sku-{index:03d}": float(index % 3) for index in range(599)}
        payload = _write_json(
            root / "orders" / f"round-{round_number}.json",
            {
                "origin": f"2026-0{round_number}-05",
                "orders": orders,
                "round_num": round_number,
            },
        )
        round_digests[f"round-{round_number}.json"] = _sha256(payload)
    _write_json(
        root / "orders" / "extraction-report.json",
        {
            "config": CONFIG_IDENTITY,
            "files": round_digests,
            "rounds": 6,
            "series_per_round": 599,
        },
    )
    payload_paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode(),
    )
    files = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path.read_bytes()),
        }
        for path in payload_paths
    ]
    listing = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in files).encode()
    (root / "files.sha256").write_bytes(listing)
    config_digest = _sha256(config_path.read_bytes())
    input_digest = _sha256(inventory_path.read_bytes())
    inner_digest = _sha256(listing)
    capture_digest = _capture_digest(files)
    _write_json(
        root / "manifest.json",
        {
            "actuals_semantics": "censored_sales_surrogate",
            "artifact_kind": "vn2-oracle-orders",
            "artifact_name": f"oracle-capture-{CANDIDATE_SHA}",
            "candidate_sha": CANDIDATE_SHA,
            "capture_digest": capture_digest,
            "config_digest": config_digest,
            "environment": environment,
            "environment_digest": _environment_digest(
                environment=environment,
                config_digest=config_digest,
                input_digest=input_digest,
                capture_digest=capture_digest,
            ),
            "files": files,
            "inner_bundle_digest": inner_digest,
            "input_inventory": INVENTORY_IDENTITY,
            "input_inventory_digest": input_digest,
            "oracle_commit": ORACLE_COMMIT,
            "oracle_lock_sha256": ORACLE_LOCK_SHA256,
            "oracle_tag": ORACLE_TAG,
            "run_id": RUN_ID,
            "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
            "schema": 1,
            "workflow_sha": WORKFLOW_SHA,
        },
    )
    return root


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _trusted_inputs(root: Path) -> tuple[Path, Path]:
    config_path = root.parent / "vn2-winning-loop.yaml"
    inventory_path = root.parent / "vn2-input-digests.json"
    config_path.write_bytes(b"fixture: vn2-winning-loop\n")
    inventory_path.write_bytes(b'{"files": []}\n')
    return config_path, inventory_path


def _validation_kwargs(root: Path) -> dict[str, object]:
    config_path, inventory_path = _trusted_inputs(root)
    return {
        "expected_candidate_sha": CANDIDATE_SHA,
        "expected_workflow_sha": WORKFLOW_SHA,
        "expected_run_id": RUN_ID,
        "expected_config_path": config_path,
        "expected_input_inventory_path": inventory_path,
    }


def _validate(root: Path, **overrides: object):
    kwargs = _validation_kwargs(root)
    kwargs.update(overrides)
    return validate_capture_bundle(root, **kwargs)  # type: ignore[arg-type]


def _environment_digest(
    *,
    environment: dict[str, object],
    config_digest: str,
    input_digest: str,
    capture_digest: str,
) -> str:
    os_release = environment["os"]
    assert isinstance(os_release, dict)
    canonical = json.dumps(
        {
            "actuals_semantics": "censored_sales_surrogate",
            "architecture": environment["arch"],
            "capture_digest": capture_digest,
            "config_digest": config_digest,
            "input_digest": input_digest,
            "lockfile_sha256": ORACLE_LOCK_SHA256,
            "os_release": {
                "id": os_release["id"],
                "version_id": os_release["version_id"],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return _sha256(canonical)


def test_capture_bundle_binds_complete_manifest_identity_and_payload_bytes(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")

    bundle = _validate(root)

    assert bundle.manifest.candidate_sha == CANDIDATE_SHA
    assert bundle.manifest.run_id == RUN_ID
    assert bundle.manifest.environment.thread_policy["OMP_NUM_THREADS"] == "1"
    assert len(bundle.manifest.files) == 8
    assert bundle.manifest_sha256 == _sha256((root / "manifest.json").read_bytes())


@pytest.mark.skipif(sys.platform != "linux", reason="symlink trust checks are Linux-ratified")
def test_promoted_captures_root_refuses_a_directory_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real-captures"
    target.mkdir()
    linked_root = tmp_path / "linked-captures"
    linked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(OracleEvidenceError, match="promoted captures root.*symbolic link"):
        validate_promoted_captures_root(linked_root)


@pytest.mark.skipif(sys.platform != "linux", reason="symlink trust checks are Linux-ratified")
def test_capture_bundle_refuses_a_root_directory_symlink(tmp_path: Path) -> None:
    target = _valid_bundle(tmp_path / "real-bundle")
    linked_root = tmp_path / "linked-bundle"
    linked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(OracleEvidenceError, match="capture bundle.*symbolic link"):
        _validate(linked_root)


@pytest.mark.skipif(sys.platform != "linux", reason="symlink trust checks are Linux-ratified")
def test_capture_receipt_refuses_a_file_symlink(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    bundle = _validate(root)
    target = tmp_path / "receipt.json"
    _write_json(target, _receipt_value(bundle))
    linked_receipt = tmp_path / "linked-receipt.json"
    linked_receipt.symlink_to(target)

    with pytest.raises(OracleEvidenceError, match="capture receipt.*symbolic link"):
        validate_capture_receipt(
            linked_receipt,
            bundle=bundle,
            expected_artifact_id=ARTIFACT_ID,
            expected_artifact_digest=ARTIFACT_DIGEST,
            expected_artifact_name=bundle.manifest.artifact_name,
            expected_producer_sha=CANDIDATE_SHA,
            expected_workflow_sha=WORKFLOW_SHA,
            expected_workflow_run_id=RUN_ID,
        )


def test_candidate_runner_validates_the_requested_capture_identity(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    project_root = Path(__file__).parents[2]

    completed = subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts" / "capture_oracle_vn2.py"),
            "validate",
            "--bundle",
            str(root),
            "--candidate-sha",
            CANDIDATE_SHA,
            "--workflow-sha",
            WORKFLOW_SHA,
            "--run-id",
            RUN_ID,
            "--config",
            str(_trusted_inputs(root)[0]),
            "--input-inventory",
            str(_trusted_inputs(root)[1]),
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validated capture inner digest" in completed.stdout


@pytest.mark.parametrize(
    "field",
    [
        "candidate_sha",
        "workflow_sha",
        "run_id",
        "config_digest",
        "input_inventory_digest",
        "actuals_semantics",
        "capture_digest",
        "environment",
        "environment_digest",
        "files",
        "inner_bundle_digest",
    ],
)
def test_capture_manifest_refuses_every_missing_evidence_family(
    tmp_path: Path,
    field: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    manifest = _manifest(root)
    del manifest[field]
    _write_json(root / "manifest.json", manifest)

    with pytest.raises(OracleEvidenceError, match=rf"fields mismatch.*{field}"):
        _validate(root)


@pytest.mark.parametrize(
    ("keyword", "wrong", "message"),
    [
        ("expected_candidate_sha", "e" * 40, "candidate_sha"),
        ("expected_workflow_sha", "f" * 40, "workflow_sha"),
        ("expected_run_id", "654321", "run_id"),
    ],
)
def test_capture_bundle_refuses_requested_identity_mismatch(
    tmp_path: Path,
    keyword: str,
    wrong: str,
    message: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")

    with pytest.raises(OracleEvidenceError, match=message):
        _validate(root, **{keyword: wrong})


def test_capture_bundle_refuses_payload_listing_and_inner_digest_mismatch(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "payload")
    round_path = root / "orders" / "round-3.json"
    round_path.write_bytes(round_path.read_bytes() + b" ")
    with pytest.raises(OracleEvidenceError, match="payload size mismatch"):
        _validate(root)

    root = _valid_bundle(tmp_path / "listing")
    (root / "files.sha256").write_bytes(b"0" * 64 + b"  wrong.json\n")
    with pytest.raises(OracleEvidenceError, match="files.sha256"):
        _validate(root)

    root = _valid_bundle(tmp_path / "inner")
    manifest = _manifest(root)
    manifest["inner_bundle_digest"] = "0" * 64
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(OracleEvidenceError, match="inner bundle digest"):
        _validate(root)


def test_capture_bundle_refuses_wrong_order_shape_and_unexpected_files(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "shape")
    round_path = root / "orders" / "round-1.json"
    payload = json.loads(round_path.read_text(encoding="utf-8"))
    payload["orders"].pop("sku-000")
    _write_json(round_path, payload)
    _rebind_payload(root, "orders/round-1.json")
    with pytest.raises(OracleEvidenceError, match="exactly 599"):
        _validate(root)

    root = _valid_bundle(tmp_path / "extra")
    (root / "untracked.txt").write_text("not manifest-bound", encoding="utf-8")
    with pytest.raises(OracleEvidenceError, match="file set mismatch"):
        _validate(root)


@pytest.mark.parametrize("field", ["config_digest", "input_inventory_digest"])
def test_capture_bundle_refuses_self_asserted_provenance_digests(
    tmp_path: Path,
    field: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    manifest = _manifest(root)
    manifest[field] = "e" * 64
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    manifest["environment_digest"] = _environment_digest(
        environment=environment,
        config_digest=str(manifest["config_digest"]),
        input_digest=str(manifest["input_inventory_digest"]),
        capture_digest=str(manifest["capture_digest"]),
    )
    _write_json(root / "manifest.json", manifest)

    with pytest.raises(OracleEvidenceError, match=rf"trusted.*{field}|{field}.*trusted"):
        _validate(root)


def test_capture_bundle_refuses_noncanonical_inventory_and_config_identities(
    tmp_path: Path,
) -> None:
    root = _valid_bundle(tmp_path / "inventory")
    manifest = _manifest(root)
    manifest["input_inventory"] = "fixtures/fake-inventory.json"
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(OracleEvidenceError, match="input_inventory must equal"):
        _validate(root)

    root = _valid_bundle(tmp_path / "config")
    report_path = root / "orders" / "extraction-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["config"] = "benchmarks/vn2/config/other.yaml"
    _write_json(report_path, report)
    _rebind_manifest(root)
    with pytest.raises(OracleEvidenceError, match="extraction report config"):
        _validate(root)


@pytest.mark.parametrize(
    ("field", "wrong", "message"),
    [
        (
            "os",
            {"id": "ubuntu", "pretty_name": "Ubuntu 22.04", "version_id": "22.04"},
            "Ubuntu 24.04",
        ),
        ("python", "3.11.9", "Python"),
        ("runner_image", "ubuntu22/20260701.1", "runner_image"),
    ],
)
def test_capture_bundle_refuses_nonratified_environment(
    tmp_path: Path,
    field: str,
    wrong: object,
    message: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    manifest = _manifest(root)
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    environment[field] = wrong
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "environment.json", environment)
    _rebind_manifest(root)

    with pytest.raises(OracleEvidenceError, match=message):
        _validate(root)


def test_environment_digest_excludes_per_run_provenance(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    before = _validate(root)
    manifest = _manifest(root)
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    environment.update(
        {
            "cpu_model": "different hosted CPU",
            "numpy": "2.5.2",
            "numpy_config": "different locked BLAS detail",
            "python": "3.12.14",
            "runner_image": "ubuntu24/20260708.2",
        }
    )
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "environment.json", environment)
    _rebind_manifest(root)

    after = _validate(root)
    assert after.manifest.inner_bundle_digest != before.manifest.inner_bundle_digest
    assert after.manifest.environment_digest == before.manifest.environment_digest


@pytest.mark.parametrize(
    ("target", "field", "wrong", "message"),
    [
        ("manifest", "schema", True, "manifest schema"),
        ("manifest", "schema", 1.0, "manifest schema"),
        ("report", "rounds", 6.0, "report rounds"),
        ("round", "round_num", True, "round_num"),
    ],
)
def test_capture_bundle_refuses_noninteger_contract_fields(
    tmp_path: Path,
    target: str,
    field: str,
    wrong: object,
    message: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    if target == "manifest":
        value = _manifest(root)
        _write_json(root / "manifest.json", {**value, field: wrong})
    elif target == "report":
        path = root / "orders" / "extraction-report.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        _write_json(path, {**value, field: wrong})
        _rebind_manifest(root)
    else:
        path = root / "orders" / "round-1.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        _write_json(path, {**value, field: wrong})
        _rebind_payload(root, "orders/round-1.json")

    with pytest.raises(OracleEvidenceError, match=message):
        _validate(root)


def test_capture_bundle_refuses_series_identity_drift_and_noncanonical_order(
    tmp_path: Path,
) -> None:
    root = _valid_bundle(tmp_path / "drift")
    round_path = root / "orders" / "round-2.json"
    payload = json.loads(round_path.read_text(encoding="utf-8"))
    payload["orders"].pop("sku-000")
    payload["orders"]["sku-999"] = 0.0
    _write_json(round_path, payload)
    _rebind_payload(root, "orders/round-2.json")
    with pytest.raises(OracleEvidenceError, match="identical in every round"):
        _validate(root)

    root = _valid_bundle(tmp_path / "order")
    round_path = root / "orders" / "round-2.json"
    payload = json.loads(round_path.read_text(encoding="utf-8"))
    payload["orders"] = dict(reversed(tuple(payload["orders"].items())))
    round_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _rebind_payload(root, "orders/round-2.json")
    with pytest.raises(OracleEvidenceError, match="canonical UTF-8 order"):
        _validate(root)


def test_capture_bundle_refuses_another_repository_run_url(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    manifest = _manifest(root)
    manifest["run_url"] = f"https://github.com/other/repository/actions/runs/{RUN_ID}"
    _write_json(root / "manifest.json", manifest)

    with pytest.raises(OracleEvidenceError, match="Vzlentin/calibre"):
        _validate(root)


def test_capture_receipt_binds_github_artifact_and_exact_promoted_bytes(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    bundle = _validate(root)
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "artifact_digest": ARTIFACT_DIGEST,
        "artifact_id": ARTIFACT_ID,
        "artifact_name": bundle.manifest.artifact_name,
        "inner_bundle_digest": bundle.manifest.inner_bundle_digest,
        "environment_digest": bundle.manifest.environment_digest,
        "manifest_sha256": bundle.manifest_sha256,
        "producer_sha": bundle.manifest.candidate_sha,
        "run_url": bundle.manifest.run_url,
        "schema": 1,
        "workflow_sha": bundle.manifest.workflow_sha,
        "workflow_run_id": bundle.manifest.run_id,
    }
    _write_json(receipt_path, receipt)

    validated = validate_capture_receipt(
        receipt_path,
        bundle=bundle,
        expected_artifact_id=ARTIFACT_ID,
        expected_artifact_digest=ARTIFACT_DIGEST,
        expected_artifact_name=bundle.manifest.artifact_name,
        expected_producer_sha=CANDIDATE_SHA,
        expected_workflow_sha=WORKFLOW_SHA,
        expected_workflow_run_id=RUN_ID,
    )
    assert validated.artifact_id == ARTIFACT_ID

    receipt["manifest_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(OracleEvidenceError, match="manifest_sha256"):
        validate_capture_receipt(
            receipt_path,
            bundle=bundle,
            expected_artifact_id=ARTIFACT_ID,
            expected_artifact_digest=ARTIFACT_DIGEST,
            expected_artifact_name=bundle.manifest.artifact_name,
            expected_producer_sha=CANDIDATE_SHA,
            expected_workflow_sha=WORKFLOW_SHA,
            expected_workflow_run_id=RUN_ID,
        )

    with pytest.raises(OracleEvidenceError, match="producer_sha"):
        validate_capture_receipt(
            receipt_path,
            bundle=bundle,
            expected_artifact_id=ARTIFACT_ID,
            expected_artifact_digest=ARTIFACT_DIGEST,
            expected_artifact_name=bundle.manifest.artifact_name,
            expected_producer_sha="e" * 40,
            expected_workflow_sha=WORKFLOW_SHA,
            expected_workflow_run_id=RUN_ID,
        )


def test_promoted_capture_binds_live_github_artifact_metadata(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    bundle = _validate(root)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, _receipt_value(bundle))

    promoted_bundle, receipt = validate_promoted_capture(
        root,
        receipt_path,
        artifact_metadata=_artifact_metadata(),
        run_metadata=_run_metadata(),
        expected_config_path=_trusted_inputs(root)[0],
        expected_input_inventory_path=_trusted_inputs(root)[1],
    )

    assert promoted_bundle.manifest.candidate_sha == CANDIDATE_SHA
    assert receipt.artifact_digest == ARTIFACT_DIGEST


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"expired": True}, "unexpired"),
        ({"id": True}, "positive integer"),
        ({"name": f"oracle-capture-{'c' * 40}"}, "candidate_sha"),
        ({"digest": f"sha256:{'2' * 64}"}, "artifact_digest"),
        ({"workflow_run": {"head_branch": "feature"}}, "from main"),
    ],
)
def test_promoted_capture_refuses_untrusted_artifact_metadata(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    bundle = _validate(root)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, _receipt_value(bundle))
    metadata = _artifact_metadata()
    if "workflow_run" in mutation:
        workflow_run = metadata["workflow_run"]
        assert isinstance(workflow_run, dict)
        nested = mutation["workflow_run"]
        assert isinstance(nested, dict)
        workflow_run.update(nested)
    else:
        metadata.update(mutation)

    with pytest.raises(OracleEvidenceError, match=message):
        validate_promoted_capture(
            root,
            receipt_path,
            artifact_metadata=metadata,
            run_metadata=_run_metadata(),
            expected_config_path=_trusted_inputs(root)[0],
            expected_input_inventory_path=_trusted_inputs(root)[1],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"event": "push"}, "event"),
        ({"path": ".github/workflows/other.yml"}, "path"),
        ({"conclusion": "failure"}, "conclusion"),
        ({"head_sha": "c" * 40}, "SHAs do not match"),
        ({"repository": {"full_name": "other/repository"}}, "Vzlentin/calibre"),
    ],
)
def test_promoted_capture_refuses_wrong_workflow_run(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    bundle = _validate(root)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, _receipt_value(bundle))
    run_metadata = _run_metadata()
    run_metadata.update(mutation)

    with pytest.raises(OracleEvidenceError, match=message):
        validate_promoted_capture(
            root,
            receipt_path,
            artifact_metadata=_artifact_metadata(),
            run_metadata=run_metadata,
            expected_config_path=_trusted_inputs(root)[0],
            expected_input_inventory_path=_trusted_inputs(root)[1],
        )


def test_committed_capture_helper_validates_one_bundle_and_receipt(tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    captures.mkdir()
    bundle_root = captures / CANDIDATE_SHA
    bundle = _validate(_valid_bundle(bundle_root))
    receipt_path = captures / f"{CANDIDATE_SHA}-receipt.json"
    (captures / "vn2-winning-loop.yaml").unlink()
    (captures / "vn2-input-digests.json").unlink()
    _write_json(receipt_path, _receipt_value(bundle))

    validated_bundle, validated_receipt = validate_committed_promoted_capture(captures)

    assert validated_bundle.manifest == bundle.manifest
    assert validated_receipt.artifact_id == ARTIFACT_ID


@pytest.mark.parametrize("entry_count", [0, 3])
def test_committed_capture_helper_bounds_root_enumeration(tmp_path: Path, entry_count: int) -> None:
    captures = tmp_path / "captures"
    captures.mkdir()
    if entry_count == 3:
        (captures / "a").mkdir()
        (captures / "b").write_bytes(b"receipt")
        (captures / "c").write_bytes(b"extra")

    with pytest.raises(
        OracleEvidenceError,
        match="promoted captures root must contain exactly one SHA-named bundle and receipt",
    ):
        validate_committed_promoted_capture(captures)


@pytest.mark.skipif(sys.platform != "linux", reason="symlink trust checks are Linux-ratified")
def test_committed_capture_helper_rejects_symlink_bundle(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    real_bundle = _valid_bundle(tmp_path / "real" / CANDIDATE_SHA)
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / CANDIDATE_SHA).symlink_to(real_bundle, target_is_directory=True)
    (captures / f"{CANDIDATE_SHA}-receipt.json").write_bytes(b"{}")

    with pytest.raises(OracleEvidenceError, match="symbolic link"):
        validate_committed_promoted_capture(captures)


def test_public_capture_validator_requires_trusted_paths(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path / "bundle")
    with pytest.raises(OracleEvidenceError, match="requires trusted"):
        validate_capture_bundle(
            root,
            expected_candidate_sha=CANDIDATE_SHA,
            expected_workflow_sha=WORKFLOW_SHA,
            expected_run_id=RUN_ID,
            expected_config_path=None,  # type: ignore[arg-type]
            expected_input_inventory_path=None,  # type: ignore[arg-type]
        )


def _receipt_value(bundle: CaptureBundle) -> dict[str, object]:
    manifest = bundle.manifest
    return {
        "artifact_digest": ARTIFACT_DIGEST,
        "artifact_id": ARTIFACT_ID,
        "artifact_name": manifest.artifact_name,
        "environment_digest": manifest.environment_digest,
        "inner_bundle_digest": manifest.inner_bundle_digest,
        "manifest_sha256": bundle.manifest_sha256,
        "producer_sha": manifest.candidate_sha,
        "run_url": manifest.run_url,
        "schema": 1,
        "workflow_run_id": manifest.run_id,
        "workflow_sha": manifest.workflow_sha,
    }


def _artifact_metadata() -> dict[str, object]:
    return {
        "digest": f"sha256:{ARTIFACT_DIGEST}",
        "expired": False,
        "id": int(ARTIFACT_ID),
        "name": f"oracle-capture-{CANDIDATE_SHA}",
        "workflow_run": {
            "head_branch": "main",
            "head_sha": WORKFLOW_SHA,
            "id": int(RUN_ID),
        },
    }


def _run_metadata() -> dict[str, object]:
    return {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": WORKFLOW_SHA,
        "html_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
        "id": int(RUN_ID),
        "path": ".github/workflows/oracle-capture.yml",
        "repository": {"full_name": "Vzlentin/calibre"},
        "status": "completed",
    }


def _rebind_payload(root: Path, relative_path: str) -> None:
    payload = root / relative_path
    report_path = root / "orders" / "extraction-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["files"][Path(relative_path).name] = _sha256(payload.read_bytes())
    _write_json(report_path, report)
    _rebind_manifest(root)


def _rebind_manifest(root: Path) -> None:
    manifest = _manifest(root)
    for entry in manifest["files"]:
        path = root / entry["path"]
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = _sha256(path.read_bytes())
    listing = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in manifest["files"]
    ).encode()
    (root / "files.sha256").write_bytes(listing)
    manifest["inner_bundle_digest"] = _sha256(listing)
    manifest["capture_digest"] = _capture_digest(manifest["files"])
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    config_digest = manifest["config_digest"]
    input_digest = manifest["input_inventory_digest"]
    assert isinstance(config_digest, str)
    assert isinstance(input_digest, str)
    manifest["environment_digest"] = _environment_digest(
        environment=environment,
        config_digest=config_digest,
        input_digest=input_digest,
        capture_digest=manifest["capture_digest"],
    )
    _write_json(root / "manifest.json", manifest)


def _capture_digest(files: object) -> str:
    assert isinstance(files, list)
    listing = "".join(
        f"{entry['sha256']}  {entry['path']}\n"
        for entry in files
        if isinstance(entry, dict) and str(entry["path"]).startswith("orders/")
    ).encode()
    return _sha256(listing)
