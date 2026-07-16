"""Exercise the strict successor VN2 tracking record contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from tests.vn2_fixtures import (
    BASE_WEEKS,
    synthetic_config_payload,
    write_config,
    write_dataset,
)

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.oracle import ORACLE_COMMIT, ORACLE_LOCK_SHA256, ORACLE_TAG
from newcalibre.protocols.vn2 import (
    THREAD_VARIABLES,
    VN2EvidenceEnvironment,
    VN2ResultError,
    emit_vn2_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
)
from newcalibre.protocols.vn2.tracking import (
    TRACKING_KIND,
    TRACKING_SCHEMA,
    TRACKING_SERIES_PATH,
    TrackingError,
    VN2TrackingRecord,
    build_promotion_receipt,
    build_tracking_record,
    compare_tracking_records,
    decide_append,
    parse_tracking_history,
    parse_tracking_record,
    promotion_receipt_path,
    validate_tracking_promotion,
    write_proposal_record,
)

pytestmark = pytest.mark.tier1

CANDIDATE = "a" * 40
WORKFLOW = "b" * 40
RUN_ID = "123456"
DIGEST = "1" * 64


def _environment() -> dict[str, object]:
    return {
        "arch": "x86_64",
        "cpu_model": "fixture cpu",
        "numpy": "2.3.1",
        "numpy_config": "OpenBLAS fixture",
        "os": {"id": "ubuntu", "pretty_name": "Ubuntu 24.04", "version_id": "24.04"},
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


def _record(*, total_cost: float = 3.0, candidate: str = CANDIDATE) -> VN2TrackingRecord:
    environment = _environment()
    config_digest = "2" * 64
    input_digest = "3" * 64
    environment_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "actuals_semantics": "censored_sales_surrogate",
                "architecture": environment["arch"],
                "config_digest": config_digest,
                "input_inventory_digest": input_digest,
                "lockfile_digest": "4" * 64,
                "os_id": environment["os"]["id"],  # type: ignore[index]
                "os_version": environment["os"]["version_id"],  # type: ignore[index]
                "promoted_capture_digest": "5" * 64,
            },
            path="GA1 comparability key",
        )
    ).hexdigest()
    toolchain_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "numpy": environment["numpy"],
                "numpy_config": environment["numpy_config"],
                "python": environment["python"],
                "schema": 1,
            },
            path="tracking toolchain",
        )
    ).hexdigest()
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "artifact_kind": "vn2-gate-a-results",
                "candidate_sha": candidate,
                "config_digest": config_digest,
                "environment_digest": environment_digest,
                "input_inventory_digest": input_digest,
            },
            path="tracking identity",
        )
    ).hexdigest()
    payload = {
        "schema": TRACKING_SCHEMA,
        "record_kind": TRACKING_KIND,
        "identity": identity,
        "subject": {"repository": "Vzlentin/calibre", "candidate_sha": candidate},
        "workflow": {
            "definition_ref": "Vzlentin/calibre/.github/workflows/newcalibre.yml@main",
            "definition_sha": WORKFLOW,
            "run_id": RUN_ID,
            "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
        },
        "result_artifact": {
            "id": "789012",
            "name": f"vn2-acceptance-{candidate}",
            "digest": DIGEST,
        },
        "result_bundle": {
            "artifact_kind": "vn2-gate-a-results",
            "manifest_sha256": DIGEST,
            "inner_bundle_digest": DIGEST,
            "provenance_digest": DIGEST,
            "files": {
                "r1-orders.jsonl": DIGEST,
                "r2-cost-ledger.jsonl": DIGEST,
                "r3-final-triple.json": DIGEST,
                "r4-cost-trajectory.json": DIGEST,
            },
        },
        "evidence": {
            "config": {"path": "benchmarks/vn2/protocol.yaml", "digest": config_digest},
            "input_inventory": {
                "path": "benchmarks/vn2/vn2-input-digests.json",
                "digest": input_digest,
            },
            "lockfile": {"path": "uv.lock", "digest": "4" * 64},
            "promoted_capture": {
                "artifact_id": "789013",
                "artifact_digest": DIGEST,
                "artifact_name": f"oracle-capture-{candidate}",
                "capture_digest": "5" * 64,
                "manifest_sha256": DIGEST,
                "inner_bundle_digest": DIGEST,
                "environment_digest": DIGEST,
                "producer_sha": candidate,
                "workflow_sha": WORKFLOW,
                "workflow_run_id": RUN_ID,
                "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
            },
            "actuals_semantics": "censored_sales_surrogate",
            "session": {"id": DIGEST, "series_count": 1, "series_identity_digest": DIGEST},
        },
        "environment": {
            "facts": environment,
            "digest": environment_digest,
            "toolchain_digest": toolchain_digest,
        },
        "objective": {
            "holding_cost": 1.0,
            "shortage_cost": total_cost - 1.0,
            "total_cost": total_cost,
        },
    }

    return VN2TrackingRecord._from_evidence(payload)


def _json_payload(record: VN2TrackingRecord) -> dict[str, object]:
    return json.loads(record.to_json())


def _ga1_digest(payload: dict[str, object]) -> str:
    evidence = payload["evidence"]
    environment = payload["environment"]
    assert isinstance(evidence, dict)
    assert isinstance(environment, dict)
    facts = environment["facts"]
    assert isinstance(facts, dict)
    os_facts = facts["os"]
    assert isinstance(os_facts, dict)
    promoted_capture = evidence["promoted_capture"]
    config = evidence["config"]
    input_inventory = evidence["input_inventory"]
    lockfile = evidence["lockfile"]
    assert isinstance(promoted_capture, dict)
    assert isinstance(config, dict)
    assert isinstance(input_inventory, dict)
    assert isinstance(lockfile, dict)
    key = {
        "actuals_semantics": evidence["actuals_semantics"],
        "architecture": facts["arch"],
        "config_digest": config["digest"],
        "input_inventory_digest": input_inventory["digest"],
        "lockfile_digest": lockfile["digest"],
        "os_id": os_facts["id"],
        "os_version": os_facts["version_id"],
        "promoted_capture_digest": promoted_capture["capture_digest"],
    }
    return hashlib.sha256(canonical_json_bytes(key, path="GA1 comparability key")).hexdigest()


def _promotion_fixture(
    tmp_path: Path,
) -> tuple[
    VN2TrackingRecord,
    Path,
    Path,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    result_archive = tmp_path / "result.zip"
    proposal_archive = tmp_path / "proposal.zip"
    result_archive.write_bytes(b"immutable result archive")
    proposal_archive.write_bytes(b"immutable proposal archive")
    result_digest = hashlib.sha256(result_archive.read_bytes()).hexdigest()
    proposal_digest = hashlib.sha256(proposal_archive.read_bytes()).hexdigest()
    payload = _json_payload(_record())
    payload["workflow"] = {
        "definition_ref": ("Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main"),
        "definition_sha": CANDIDATE,
        "run_id": RUN_ID,
        "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
    }
    payload["result_artifact"]["digest"] = result_digest  # type: ignore[index]
    record = VN2TrackingRecord._from_evidence(payload)
    workflow_run = {
        "head_branch": "main",
        "head_repository_id": 42,
        "head_sha": CANDIDATE,
        "id": int(RUN_ID),
        "repository_id": 42,
    }
    result_metadata = {
        "digest": f"sha256:{result_digest}",
        "expired": False,
        "id": 789012,
        "name": f"vn2-acceptance-{CANDIDATE}",
        "workflow_run": workflow_run,
    }
    proposal_metadata = {
        "digest": f"sha256:{proposal_digest}",
        "expired": False,
        "id": 789014,
        "name": f"vn2-tracking-proposal-{CANDIDATE}",
        "workflow_run": workflow_run,
    }
    run_metadata = {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": CANDIDATE,
        "html_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
        "id": int(RUN_ID),
        "path": ".github/workflows/newcalibre.yml",
        "repository": {"full_name": "Vzlentin/calibre", "id": 42},
        "run_attempt": 1,
        "status": "completed",
    }
    return (
        record,
        result_archive,
        proposal_archive,
        result_metadata,
        proposal_metadata,
        run_metadata,
    )


def _write_test_json(path: Path, value: object) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _create_committed_capture(root: Path) -> None:
    bundle_root = root / CANDIDATE
    bundle_root.mkdir(parents=True)
    environment = _environment()
    _write_test_json(bundle_root / "environment.json", environment)
    round_digests: dict[str, str] = {}
    for round_number in range(1, 7):
        round_payload = _write_test_json(
            bundle_root / "orders" / f"round-{round_number}.json",
            {
                "origin": f"2026-0{round_number}-05",
                "orders": {f"sku-{index:03d}": float(index % 3) for index in range(599)},
                "round_num": round_number,
            },
        )
        round_digests[f"round-{round_number}.json"] = hashlib.sha256(round_payload).hexdigest()
    _write_test_json(
        bundle_root / "orders" / "extraction-report.json",
        {
            "config": "benchmarks/vn2/config/vn2-winning-loop.yaml",
            "files": round_digests,
            "rounds": 6,
            "series_per_round": 599,
        },
    )
    payload_paths = sorted(
        (path for path in bundle_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(bundle_root).as_posix().encode(),
    )
    files = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(bundle_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in payload_paths
    ]
    listing = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in files).encode()
    (bundle_root / "files.sha256").write_bytes(listing)
    capture_listing = "".join(
        f"{entry['sha256']}  {entry['path']}\n"
        for entry in files
        if str(entry["path"]).startswith("orders/")
    ).encode()
    capture_digest = hashlib.sha256(capture_listing).hexdigest()
    config_digest = "2" * 64
    input_digest = "3" * 64
    os_facts = environment["os"]
    assert isinstance(os_facts, dict)
    environment_digest = hashlib.sha256(
        json.dumps(
            {
                "actuals_semantics": "censored_sales_surrogate",
                "architecture": environment["arch"],
                "capture_digest": capture_digest,
                "config_digest": config_digest,
                "input_digest": input_digest,
                "lockfile_sha256": ORACLE_LOCK_SHA256,
                "os_release": {
                    "id": os_facts["id"],
                    "version_id": os_facts["version_id"],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = {
        "actuals_semantics": "censored_sales_surrogate",
        "artifact_kind": "vn2-oracle-orders",
        "artifact_name": f"oracle-capture-{CANDIDATE}",
        "candidate_sha": CANDIDATE,
        "capture_digest": capture_digest,
        "config_digest": config_digest,
        "environment": environment,
        "environment_digest": environment_digest,
        "files": files,
        "inner_bundle_digest": hashlib.sha256(listing).hexdigest(),
        "input_inventory": "stage3/evidence/vn2-input-digests.json",
        "input_inventory_digest": input_digest,
        "oracle_commit": ORACLE_COMMIT,
        "oracle_lock_sha256": ORACLE_LOCK_SHA256,
        "oracle_tag": ORACLE_TAG,
        "run_id": RUN_ID,
        "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
        "schema": 1,
        "workflow_sha": WORKFLOW,
    }
    manifest_bytes = _write_test_json(bundle_root / "manifest.json", manifest)
    _write_test_json(
        root / f"{CANDIDATE}-receipt.json",
        {
            "artifact_digest": DIGEST,
            "artifact_id": "789013",
            "artifact_name": manifest["artifact_name"],
            "environment_digest": environment_digest,
            "inner_bundle_digest": manifest["inner_bundle_digest"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "producer_sha": CANDIDATE,
            "run_url": manifest["run_url"],
            "schema": 1,
            "workflow_run_id": RUN_ID,
            "workflow_sha": WORKFLOW,
        },
    )


def _result_environment() -> VN2EvidenceEnvironment:
    return VN2EvidenceEnvironment(
        arch="x86_64",
        cpu_model="Synthetic x86_64",
        os_id="ubuntu",
        os_version_id="24.04",
        os_pretty_name="Ubuntu 24.04.2 LTS",
        python="3.12.10",
        numpy="2.3.1",
        numpy_config="OpenBLAS synthetic provenance",
        runner_image="ubuntu24/20250701.1",
        thread_policy={name: "1" for name in THREAD_VARIABLES},
    )


def _create_result_bundle(
    root: Path,
    *,
    candidate_sha: str,
) -> tuple[Path, Path, Path, Path]:
    data_root, inventory_path, config_path = write_dataset(root / "inputs")
    config_payload = synthetic_config_payload()
    model_config = config_payload["model_config"]
    assert isinstance(model_config, dict)
    model_config["m"] = len(BASE_WEEKS)
    write_config(config_path, config_payload)
    config = load_vn2_config(config_path)
    dataset = load_vn2_dataset(data_root, inventory_path, config)
    lock_path = root / "inputs" / "uv.lock"
    lock_path.write_bytes(b"synthetic locked environment\n")
    result_root = root / "result-bundle"
    emit_vn2_result_bundle(
        result_root,
        result=run_vn2(dataset),
        config=config,
        candidate_sha=candidate_sha,
        workflow_sha=candidate_sha,
        run_id=RUN_ID,
        run_url=f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
        config_path=config_path,
        input_inventory_path=inventory_path,
        lock_path=lock_path,
        environment=_result_environment(),
    )
    return result_root, config_path, inventory_path, lock_path


def _archive_directory(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix().encode()):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def _archive_file(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        archive.write(source, source.name)


def _initialize_promotion_repository(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for key, value in (("user.name", "VN2 test"), ("user.email", "vn2@example.invalid")):
        subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
    project = root / "newcalibre"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'newcalibre'\n")
    subprocess.run(["git", "-C", str(root), "add", "newcalibre/pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_promotion(
    root: Path,
    *,
    candidate_sha: str,
    record: VN2TrackingRecord,
    receipt: bytes,
    symlink_series: bool = False,
) -> str:
    series_path = root / TRACKING_SERIES_PATH
    series_path.parent.mkdir(parents=True)
    if symlink_series:
        series_path.symlink_to("untrusted-series.jsonl")
    else:
        series_path.write_bytes(record.to_bytes())
    receipt_relative = promotion_receipt_path(candidate_sha)
    receipt_path = root / receipt_relative
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(receipt)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "add",
            "--",
            TRACKING_SERIES_PATH,
            receipt_relative,
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "promotion"], check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_promotion_repository(
    root: Path,
    *,
    record: VN2TrackingRecord,
    receipt: bytes,
    symlink_series: bool = False,
) -> tuple[str, str]:
    base_sha = _initialize_promotion_repository(root)
    head_sha = _commit_promotion(
        root,
        candidate_sha=CANDIDATE,
        record=record,
        receipt=receipt,
        symlink_series=symlink_series,
    )
    return base_sha, head_sha


def test_promote_cli_validates_real_git_and_evidence_chain(tmp_path: Path) -> None:
    from scripts import vn2_tracking

    repository_root = tmp_path / "repository"
    base_sha = _initialize_promotion_repository(repository_root)
    capture_root = tmp_path / "captures"
    _create_committed_capture(capture_root)
    result_root, config_path, inventory_path, lock_path = _create_result_bundle(
        tmp_path / "evidence",
        candidate_sha=base_sha,
    )
    result_archive = tmp_path / "result.zip"
    _archive_directory(result_root, result_archive)
    result_digest = hashlib.sha256(result_archive.read_bytes()).hexdigest()
    run_url = f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}"
    result_artifact_name = f"vn2-acceptance-{base_sha}"
    proposal_artifact_name = f"vn2-tracking-proposal-{base_sha}"
    record = build_tracking_record(
        result_root,
        capture_root,
        candidate_sha=base_sha,
        definition_ref="Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main",
        definition_sha=base_sha,
        run_id=RUN_ID,
        run_url=run_url,
        result_artifact_id="789012",
        result_artifact_name=result_artifact_name,
        result_artifact_digest=result_digest,
        config_path=config_path,
        input_inventory_path=inventory_path,
        lockfile_path=lock_path,
    )
    proposal = tmp_path / "proposal.jsonl"
    proposal.write_bytes(record.to_bytes())
    proposal_archive = tmp_path / "proposal.zip"
    _archive_file(proposal, proposal_archive)
    proposal_digest = hashlib.sha256(proposal_archive.read_bytes()).hexdigest()
    workflow_run = {
        "head_branch": "main",
        "head_repository_id": 42,
        "head_sha": base_sha,
        "id": int(RUN_ID),
        "repository_id": 42,
    }
    result_metadata = {
        "digest": f"sha256:{result_digest}",
        "expired": False,
        "id": 789012,
        "name": result_artifact_name,
        "workflow_run": workflow_run,
    }
    proposal_metadata = {
        "digest": f"sha256:{proposal_digest}",
        "expired": False,
        "id": 789014,
        "name": proposal_artifact_name,
        "workflow_run": workflow_run,
    }
    run_metadata = {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": base_sha,
        "html_url": run_url,
        "id": int(RUN_ID),
        "path": ".github/workflows/newcalibre.yml",
        "repository": {"full_name": "Vzlentin/calibre", "id": 42},
        "run_attempt": 1,
        "status": "completed",
    }
    result_metadata_path = tmp_path / "result-metadata.json"
    proposal_metadata_path = tmp_path / "proposal-metadata.json"
    run_metadata_path = tmp_path / "run-metadata.json"
    _write_test_json(result_metadata_path, result_metadata)
    _write_test_json(proposal_metadata_path, proposal_metadata)
    _write_test_json(run_metadata_path, run_metadata)
    receipt = build_promotion_receipt(
        record,
        proposal,
        result_artifact_metadata=result_metadata,
        proposal_artifact_metadata=proposal_metadata,
        run_metadata=run_metadata,
        result_archive=result_archive,
        proposal_archive=proposal_archive,
    )
    head_sha = _commit_promotion(
        repository_root,
        candidate_sha=base_sha,
        record=record,
        receipt=receipt.to_bytes(),
    )
    arguments = [
        "promote",
        "--result-root",
        str(result_root),
        "--capture-root",
        str(capture_root),
        "--candidate-sha",
        base_sha,
        "--workflow-ref",
        "Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main",
        "--workflow-sha",
        base_sha,
        "--run-id",
        RUN_ID,
        "--run-url",
        run_url,
        "--result-artifact-id",
        "789012",
        "--result-artifact-name",
        result_artifact_name,
        "--result-artifact-digest",
        result_digest,
        "--config",
        str(config_path),
        "--input-inventory",
        str(inventory_path),
        "--lockfile",
        str(lock_path),
        "--proposal",
        str(proposal),
        "--result-artifact-metadata",
        str(result_metadata_path),
        "--proposal-artifact-metadata",
        str(proposal_metadata_path),
        "--run-metadata",
        str(run_metadata_path),
        "--result-archive",
        str(result_archive),
        "--proposal-archive",
        str(proposal_archive),
        "--repository-root",
        str(repository_root),
        "--base-sha",
        base_sha,
        "--head-sha",
        head_sha,
        "--default-branch-sha",
        base_sha,
    ]
    production_validator_called = False
    validator_code = validate_tracking_promotion.__code__
    previous_profiler = sys.getprofile()

    def observe_validator(frame: object, event: str, argument: object) -> None:
        del argument
        nonlocal production_validator_called
        if event == "call" and getattr(frame, "f_code", None) is validator_code:
            production_validator_called = True

    sys.setprofile(observe_validator)
    try:
        assert vn2_tracking.main(arguments) == 0
    finally:
        sys.setprofile(previous_profiler)
    assert production_validator_called


def test_promotion_git_diff_preserves_deleted_rename_source(tmp_path: Path) -> None:
    from scripts import vn2_tracking

    repository_root = tmp_path / "rename-repository"
    _initialize_promotion_repository(repository_root)
    old_candidate = "c" * 40
    series_path = repository_root / TRACKING_SERIES_PATH
    series_path.parent.mkdir(parents=True)
    series_path.write_bytes(_record(candidate=old_candidate).to_bytes())
    old_receipt_relative = promotion_receipt_path(old_candidate)
    old_receipt_path = repository_root / old_receipt_relative
    old_receipt_path.write_bytes(b'{"old":true}\n')
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "add",
            "--",
            TRACKING_SERIES_PATH,
            old_receipt_relative,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository_root), "commit", "-qm", "existing tracking evidence"],
        check=True,
    )
    base_sha = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    new_receipt_relative = promotion_receipt_path(CANDIDATE)
    old_receipt_path.rename(repository_root / new_receipt_relative)
    series_path.write_bytes(series_path.read_bytes() + _record().to_bytes())
    subprocess.run(["git", "-C", str(repository_root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "commit", "-qm", "rename receipt while appending"],
        check=True,
    )
    head_sha = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    detected_with_renames = {
        path
        for path in subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "diff",
                "--name-only",
                "-z",
                base_sha,
                head_sha,
                "--",
            ],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
        if path
    }
    assert detected_with_renames == {TRACKING_SERIES_PATH, new_receipt_relative}
    with pytest.raises(TrackingError, match="exactly"):
        vn2_tracking._promotion_git_blobs(
            repository_root,
            base_sha=base_sha,
            head_sha=head_sha,
            candidate_sha=CANDIDATE,
        )


@pytest.mark.parametrize(
    ("mode", "object_type"),
    [
        ("100755", "blob"),
        ("120000", "blob"),
        ("040000", "tree"),
        ("160000", "commit"),
    ],
)
def test_git_blob_rejects_noncanonical_tree_entry_modes_and_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    object_type: str,
) -> None:
    from scripts import vn2_tracking

    path = "stage3/evidence/tracking/series.jsonl"
    entry = f"{mode} {object_type} {'1' * 40}\t{path}\0".encode()

    def inspect(root: Path, *args: str) -> bytes:
        assert args[0] == "ls-tree"
        return entry

    monkeypatch.setattr(vn2_tracking, "_git", inspect)
    with pytest.raises(TrackingError, match="mode 100644 and type blob"):
        vn2_tracking._git_blob(tmp_path, CANDIDATE, path)


def test_git_blob_distinguishes_missing_from_ambiguous_tree_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import vn2_tracking

    path = "stage3/evidence/tracking/series.jsonl"
    monkeypatch.setattr(vn2_tracking, "_git", lambda root, *args: b"")
    assert (
        vn2_tracking._git_blob(
            tmp_path,
            CANDIDATE,
            path,
            allow_missing=True,
        )
        is None
    )
    with pytest.raises(TrackingError, match="missing"):
        vn2_tracking._git_blob(tmp_path, CANDIDATE, path)

    entry = f"100644 blob {'1' * 40}\t{path}\0".encode()
    monkeypatch.setattr(vn2_tracking, "_git", lambda root, *args: entry + entry)
    with pytest.raises(TrackingError, match="ambiguous"):
        vn2_tracking._git_blob(
            tmp_path,
            CANDIDATE,
            path,
            allow_missing=True,
        )


def test_promotion_receipt_proves_exact_first_and_later_append(tmp_path: Path) -> None:
    import newcalibre.protocols.vn2.tracking as promotion

    record, result_zip, proposal_zip, result_meta, proposal_meta, run_meta = _promotion_fixture(
        tmp_path
    )
    receipt = promotion.build_promotion_receipt(
        record,
        record.to_bytes(),
        result_artifact_metadata=result_meta,
        proposal_artifact_metadata=proposal_meta,
        run_metadata=run_meta,
        result_archive=result_zip,
        proposal_archive=proposal_zip,
    )
    parsed_receipt = promotion.parse_promotion_receipt(receipt.to_bytes())
    assert parsed_receipt == receipt
    promotion.validate_promotion_paths(
        [
            promotion.TRACKING_SERIES_PATH,
            promotion.promotion_receipt_path(CANDIDATE),
        ],
        candidate_sha=CANDIDATE,
    )
    promotion.validate_tracking_promotion(
        record,
        record.to_bytes(),
        receipt.to_bytes(),
        record.to_bytes(),
        prior_history=None,
        result_artifact_metadata=result_meta,
        proposal_artifact_metadata=proposal_meta,
        run_metadata=run_meta,
        result_archive=result_zip,
        proposal_archive=proposal_zip,
        base_sha=CANDIDATE,
        default_branch_sha=CANDIDATE,
    )
    prior = _record(candidate="c" * 40)
    promotion.validate_tracking_promotion(
        record,
        record.to_bytes(),
        receipt.to_bytes(),
        prior.to_bytes() + record.to_bytes(),
        prior_history=prior.to_bytes(),
        result_artifact_metadata=result_meta,
        proposal_artifact_metadata=proposal_meta,
        run_metadata=run_meta,
        result_archive=result_zip,
        proposal_archive=proposal_zip,
        base_sha=CANDIDATE,
        default_branch_sha=CANDIDATE,
    )


def test_promotion_refuses_replay_conflict_and_old_prefix_mutation(tmp_path: Path) -> None:
    import newcalibre.protocols.vn2.tracking as promotion

    record, result_zip, proposal_zip, result_meta, proposal_meta, run_meta = _promotion_fixture(
        tmp_path
    )
    receipt = promotion.build_promotion_receipt(
        record,
        record.to_bytes(),
        result_artifact_metadata=result_meta,
        proposal_artifact_metadata=proposal_meta,
        run_metadata=run_meta,
        result_archive=result_zip,
        proposal_archive=proposal_zip,
    ).to_bytes()
    common = {
        "result_artifact_metadata": result_meta,
        "proposal_artifact_metadata": proposal_meta,
        "run_metadata": run_meta,
        "result_archive": result_zip,
        "proposal_archive": proposal_zip,
        "base_sha": CANDIDATE,
        "default_branch_sha": CANDIDATE,
    }
    with pytest.raises(TrackingError, match="no-op"):
        promotion.validate_tracking_promotion(
            record,
            record.to_bytes(),
            receipt,
            record.to_bytes() + record.to_bytes(),
            prior_history=record.to_bytes(),
            **common,
        )
    conflict = _json_payload(record)
    conflict["objective"]["holding_cost"] = 2.0  # type: ignore[index]
    conflict["objective"]["shortage_cost"] = 1.0  # type: ignore[index]
    conflict_record = VN2TrackingRecord._from_evidence(conflict)
    with pytest.raises(TrackingError, match="conflicts"):
        promotion.validate_tracking_promotion(
            record,
            record.to_bytes(),
            receipt,
            conflict_record.to_bytes() + record.to_bytes(),
            prior_history=conflict_record.to_bytes(),
            **common,
        )
    prior = _record(candidate="c" * 40)
    changed_prefix = prior.to_bytes().replace(b'"holding_cost":1.0', b'"holding_cost":2.0')
    with pytest.raises(TrackingError):
        promotion.validate_tracking_promotion(
            record,
            record.to_bytes(),
            receipt,
            changed_prefix + record.to_bytes(),
            prior_history=prior.to_bytes(),
            **common,
        )


@pytest.mark.parametrize(
    ("target", "path", "value"),
    [
        ("run", ("event",), "push"),
        ("run", ("path",), ".github/workflows/evil.yml"),
        ("run", ("head_branch",), "feature"),
        ("run", ("head_sha",), "c" * 40),
        ("run", ("conclusion",), "failure"),
        ("run", ("run_attempt",), 2),
        ("run", ("repository", "full_name"), "evil/calibre"),
        ("result", ("expired",), True),
        ("result", ("workflow_run", "id"), 999),
        ("proposal", ("workflow_run", "head_sha"), "c" * 40),
        ("proposal", ("name",), "vn2-tracking-proposal-wrong"),
        ("proposal", ("digest",), "sha256:" + "f" * 64),
    ],
)
def test_promotion_refuses_live_run_and_artifact_mismatches(
    tmp_path: Path,
    target: str,
    path: tuple[str, ...],
    value: object,
) -> None:
    import newcalibre.protocols.vn2.tracking as promotion

    record, result_zip, proposal_zip, result_meta, proposal_meta, run_meta = _promotion_fixture(
        tmp_path
    )
    values = {
        "result": copy.deepcopy(result_meta),
        "proposal": copy.deepcopy(proposal_meta),
        "run": copy.deepcopy(run_meta),
    }
    selected = values[target]
    cursor = selected
    for key in path[:-1]:
        nested = cursor[key]
        assert isinstance(nested, dict)
        cursor = nested
    cursor[path[-1]] = value
    with pytest.raises(TrackingError):
        promotion.build_promotion_receipt(
            record,
            record.to_bytes(),
            result_artifact_metadata=values["result"],
            proposal_artifact_metadata=values["proposal"],
            run_metadata=values["run"],
            result_archive=result_zip,
            proposal_archive=proposal_zip,
        )


def test_promotion_refuses_proposal_archive_receipt_and_tip_mismatches(
    tmp_path: Path,
) -> None:
    import newcalibre.protocols.vn2.tracking as promotion

    record, result_zip, proposal_zip, result_meta, proposal_meta, run_meta = _promotion_fixture(
        tmp_path
    )
    receipt = promotion.build_promotion_receipt(
        record,
        record.to_bytes(),
        result_artifact_metadata=result_meta,
        proposal_artifact_metadata=proposal_meta,
        run_metadata=run_meta,
        result_archive=result_zip,
        proposal_archive=proposal_zip,
    )
    common = {
        "prior_history": None,
        "result_artifact_metadata": result_meta,
        "proposal_artifact_metadata": proposal_meta,
        "run_metadata": run_meta,
        "result_archive": result_zip,
        "proposal_archive": proposal_zip,
        "base_sha": CANDIDATE,
        "default_branch_sha": CANDIDATE,
    }
    with pytest.raises(TrackingError, match="proposal bytes"):
        promotion.validate_tracking_promotion(
            record,
            _record(candidate="c" * 40).to_bytes(),
            receipt.to_bytes(),
            record.to_bytes(),
            **common,
        )
    with pytest.raises(TrackingError, match="base"):
        promotion.validate_tracking_promotion(
            record,
            record.to_bytes(),
            receipt.to_bytes(),
            record.to_bytes(),
            **{**common, "base_sha": "c" * 40},
        )
    with pytest.raises(TrackingError, match="default-branch"):
        promotion.validate_tracking_promotion(
            record,
            record.to_bytes(),
            receipt.to_bytes(),
            record.to_bytes(),
            **{**common, "default_branch_sha": "c" * 40},
        )
    receipt_value = json.loads(receipt.to_bytes())
    receipt_value["record_sha256"] = "f" * 64
    mismatched_receipt = (
        canonical_json_bytes(receipt_value, path="fixture promotion receipt") + b"\n"
    )
    with pytest.raises(TrackingError, match="validated live evidence"):
        promotion.validate_tracking_promotion(
            record,
            record.to_bytes(),
            mismatched_receipt,
            record.to_bytes(),
            **common,
        )
    result_zip.write_bytes(b"mutated result archive")
    with pytest.raises(TrackingError, match="result archive digest"):
        promotion.build_promotion_receipt(
            record,
            record.to_bytes(),
            result_artifact_metadata=result_meta,
            proposal_artifact_metadata=proposal_meta,
            run_metadata=run_meta,
            result_archive=result_zip,
            proposal_archive=proposal_zip,
        )


def test_promotion_receipt_wire_paths_and_publication_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newcalibre.protocols.vn2._tracking_persistence as persistence
    import newcalibre.protocols.vn2.tracking as promotion

    record, result_zip, proposal_zip, result_meta, proposal_meta, run_meta = _promotion_fixture(
        tmp_path
    )
    receipt = promotion.build_promotion_receipt(
        record,
        record.to_bytes(),
        result_artifact_metadata=result_meta,
        proposal_artifact_metadata=proposal_meta,
        run_metadata=run_meta,
        result_archive=result_zip,
        proposal_archive=proposal_zip,
    )
    receipt_value = json.loads(receipt.to_bytes())
    pretty = (json.dumps(receipt_value, indent=2) + "\n").encode()
    with pytest.raises(TrackingError):
        promotion.parse_promotion_receipt(pretty)
    with pytest.raises(TrackingError, match="LF"):
        promotion.parse_promotion_receipt(receipt.to_bytes()[:-1] + b"\r\n")
    receipt_value["unexpected"] = None
    unknown = canonical_json_bytes(receipt_value, path="fixture receipt") + b"\n"
    with pytest.raises(TrackingError, match="fields"):
        promotion.parse_promotion_receipt(unknown)
    for paths in (
        [promotion.TRACKING_SERIES_PATH],
        [
            promotion.TRACKING_SERIES_PATH,
            promotion.promotion_receipt_path(CANDIDATE),
            "README.md",
        ],
        [
            promotion.TRACKING_SERIES_PATH,
            "stage3/evidence/tracking/../wrong-receipt.json",
        ],
    ):
        with pytest.raises(TrackingError):
            promotion.validate_promotion_paths(paths, candidate_sha=CANDIDATE)

    root = tmp_path / "newcalibre"
    monkeypatch.setattr(persistence, "_TRUSTED_PROJECT_ROOT", root)
    (root / "artifacts").mkdir(parents=True)
    output = root / "artifacts" / "receipt.json"
    assert promotion.write_promotion_receipt(receipt, output)
    assert output.read_bytes() == receipt.to_bytes()
    with pytest.raises(TrackingError):
        promotion.write_promotion_receipt(receipt, tmp_path / "outside.json")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires Unix special files")
def test_tracking_path_reads_refuse_symlinks_and_nonregular_leaves(tmp_path: Path) -> None:
    record_path = tmp_path / "record.jsonl"
    record_path.write_bytes(_record().to_bytes())
    symlink = tmp_path / "record-link.jsonl"
    symlink.symlink_to(record_path)
    with pytest.raises(TrackingError, match="non-symlinks"):
        parse_tracking_record(symlink)
    target_directory = tmp_path / "records"
    target_directory.mkdir()
    (target_directory / "record.jsonl").write_bytes(_record().to_bytes())
    linked_directory = tmp_path / "linked-records"
    linked_directory.symlink_to(target_directory, target_is_directory=True)
    with pytest.raises(TrackingError, match="ancestor"):
        parse_tracking_record(linked_directory / "record.jsonl")
    fifo = tmp_path / "record.fifo"
    os.mkfifo(fifo)
    with pytest.raises(TrackingError, match="regular"):
        parse_tracking_record(fifo)
    socket_path = tmp_path / "record.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(socket_path))
        with pytest.raises(TrackingError, match="path"):
            parse_tracking_record(socket_path)
    with pytest.raises(TrackingError, match="regular"):
        parse_tracking_record(tmp_path)
    with pytest.raises(TrackingError, match="regular"):
        parse_tracking_record(Path("/dev/null"))


def test_environment_digest_uses_only_the_eight_field_ga1_key() -> None:
    record = _record()
    payload = _json_payload(record)
    assert payload["environment"]["digest"] == _ga1_digest(payload)  # type: ignore[index]


def test_ga1_digest_excludes_run_artifact_toolchain_and_diagnostic_provenance() -> None:
    baseline = _json_payload(_record())
    changed = copy.deepcopy(baseline)
    changed["workflow"]["run_id"] = "654321"  # type: ignore[index]
    changed["workflow"]["run_url"] = (  # type: ignore[index]
        "https://github.com/Vzlentin/calibre/actions/runs/654321"
    )
    changed["result_artifact"]["id"] = "987654"  # type: ignore[index]
    changed["result_artifact"]["digest"] = "9" * 64  # type: ignore[index]
    changed["environment"]["facts"]["cpu_model"] = "different cpu"  # type: ignore[index]
    changed["environment"]["facts"]["runner_image"] = "different image"  # type: ignore[index]
    changed["evidence"]["session"]["id"] = "8" * 64  # type: ignore[index]
    changed["objective"]["total_cost"] = 99.0  # type: ignore[index]
    assert _ga1_digest(changed) == _ga1_digest(baseline)

    ga1_changed = copy.deepcopy(baseline)
    ga1_changed["evidence"]["config"]["digest"] = "7" * 64  # type: ignore[index]
    assert _ga1_digest(ga1_changed) != _ga1_digest(baseline)


def test_verify_requires_comparable_zero_delta_against_latest_history() -> None:
    import newcalibre.protocols.vn2._tracking_validation as validation

    prior = _record()
    current = _record(candidate="c" * 40)
    validation.require_exact_recomputation(current, (prior,))
    assert (
        validation.resolve_tracking_history_mode(
            mode="verify",
            require_history="true",
            history=(prior,),
        )
        == "compare"
    )
    assert (
        validation.resolve_tracking_history_mode(
            mode="verify",
            require_history="false",
            history=(),
        )
        == "skip"
    )
    assert (
        validation.resolve_tracking_history_mode(
            mode="mint",
            require_history="true",
            history=(),
        )
        == "mint"
    )
    with pytest.raises(TrackingError, match="requires non-empty canonical tracking history"):
        validation.resolve_tracking_history_mode(
            mode="verify",
            require_history="true",
            history=(),
        )
    with pytest.raises(TrackingError, match="objective"):
        validation.require_exact_recomputation(_record(total_cost=4.0), (prior,))


def test_canonical_record_round_trip_and_history_duplicate_refusal() -> None:
    record = _record()
    parsed = parse_tracking_record(record.to_bytes())
    assert parsed.to_bytes() == record.to_bytes()
    with pytest.raises(TrackingError, match="duplicate identity"):
        parse_tracking_history(record.to_bytes() + record.to_bytes())
    with pytest.raises(TrackingError, match="derived from validated evidence"):
        write_proposal_record(parsed, Path("artifacts") / "parsed.jsonl")


def test_comparison_is_informational_and_exact_key_mismatches_have_no_delta() -> None:
    current = _record(total_cost=4.0)
    prior = _record(total_cost=3.0)
    comparison = compare_tracking_records(current, prior)
    assert comparison.comparable
    changed = _record(total_cost=4.0, candidate="c" * 40)
    changed_comparison = compare_tracking_records(changed, prior)
    assert changed_comparison.comparable
    assert changed_comparison.total_cost_delta == 1.0

    mismatch_payload = _json_payload(_record())
    mismatch_payload["evidence"]["config"]["digest"] = "7" * 64  # type: ignore[index]
    mismatch_payload["environment"]["digest"] = _ga1_digest(mismatch_payload)  # type: ignore[index]
    identity_preimage = {
        "artifact_kind": "vn2-gate-a-results",
        "candidate_sha": mismatch_payload["subject"]["candidate_sha"],  # type: ignore[index]
        "config_digest": mismatch_payload["evidence"]["config"]["digest"],  # type: ignore[index]
        "environment_digest": mismatch_payload["environment"]["digest"],  # type: ignore[index]
        "input_inventory_digest": mismatch_payload["evidence"]["input_inventory"][  # type: ignore[index]
            "digest"
        ],
    }
    mismatch_payload["identity"] = hashlib.sha256(
        canonical_json_bytes(identity_preimage, path="tracking identity")
    ).hexdigest()
    mismatch = VN2TrackingRecord._from_evidence(mismatch_payload)
    mismatch_comparison = compare_tracking_records(mismatch, _record())
    assert not mismatch_comparison.comparable
    assert mismatch_comparison.mismatched_fields == ("config_digest",)
    assert mismatch_comparison.total_cost_delta is None
    assert mismatch_comparison.cost_jump_detected is None


def test_append_idempotency_and_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    record = _record()
    assert decide_append(record, ()).action == "append"
    assert decide_append(record, (record,)).action == "noop"
    root = tmp_path / "newcalibre"
    import newcalibre.protocols.vn2._tracking_persistence as persistence

    monkeypatch.setattr(persistence, "_TRUSTED_PROJECT_ROOT", root)
    (root / "artifacts").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'newcalibre'\n")
    path = root / "artifacts" / "proposal.jsonl"
    assert write_proposal_record(record, path)
    assert not write_proposal_record(record, path)
    with pytest.raises(TrackingError, match="conflicts"):
        write_proposal_record(_record(total_cost=4.0), path)


def test_strict_codec_refuses_pretty_json_and_crlf() -> None:
    record = _record()
    body = record.to_bytes()[:-1]
    assert parse_tracking_record(body + b"\n").to_bytes() == record.to_bytes()
    with pytest.raises(TrackingError, match="canonical"):
        parse_tracking_record((record.to_json().rstrip("\n").replace(":", ": ") + "\n").encode())
    with pytest.raises(TrackingError, match="LF"):
        parse_tracking_record(body + b"\r\n")


def test_tracking_record_owns_nested_values_and_exposes_no_aliases() -> None:
    payload = _json_payload(_record())
    with pytest.raises(TrackingError, match="construction is private"):
        VN2TrackingRecord(payload)
    record = VN2TrackingRecord._from_evidence(payload)
    payload["subject"]["repository"] = "evil"  # type: ignore[index]
    assert record.payload["subject"]["repository"] == "Vzlentin/calibre"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.payload["subject"]["repository"] = "evil"  # type: ignore[index]
    import newcalibre.protocols.vn2.tracking as tracking

    assert not hasattr(tracking, "TrackingRecord")
    assert not hasattr(tracking, "append_decision")
    assert not hasattr(tracking, "build_proposed_record")
    assert not hasattr(tracking, "compare_records")
    assert not hasattr(tracking, "write_tracking_record")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"{}\n\n",
        b"\xef\xbb\xbf{}\n",
        b"\xff{}\n",
        b"{}",
    ],
)
def test_tracking_codec_rejects_malformed_line_shapes(payload: bytes) -> None:
    with pytest.raises(TrackingError):
        parse_tracking_record(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("subject"),
        lambda value: value.__setitem__("unknown", None),
        lambda value: value["environment"].__setitem__("facts", []),
        lambda value: value["objective"].__setitem__("total_cost", math.nan),
        lambda value: value["evidence"]["promoted_capture"].__setitem__(
            "artifact_name", "oracle-capture-" + "c" * 40
        ),
        lambda value: value["workflow"].__setitem__(
            "definition_ref", "evil.example/workflow.yml@main"
        ),
    ],
)
def test_tracking_constructor_refuses_schema_and_fact_corruption(mutation) -> None:
    payload = _json_payload(_record())
    mutation(payload)
    with pytest.raises(TrackingError):
        VN2TrackingRecord(payload)


def test_tracking_history_and_publication_paths_are_successor_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "newcalibre"
    import newcalibre.protocols.vn2._tracking_persistence as persistence

    monkeypatch.setattr(persistence, "_TRUSTED_PROJECT_ROOT", root)
    (root / "artifacts").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'newcalibre'\n")
    monkeypatch.chdir(root)
    record = _record()
    relative = Path("artifacts") / "nested" / "proposal.jsonl"
    assert write_proposal_record(record, relative)
    with pytest.raises(TrackingError):
        write_proposal_record(record, tmp_path / "outside" / "proposal.jsonl")
    with pytest.raises(TrackingError, match="tracked history"):
        write_proposal_record(record, root / "stage3" / "tracking" / "series.jsonl")


def test_builder_translates_domain_validator_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import newcalibre.protocols.vn2._tracking_projection as projection

    def fail(*args: object, **kwargs: object) -> object:
        raise VN2ResultError("fixture validator failure")

    monkeypatch.setattr(projection, "validate_vn2_result_bundle", fail)
    with pytest.raises(TrackingError) as caught:
        build_tracking_record(
            Path("result"),
            Path("captures"),
            candidate_sha=CANDIDATE,
            definition_ref="Vzlentin/calibre/.github/workflows/newcalibre.yml@main",
            definition_sha=WORKFLOW,
            run_id=RUN_ID,
            run_url=f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
            result_artifact_id="789012",
            result_artifact_name=f"vn2-acceptance-{CANDIDATE}",
            result_artifact_digest=DIGEST,
        )
    assert isinstance(caught.value.__cause__, VN2ResultError)
