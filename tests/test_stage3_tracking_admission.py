"""Exercise the Stage 3 tracking-artifact admission boundary."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import warnings
import zipfile
from pathlib import Path

import pytest
import yaml

from tests.infra import load_script_module

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "stage3_tracking_admission.py"
WORKFLOW = ROOT / ".github" / "workflows" / "s3-activation-gate.yml"

stage3_tracking_admission = load_script_module(SCRIPT)

CANDIDATE_SHA = "c" * 40
RUN_ID = 123
RUN_URL = f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}"
REPOSITORY_ID = 456


def _write_zip(
    path: Path,
    entries: list[tuple[str, bytes, int | None]],
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bundle.writestr(info, payload)


def _run_metadata(**updates: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "id": RUN_ID,
        "event": "workflow_dispatch",
        "path": ".github/workflows/newcalibre.yml",
        "head_branch": "main",
        "head_sha": CANDIDATE_SHA,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "html_url": RUN_URL,
        "repository": {"full_name": "Vzlentin/calibre", "id": REPOSITORY_ID},
    }
    metadata.update(updates)
    return metadata


def _artifact_metadata(
    *,
    artifact_id: int,
    name: str,
    digest: str,
    **updates: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "id": artifact_id,
        "name": name,
        "digest": f"sha256:{digest}",
        "expired": False,
        "workflow_run": {
            "id": RUN_ID,
            "head_branch": "main",
            "head_sha": CANDIDATE_SHA,
            "repository_id": REPOSITORY_ID,
            "head_repository_id": REPOSITORY_ID,
        },
    }
    metadata.update(updates)
    return metadata


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _artifact_cli_args(
    tmp_path: Path,
    *,
    label: str,
    artifact_id: int,
    archive: Path,
) -> list[str]:
    name = (
        f"vn2-acceptance-{CANDIDATE_SHA}"
        if label == "result"
        else f"vn2-tracking-proposal-{CANDIDATE_SHA}"
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata = tmp_path / f"{label}-metadata.json"
    _write_json(
        metadata,
        _artifact_metadata(artifact_id=artifact_id, name=name, digest=digest),
    )
    return [
        f"--{label}-artifact-id",
        str(artifact_id),
        f"--{label}-artifact-name",
        name,
        f"--{label}-artifact-digest",
        digest,
        f"--{label}-artifact-metadata",
        str(metadata),
        f"--{label}-archive",
        str(archive),
        f"--{label}-destination",
        str(tmp_path / label),
    ]


def test_cli_validates_and_atomically_extracts_both_artifacts(tmp_path: Path) -> None:
    run_metadata = tmp_path / "run-metadata.json"
    _write_json(run_metadata, _run_metadata())
    result_archive = tmp_path / "result.zip"
    proposal_archive = tmp_path / "proposal.zip"
    _write_zip(
        result_archive,
        [
            ("reports/", b"", stat.S_IFDIR | 0o700),
            ("reports/result.json", b"{}\n", stat.S_IFREG | 0o600),
        ],
    )
    _write_zip(
        proposal_archive,
        [("proposed-record.jsonl", b'{"record":1}\n', stat.S_IFREG | 0o600)],
    )

    args = [
        "admit",
        "--candidate-sha",
        CANDIDATE_SHA,
        "--run-id",
        str(RUN_ID),
        "--run-url",
        RUN_URL,
        "--run-metadata",
        str(run_metadata),
        *_artifact_cli_args(tmp_path, label="result", artifact_id=1001, archive=result_archive),
        *_artifact_cli_args(tmp_path, label="proposal", artifact_id=1002, archive=proposal_archive),
    ]

    assert stage3_tracking_admission.main(args) == 0
    assert (tmp_path / "result" / "reports" / "result.json").read_bytes() == b"{}\n"
    assert (tmp_path / "proposal" / "proposed-record.jsonl").read_bytes() == (b'{"record":1}\n')
    assert not list(tmp_path.glob(".stage3-admission-*"))


def test_cli_failure_publishes_neither_artifact_and_cleans_stages(
    tmp_path: Path,
) -> None:
    run_metadata = tmp_path / "run-metadata.json"
    _write_json(run_metadata, _run_metadata())
    result_archive = tmp_path / "result.zip"
    proposal_archive = tmp_path / "proposal.zip"
    _write_zip(
        result_archive,
        [("result.json", b"{}\n", stat.S_IFREG | 0o600)],
    )
    proposal_archive.write_bytes(b"PK\x03\x04")
    args = [
        "admit",
        "--candidate-sha",
        CANDIDATE_SHA,
        "--run-id",
        str(RUN_ID),
        "--run-url",
        RUN_URL,
        "--run-metadata",
        str(run_metadata),
        *_artifact_cli_args(tmp_path, label="result", artifact_id=1001, archive=result_archive),
        *_artifact_cli_args(tmp_path, label="proposal", artifact_id=1002, archive=proposal_archive),
    ]

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(args)

    assert error.value.code == 1
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / "proposal").exists()
    assert not list(tmp_path.glob(".stage3-admission-*"))


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "dir\\file",
        "/",
        "dir//file",
        "./file",
        "dir/./file",
        "dir/../file",
    ],
)
def test_safe_zip_rejects_unsafe_path_components(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive = tmp_path / "unsafe.zip"
    _write_zip(archive, [(unsafe_name, b"hostile", stat.S_IFREG | 0o600)])
    destination = tmp_path / "accepted"

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="unsafe path"):
        stage3_tracking_admission.extract_safe_zip(archive, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".stage3-admission-*"))


def test_safe_zip_rejects_duplicate_paths(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    _write_zip(
        archive,
        [
            ("same.txt", b"first", stat.S_IFREG | 0o600),
            ("same.txt", b"second", stat.S_IFREG | 0o600),
        ],
    )

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="duplicate path"):
        stage3_tracking_admission.extract_safe_zip(archive, tmp_path / "accepted")

    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize(
    "entries",
    [
        [
            ("node", b"file", stat.S_IFREG | 0o600),
            ("node/child", b"child", stat.S_IFREG | 0o600),
        ],
        [
            ("node/child", b"child", stat.S_IFREG | 0o600),
            ("node", b"file", stat.S_IFREG | 0o600),
        ],
    ],
)
def test_safe_zip_rejects_file_directory_prefix_collisions(
    tmp_path: Path,
    entries: list[tuple[str, bytes, int | None]],
) -> None:
    archive = tmp_path / "collision.zip"
    _write_zip(archive, entries)

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="prefix collision"):
        stage3_tracking_admission.extract_safe_zip(archive, tmp_path / "accepted")

    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        (stat.S_IFLNK | 0o600, "special or mismatched mode"),
        (stat.S_IFIFO | 0o600, "special or mismatched mode"),
        (stat.S_IFREG | 0o700, "executable mode"),
    ],
)
def test_safe_zip_rejects_symlink_special_and_executable_modes(
    tmp_path: Path,
    mode: int,
    message: str,
) -> None:
    archive = tmp_path / "mode.zip"
    _write_zip(archive, [("entry", b"payload", mode)])

    with pytest.raises(stage3_tracking_admission.AdmissionError, match=message):
        stage3_tracking_admission.extract_safe_zip(archive, tmp_path / "accepted")

    assert not (tmp_path / "accepted").exists()


def test_safe_zip_rejects_oversized_total_expansion(tmp_path: Path) -> None:
    archive = tmp_path / "oversized.zip"
    _write_zip(archive, [("entry", b"1234", stat.S_IFREG | 0o600)])

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="expands beyond"):
        stage3_tracking_admission.extract_safe_zip(
            archive,
            tmp_path / "accepted",
            max_total_size=3,
        )

    assert not (tmp_path / "accepted").exists()


def test_metadata_loader_accepts_canonical_json_object(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(b'{"id":1,"nested":{"ok":true}}')

    assert stage3_tracking_admission.load_promotion_metadata(metadata, name="run metadata") == {
        "id": 1,
        "nested": {"ok": True},
    }


@pytest.mark.parametrize("payload", [b"not JSON", b"[]"])
def test_metadata_loader_rejects_malformed_metadata(
    tmp_path: Path,
    payload: bytes,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(payload)

    with pytest.raises(ValueError, match="metadata"):
        stage3_tracking_admission.load_promotion_metadata(metadata, name="run metadata")


def test_metadata_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(b'{"id":1,"id":2}')

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        stage3_tracking_admission.load_promotion_metadata(metadata, name="run metadata")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_metadata_loader_rejects_nonfinite_json(
    tmp_path: Path,
    constant: str,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(f'{{"value":{constant}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        stage3_tracking_admission.load_promotion_metadata(metadata, name="run metadata")


def test_metadata_loader_rejects_oversized_json(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(b'{"value":"' + b"x" * (4 * 1024 * 1024) + b'"}')

    with pytest.raises(ValueError, match="exceeds the maximum size"):
        stage3_tracking_admission.load_promotion_metadata(metadata, name="run metadata")


@pytest.mark.parametrize("link_kind", ["leaf", "ancestor"])
def test_metadata_loader_rejects_symlink_leaf_and_ancestor(
    tmp_path: Path,
    link_kind: str,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    metadata = real_root / "metadata.json"
    metadata.write_bytes(b'{"id":1}')
    if link_kind == "leaf":
        hostile_path = tmp_path / "metadata.json"
        hostile_path.symlink_to(metadata)
    else:
        hostile_root = tmp_path / "metadata-root"
        hostile_root.symlink_to(real_root, target_is_directory=True)
        hostile_path = hostile_root / metadata.name

    with pytest.raises(ValueError, match="every ancestor"):
        stage3_tracking_admission.load_promotion_metadata(hostile_path, name="run metadata")


@pytest.mark.parametrize("special", ["fifo", "directory", "socket"])
def test_metadata_loader_rejects_special_files_without_blocking(
    tmp_path: Path,
    special: str,
) -> None:
    metadata = tmp_path / "metadata.json"
    bound_socket: socket.socket | None = None
    try:
        if special == "fifo":
            os.mkfifo(metadata)
        elif special == "directory":
            metadata.mkdir()
        else:
            bound_socket = socket.socket(socket.AF_UNIX)
            bound_socket.bind(str(metadata))

        with pytest.raises(ValueError, match="regular non-symlink file|every ancestor"):
            stage3_tracking_admission.load_promotion_metadata(metadata, name="run metadata")
    finally:
        if bound_socket is not None:
            bound_socket.close()


@pytest.mark.parametrize("metadata_label", ["run", "result", "proposal"])
def test_cli_applies_safe_loader_to_all_metadata_paths(
    tmp_path: Path,
    metadata_label: str,
) -> None:
    run_metadata = tmp_path / "run-metadata.json"
    _write_json(run_metadata, _run_metadata())
    result_archive = tmp_path / "result.zip"
    proposal_archive = tmp_path / "proposal.zip"
    _write_zip(
        result_archive,
        [("result.json", b"{}\n", stat.S_IFREG | 0o600)],
    )
    _write_zip(
        proposal_archive,
        [("proposed-record.jsonl", b'{"record":1}\n', stat.S_IFREG | 0o600)],
    )
    args = [
        "admit",
        "--candidate-sha",
        CANDIDATE_SHA,
        "--run-id",
        str(RUN_ID),
        "--run-url",
        RUN_URL,
        "--run-metadata",
        str(run_metadata),
        *_artifact_cli_args(tmp_path, label="result", artifact_id=1001, archive=result_archive),
        *_artifact_cli_args(tmp_path, label="proposal", artifact_id=1002, archive=proposal_archive),
    ]
    metadata = (
        run_metadata if metadata_label == "run" else tmp_path / f"{metadata_label}-metadata.json"
    )
    target = tmp_path / f"real-{metadata.name}"
    metadata.replace(target)
    metadata.symlink_to(target)

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(args)

    assert error.value.code == 1
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / "proposal").exists()


def test_run_metadata_mismatch_is_rejected() -> None:
    metadata = _run_metadata(head_sha="d" * 40)

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="workflow-run metadata"):
        stage3_tracking_admission.validate_run_metadata(
            metadata,
            run_id=RUN_ID,
            candidate_sha=CANDIDATE_SHA,
            run_url=RUN_URL,
        )


def test_artifact_metadata_mismatch_is_rejected() -> None:
    metadata = _artifact_metadata(
        artifact_id=1001,
        name=f"vn2-acceptance-{CANDIDATE_SHA}",
        digest="a" * 64,
    )
    metadata["workflow_run"] = {
        **metadata["workflow_run"],
        "head_repository_id": REPOSITORY_ID + 1,
    }

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="artifact metadata"):
        stage3_tracking_admission.validate_artifact_metadata(
            metadata,
            label="result",
            artifact_id=1001,
            artifact_name=f"vn2-acceptance-{CANDIDATE_SHA}",
            receipt_digest="a" * 64,
            run_id=RUN_ID,
            candidate_sha=CANDIDATE_SHA,
            repository_id=REPOSITORY_ID,
        )


def test_wrong_archive_digest_is_rejected_before_destination_creation(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "artifact.zip"
    _write_zip(archive, [("entry", b"payload", stat.S_IFREG | 0o600)])
    destination = tmp_path / "accepted"

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="digest"):
        stage3_tracking_admission.admit_artifact_archive(
            archive,
            destination,
            expected_digest="0" * 64,
            label="result",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".stage3-admission-*"))


@pytest.mark.parametrize("corruption", ["short", "crc"])
def test_short_or_corrupt_archive_leaves_no_partial_destination(
    tmp_path: Path,
    corruption: str,
) -> None:
    archive = tmp_path / "corrupt.zip"
    if corruption == "short":
        archive.write_bytes(b"PK\x03\x04")
    else:
        _write_zip(archive, [("entry", b"payload", stat.S_IFREG | 0o600)])
        archive.write_bytes(archive.read_bytes().replace(b"payload", b"PAYLOAD", 1))
    destination = tmp_path / "accepted"

    with pytest.raises(stage3_tracking_admission.AdmissionError, match="corrupt"):
        stage3_tracking_admission.extract_safe_zip(archive, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".stage3-admission-*"))


def _repo_git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repo_git(repository, "init", "--quiet")
    _repo_git(repository, "config", "user.email", "test@example.com")
    _repo_git(repository, "config", "user.name", "Test")
    (repository / "README").write_text("base\n", encoding="utf-8")
    return repository, _commit_all(repository, "base")


def _commit_all(repository: Path, message: str) -> str:
    _repo_git(repository, "add", "-A")
    _repo_git(repository, "commit", "--quiet", "-m", message)
    return _repo_git(repository, "rev-parse", "HEAD")


def _receipt_value(candidate: str) -> dict[str, object]:
    return {
        "candidate_sha": candidate,
        "proposal_artifact": {
            "digest": "b" * 64,
            "id": "1002",
            "name": f"vn2-tracking-proposal-{candidate}",
        },
        "receipt_kind": "vn2-tracking-promotion-receipt",
        "record_sha256": "a" * 64,
        "repository": "Vzlentin/calibre",
        "result_artifact": {
            "digest": "c" * 64,
            "id": "1001",
            "name": f"vn2-acceptance-{candidate}",
        },
        "schema": 1,
        "workflow": {
            "definition_ref": ("Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main"),
            "definition_sha": candidate,
            "run_id": str(RUN_ID),
            "run_url": RUN_URL,
        },
    }


def _canonical_receipt(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_promotion(
    repository: Path,
    candidate: str,
    *,
    receipt: bytes | None = None,
) -> Path:
    tracking = repository / "stage3" / "evidence" / "tracking"
    tracking.mkdir(parents=True, exist_ok=True)
    (tracking / "series.jsonl").write_text('{"record":1}\n', encoding="utf-8")
    receipt_path = tracking / f"{candidate}-receipt.json"
    receipt_path.write_bytes(
        _canonical_receipt(_receipt_value(candidate)) if receipt is None else receipt
    )
    return receipt_path


def _inspect_args(repository: Path, base: str, head: str, output: Path) -> list[str]:
    return [
        "inspect",
        "--base-sha",
        base,
        "--head-sha",
        head,
        "--github-output",
        str(output),
        "--repository-root",
        str(repository),
    ]


def _read_outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def test_inspect_real_git_no_tracking_writes_only_absent(tmp_path: Path) -> None:
    repository, base = _init_repository(tmp_path)
    (repository / "README").write_text("ordinary change\n", encoding="utf-8")
    head = _commit_all(repository, "ordinary")
    output = tmp_path / "github-output"

    assert stage3_tracking_admission.main(_inspect_args(repository, base, head, output)) == 0

    assert output.read_text(encoding="utf-8") == "present=false\n"


def test_inspect_ignores_tracking_root_prefix_sibling(tmp_path: Path) -> None:
    repository, base = _init_repository(tmp_path)
    sibling = repository / "stage3" / "evidence" / "tracking-escape"
    sibling.mkdir(parents=True)
    (sibling / "series.jsonl").write_text("not tracking\n", encoding="utf-8")
    head = _commit_all(repository, "sibling")
    output = tmp_path / "github-output"

    assert stage3_tracking_admission.main(_inspect_args(repository, base, head, output)) == 0

    assert output.read_text(encoding="utf-8") == "present=false\n"


def test_inspect_real_git_valid_promotion_writes_all_bindings(tmp_path: Path) -> None:
    repository, base = _init_repository(tmp_path)
    receipt_path = _write_promotion(repository, base)
    head = _commit_all(repository, "promotion")
    output = tmp_path / "github-output"

    assert stage3_tracking_admission.main(_inspect_args(repository, base, head, output)) == 0

    assert _read_outputs(output) == {
        "present": "true",
        "receipt": receipt_path.relative_to(repository).as_posix(),
        "candidate": base,
        "workflow_ref": ("Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main"),
        "workflow_sha": base,
        "run_id": str(RUN_ID),
        "run_url": RUN_URL,
        "result_id": "1001",
        "result_name": f"vn2-acceptance-{base}",
        "result_digest": "c" * 64,
        "proposal_id": "1002",
        "proposal_name": f"vn2-tracking-proposal-{base}",
        "proposal_digest": "b" * 64,
    }


def test_inspect_rejects_exact_tracking_root_file(tmp_path: Path) -> None:
    repository, _ = _init_repository(tmp_path)
    tracking_root = repository / "stage3" / "evidence" / "tracking"
    tracking_root.parent.mkdir(parents=True)
    tracking_root.write_text("base\n", encoding="utf-8")
    base = _commit_all(repository, "tracking root")
    tracking_root.write_text("head\n", encoding="utf-8")
    head = _commit_all(repository, "change tracking root")

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


def test_inspect_rejects_mixed_promotion_paths(tmp_path: Path) -> None:
    repository, base = _init_repository(tmp_path)
    _write_promotion(repository, base)
    (repository / "README").write_text("mixed\n", encoding="utf-8")
    head = _commit_all(repository, "mixed")

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


@pytest.mark.parametrize("kind", ["malformed", "noncanonical", "duplicate"])
def test_inspect_rejects_hostile_receipt_bytes(tmp_path: Path, kind: str) -> None:
    repository, base = _init_repository(tmp_path)
    value = _receipt_value(base)
    if kind == "malformed":
        receipt = b"{\n"
    elif kind == "noncanonical":
        receipt = json.dumps(value, indent=2).encode("utf-8") + b"\n"
    else:
        canonical = _canonical_receipt(value)
        receipt = canonical.replace(
            b'{"candidate_sha":',
            b'{"candidate_sha":"' + base.encode() + b'","candidate_sha":',
            1,
        )
    _write_promotion(repository, base, receipt=receipt)
    head = _commit_all(repository, kind)

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


@pytest.mark.parametrize(
    "binding",
    ["candidate", "workflow-sha", "result-name", "run-url", "artifact-id"],
)
def test_inspect_rejects_wrong_receipt_bindings(
    tmp_path: Path,
    binding: str,
) -> None:
    repository, base = _init_repository(tmp_path)
    value = _receipt_value(base)
    if binding == "candidate":
        value["candidate_sha"] = "d" * 40
    elif binding == "workflow-sha":
        value["workflow"]["definition_sha"] = "d" * 40  # type: ignore[index]
    elif binding == "result-name":
        value["result_artifact"]["name"] = "wrong"  # type: ignore[index]
    elif binding == "run-url":
        value["workflow"]["run_url"] = f"{RUN_URL}/wrong"  # type: ignore[index]
    else:
        value["proposal_artifact"]["id"] = "1001"  # type: ignore[index]
    _write_promotion(repository, base, receipt=_canonical_receipt(value))
    head = _commit_all(repository, binding)

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


@pytest.mark.parametrize("entry_kind", ["symlink", "executable", "tree"])
def test_inspect_rejects_non_regular_receipt_git_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    repository, base = _init_repository(tmp_path)
    receipt_path = _write_promotion(repository, base)
    if entry_kind == "symlink":
        receipt_path.unlink()
        receipt_path.symlink_to(repository / "README")
    elif entry_kind == "executable":
        receipt_path.chmod(0o755)
    else:
        receipt_path.unlink()
        receipt_path.mkdir()
        (receipt_path / "payload").write_text("tree\n", encoding="utf-8")
    head = _commit_all(repository, entry_kind)

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


def test_inspect_rejects_non_utf8_changed_path(tmp_path: Path) -> None:
    repository, base = _init_repository(tmp_path)
    raw_path = os.fsencode(repository) + b"/stage3/evidence/tracking/\xff"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    descriptor = os.open(raw_path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, b"hostile\n")
    finally:
        os.close(descriptor)
    head = _commit_all(repository, "non utf8")

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


def test_inspect_disables_rename_detection_for_path_confinement(tmp_path: Path) -> None:
    repository, _ = _init_repository(tmp_path)
    tracking = repository / "stage3" / "evidence" / "tracking"
    tracking.mkdir(parents=True)
    (tracking / "series.jsonl").write_text("base\n", encoding="utf-8")
    old_receipt = tracking / "old-receipt.json"
    old_receipt.write_text("unchanged rename source\n", encoding="utf-8")
    base = _commit_all(repository, "old receipt")
    (tracking / "series.jsonl").write_text("head\n", encoding="utf-8")
    old_receipt.rename(tracking / f"{base}-receipt.json")
    head = _commit_all(repository, "rename")

    with pytest.raises(SystemExit) as error:
        stage3_tracking_admission.main(
            _inspect_args(repository, base, head, tmp_path / "github-output")
        )

    assert error.value.code == 1


def test_activation_workflow_uses_the_admission_script_contract() -> None:
    raw = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(raw)
    steps = workflow["jobs"]["s3-activation-gate"]["steps"]
    checkout = next(step for step in steps if step["name"] == "Check out the immutable PR base")
    detection = next(step for step in steps if step["name"] == "Detect tracking tree changes")
    setup = next(step for step in steps if step["name"] == "Install trusted base validator")
    sync = next(step for step in steps if step["name"] == "Sync trusted base dependencies")
    promotion = next(
        step for step in steps if step["name"] == "Inspect and confine tracking promotion"
    )
    download = next(
        step
        for step in steps
        if step["name"] == "Download live same-run artifact metadata and archives"
    )
    validate = next(
        step for step in steps if step["name"] == "Validate live artifacts and exact append bytes"
    )

    assert checkout["with"]["fetch-depth"] == 1
    assert "--quiet --no-renames" in detection["run"]
    assert "stage3/evidence/tracking" in detection["run"]
    assert steps.index(detection) < steps.index(setup) < steps.index(sync) < steps.index(promotion)
    assert setup["if"] == "steps.tracking-change.outputs.present == 'true'"
    assert sync["if"] == "steps.tracking-change.outputs.present == 'true'"
    assert promotion["if"] == "steps.tracking-change.outputs.present == 'true'"
    assert "--project newcalibre --locked --no-sync python" in promotion["run"]
    assert ".github/scripts/stage3_tracking_admission.py inspect" in promotion["run"]
    assert '--github-output "$GITHUB_OUTPUT"' in promotion["run"]
    assert "--project newcalibre --locked --no-sync python" in download["run"]
    assert ".github/scripts/stage3_tracking_admission.py admit" in download["run"]
    assert "zipfile" not in download["run"]

    output_names = (
        "run_id",
        "run_url",
        "result_id",
        "result_name",
        "result_digest",
        "proposal_id",
        "proposal_name",
        "proposal_digest",
        "workflow_ref",
        "workflow_sha",
    )
    for name in output_names:
        expected = f"${{{{ steps.promotion.outputs.{name} }}}}"
        if name.startswith("proposal_"):
            assert expected in download["env"].values()
        elif name.startswith("workflow_"):
            assert expected in validate["env"].values()
        else:
            assert expected in download["env"].values() or expected in validate["env"].values()

    assert raw.count("astral-sh/setup-uv@") == 1
    assert raw.count("uv sync --project newcalibre --locked --group dev") == 1
    assert "steps.receipt.outputs" not in raw
    assert "<<'PY'" not in raw
    assert "python3 - <<" not in raw
